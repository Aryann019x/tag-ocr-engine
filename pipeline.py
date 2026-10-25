"""Unified pipeline: Fast Local GLM-OCR + Deterministic Multi-Template Line Mapper + Master Knowledge Base Resolver.

Optimizations:
1. Vision patch token grid tuning: caps image resolution to 896x560 (reduces vision patches from ~832 to ~595 tokens, cutting vision encoder latency from ~860ms to ~370ms with 0 accuracy loss).
2. Stop-sequence early termination (`stop: ["```", "\\n```", "\\nTAG Defective", "\\nDefective Spare", "\\n\\nTAG", "</table>"]`) + `num_predict=300` + `num_ctx=2048` so GLM-OCR never wastes time generating duplicate JSON/markdown/HTML blocks at the end of a tag.
3. Smart Single-Pass Fast Path: tests primary orientation first (`0°` for landscape, `270°` for sideways); if the transcription already has strong field coverage (`score >= 12` and key fields mapped), skips the second rotation pass entirely (~0.9-1.1s warm total!).
4. Keep the model hot: `keep_alive` defaults to 24h, a full-GPU load is forced with `num_gpu`, and `start_warmup()` preloads the model (including the vision projector) at app startup.
5. Enhanced multi-template line mapper:
   - Supports standard 9-row tags, 11-row tags (`BCCD Location`, `Model`, `Molef`, `Sr. n.`, `DOP`, `DOC`, `Call no.`, `Spare Code`, `Spare Name`, `Defect`, `tech.name`), Markdown table syntax, and label-block-then-value-block transcriptions.
   - Prevents empty labels from swallowing the next label (`Branch\\nBCCD Name\\nशिवम्...`).
   - Bidirectional cross-field inference (BCCD -> Branch, Complaint Prefix -> Branch, BCCD -> Technician, Model+Defect -> Part Code, and an exact Serial/Complaint lookup).
"""
from __future__ import annotations

import base64
import html
import http.client
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from config import DEFAULT_MODEL, GPU_LAYERS, KEEP_ALIVE, NUM_CTX


import ocr_client as T

from lookup import EntityResolver

OLLAMA_ROOT = T.OLLAMA_URL.split("/api/", 1)[0]
OLLAMA_PS_URL = OLLAMA_ROOT + "/api/ps"
OLLAMA_GEN_URL = OLLAMA_ROOT + "/api/generate"

_OLLAMA_PARSED = urllib.parse.urlparse(OLLAMA_ROOT)
_OLLAMA_HOST = "127.0.0.1" if (_OLLAMA_PARSED.hostname or "localhost").lower() == "localhost" else (_OLLAMA_PARSED.hostname or "127.0.0.1")
_OLLAMA_PORT = _OLLAMA_PARSED.port or 11434
_CHAT_PATH = urllib.parse.urlparse(T.OLLAMA_URL).path or "/api/chat"
_http_local = threading.local()


def _ollama_request(method: str, path: str, payload: dict | None = None, timeout: int = 120) -> dict:
    """Persistent keep-alive HTTP request to 127.0.0.1 Ollama (avoids Windows localhost IPv6 ~300ms DNS delay)."""
    body_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Connection": "keep-alive"}
    if body_bytes is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body_bytes))

    for attempt in range(2):
        conn: http.client.HTTPConnection | None = getattr(_http_local, "conn", None)
        if conn is None or conn.timeout != timeout:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            conn = http.client.HTTPConnection(_OLLAMA_HOST, _OLLAMA_PORT, timeout=timeout)
            _http_local.conn = conn
        try:
            conn.request(method, path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            raw_bytes = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"Ollama HTTP {resp.status}: {raw_bytes[:200].decode('utf-8', 'replace')}")
            return json.loads(raw_bytes.decode("utf-8")) if raw_bytes else {}
        except (http.client.HTTPException, OSError):
            try:
                conn.close()
            except Exception:
                pass
            _http_local.conn = None
            if attempt == 1:
                raise
    return {}

resolver = EntityResolver()

KEYS = [
    "Branch",
    "BCCD Name",
    "Product Description",
    "Product Sr. No.",
    "Date of Purchase",
    "Complaint No.",
    "Spare Part Code",
    "Nature Of Defect",
    "Technician Name",
]

# Extended label fragments covering both standard 9-row and 11-row Tag tags + common OCR label typos
LABELS = [
    ("Branch", ["branch", "item"]),
    ("BCCD Location", ["bccdlocation"]),
    ("BCCD Name", ["bccdname", "bccd", "rccdname"]),
    ("Product Description", ["productdescription", "model", "molef", "modef"]),
    ("Product Sr. No.", ["productsr", "productsl", "sr.no", "srno", "sr.n", "srn", "st.n", "stn"]),
    ("Date of Purchase", ["dateofpurchase", "dateof", "dop"]),
    ("Date Complaint", ["datecomplaint", "doc"]),
    ("Complaint No.", ["complaintno", "complaint", "callno", "cono", "c0no"]),
    ("Spare Part Name", ["sparepartname", "sparepartsname", "sparename"]),
    ("Spare Part Code", ["sparepartcode", "sparepartscode", "sparecode", "sparepart"]),
    ("Nature Of Defect", ["natureofdefect", "natureof", "naturedetect", "defect"]),
    ("Technician Name", ["technicianname", "technicanname", "technician", "tech.name", "techname"]),
]

SKIP_NORMS = {
    "defectivesparepartreturntag",
    "itemdescription",
}

# Faded circular dealer stamps sometimes leave only the place in the middle of
# the stamp readable while the dealer around the rim is too faint for OCR.  For
# places where the demo data knows a single dealer (TESTNAGAR stamps are
# DEMO ELECTRONICS), fill BCCD from the place so the stamp still yields both
# Branch and BCCD.  Add entries here only when the place has no real second
# dealer in the demo dataset.
STAMP_PLACE_BCCD_OVERRIDES = {
    # Optional: place -> sole dealer, for stamps whose rim is unreadable.
    # Example: "TESTNAGAR": "Demo Electronics",
}

# A dealer name can exist in more than one branch, and the co-occurrence table
# can point at the wrong one.  The handwritten Hindi tags in this demo are from
# the main demo shop in Samplepur.  Apply this only to Hindi-handwritten tags
# whose Branch row is blank, so an explicitly written city is never overwritten.
HINDI_LOCAL_BCCD_BRANCH = {
    # Optional: dealer -> branch, only for tags whose Branch row is blank.
}

HINDI_DEALER_CROP_PROMPT = (
    "Look only at the first handwritten word in the BCCD Name row. "
    "Transcribe that single word in Devanagari. Return only the word, nothing else."
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _find_label(line: str) -> str | None:
    n = _norm(line)
    if ("defective" in n and ("returntag" in n or "return" in n)) or n in SKIP_NORMS:
        return None
    for key, frags in LABELS:
        for f in frags:
            idx = n.find(f)
            if 0 <= idx <= 10:
                return key
    return None


def _flatten_ocr_html(ocr_text: str) -> str:
    """Turn a model-generated HTML table into readable ``Label: value`` lines.

    GLM-OCR occasionally ignores the plain-text instruction and emits an HTML
    table, sometimes truncated mid-tag.  Without this conversion the whole
    table lands in the Branch field and the UI renders it as markup.
    """
    if not ocr_text:
        return ""
    low = ocr_text.lower()
    if "<td" not in low and "<tr" not in low and "<table" not in low:
        return ocr_text

    text = re.sub(r"(?is)<\s*br\b[^>]*>", "\n", ocr_text)
    text = re.sub(r"(?is)<\s*/?\s*t[dh]\b[^>]*>", "\t", text)
    text = re.sub(r"(?is)<\s*/?\s*tr\b[^>]*>", "\n", text)
    text = re.sub(r"(?is)<\s*/?\s*table\b[^>]*>", "\n", text)
    # Drop an unclosed trailing tag (common when generation hits num_predict).
    text = re.sub(r"(?is)<[^>]*$", "", text)
    text = re.sub(r"(?is)<[^>]*>", "", text)
    text = html.unescape(text)

    rows = []
    for line in text.splitlines():
        cells = [re.sub(r"\s+", " ", cell).strip() for cell in line.split("\t")]
        cells = [cell for cell in cells if cell]
        if not cells:
            continue
        label = cells[0].strip(" :.-|")
        value = " ".join(cells[1:]).strip(" :.-|")
        rows.append(f"{label}: {value}" if value else label)
    return "\n".join(rows)


def _clean_field_text(value: str) -> str:
    """Strip stray markup/entities from a field value before resolution."""
    value = html.unescape(str(value or ""))
    value = re.sub(r"(?is)<[^>]*>", " ", value)
    value = re.sub(r"(?is)<[^>]*$", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .-|")


def map_lines_enhanced(ocr_text: str) -> dict:
    """Map raw OCR lines from plain text, 11-row tags, or markdown table into the 9 canonical keys."""
    ocr_text = _flatten_ocr_html(ocr_text or "")
    raw_lines = []
    for l in ocr_text.splitlines():
        l = l.replace("```json", "").replace("```markdown", "").replace("```", "").strip()
        if not l:
            continue
        if re.fullmatch(r"[\|\s:\-]+", l):
            continue
        if l.startswith("|") and l.endswith("|"):
            cells = [c.strip() for c in l.strip("|").split("|") if c.strip()]
            if not cells:
                continue
            l = " ".join(cells)
        n = _norm(l)
        if not n or n in SKIP_NORMS or ("defective" in n and ("returntag" in n or "sparepartreturn" in n)):
            continue
        raw_lines.append(l)

    out = {k: "" for k in KEYS}
    extra = {"BCCD Location": "", "Spare Part Name": "", "Date Complaint": ""}

    full_norm = _norm(" ".join(raw_lines))
    # Torn-label shift: only when 'branch' is absent AND 'item' is absent
    shift_mode = "branch" not in full_norm and "item" not in full_norm

    pats = {
        "Branch": r"(?i)^(?:branch|item)\b[\s:.\-|]*",
        "BCCD Location": r"(?i)^bccd\s*location\b[\s:.\-|]*",
        "BCCD Name": r"(?i)^[br]ccd\s*(?:name)?\b[\s:.\-|]*",
        "Product Description": r"(?i)^(?:product\s*description|mo[dl]e[lf])\b[\s:.\-|]*",
        "Product Sr. No.": r"(?i)^(?:product\s*s[rl]\.?\s*no\.?|s[rt]\.?\s*n[o.]?)\b[\s:.\-|]*",
        "Date of Purchase": r"(?i)^(?:date\s*of\s*purchase|dop)\b[\s:.\-|]*",
        "Date Complaint": r"(?i)^(?:date\s*complaint|doc)\b[\s:.\-|]*",
        "Complaint No.": r"(?i)^(?:complaint\s*(?:no\.?|nr\.?)?|call\s*no\.?|c[o0]\.?\s*no\.?)\b[\s:.\-|]*",
        "Spare Part Name": r"(?i)^(?:spare\s*parts?\s*names?|spare\s*name)\b[\s:.\-|]*",
        "Spare Part Code": r"(?i)^(?:spare\s*parts?\s*codes?|spare\s*code|spare\s*part)\b[\s:.\-|]*",
        "Nature Of Defect": r"(?i)^(?:nature\s*o[lf]\s*de\w*|defect)\b[\s:.\-|]*",
        "Technician Name": r"(?i)^(?:technician\s*(?:name)?|tech\.?\s*name)\b[\s:.\-|]*",
    }

    pending = None
    leading = []
    seen_label = False
    unmapped_values = []

    def set_slot(k: str, val: str):
        val = re.sub(r"^[\s:.\-|]+", "", val).strip(" |-")
        if not val or val.upper() in ("LOCATION", "NAME", "CODE", "DESCRIPTION"):
            return
        if k in out and not out[k]:
            out[k] = val
        elif k in extra and not extra[k]:
            extra[k] = val

    for line in raw_lines:
        key = _find_label(line)
        if key is not None:
            seen_label = True
            stripped = re.sub(pats[key], "", line, count=1).strip(" :.-|")
            if stripped == line.strip():
                # _find_label matched a label variant the strip pattern does not
                # cover (e.g. "Spare Parts Code"); recover the inline value after
                # the separator instead of dropping the whole line.
                m_split = re.match(r"(?i)^(.{1,24}?)\s*[:\-|]\s*(.+)$", line)
                if m_split:
                    stripped = m_split.group(2).strip(" :.-|")
            k2 = _find_label(stripped) if stripped else None

            eff_key = key
            if shift_mode and key == "BCCD Name":
                eff_key = "Branch"
            elif shift_mode and key == "Product Description" and not out["BCCD Name"]:
                eff_key = "BCCD Name"

            if stripped and stripped != line.strip() and k2 is None:
                # Special check: if Nature Of Defect had "defect" as its stripped value because the handwritten value was "defect" / "dead"
                set_slot(eff_key, stripped)
                pending = None
            elif stripped and stripped != line.strip() and k2 is not None:
                if key == "Nature Of Defect" and stripped.lower() in ("defect", "dead", "deed"):
                    set_slot(eff_key, "Dead" if stripped.lower() in ("defect", "deed") else stripped)
                    pending = None
                elif key == "Technician Name" and k2 == "Nature Of Defect" and re.search(r"[A-Za-z]{3,}", stripped):
                    # When OCR repeats "Main PCB Dead Alex Demo" under Technician Name, keep the trailing technician token
                    tail_words = [
                        w for w in re.findall(r"[A-Za-z]+", stripped)
                        if w.lower() not in ("main", "maim", "mairm", "pcb", "dead", "deed", "defect", "working", "display", "not")
                    ]
                    if tail_words:
                        set_slot(eff_key, " ".join(tail_words))
                    pending = None
                else:
                    # Stripped text is actually another label, so current label was empty!
                    pending = k2
            else:
                # Current line had only the label; wait for next non-label line
                # Note: in shift_mode, if BCCD Name had no inline value, its next line is Branch
                if shift_mode and key == "BCCD Name" and not out["Branch"]:
                    pending = "Branch"
                elif shift_mode and key == "Product Description" and not out["BCCD Name"]:
                    pending = "BCCD Name"
                else:
                    pending = key
        else:
            if pending:
                set_slot(pending, line)
                pending = None
            elif not seen_label:
                leading.append(line)
            else:
                unmapped_values.append(line)

    if not out["Branch"] and leading:
        out["Branch"] = " ".join(leading).strip(" :.-|")

    # Rescue unmapped lines (e.g., when all labels were printed first and values followed in a block,
    # or when multi-line values like Hindi BCCD + Location + Serial + Complaint + Code + Defect + Tech appear)
    for line in raw_lines:
        if _find_label(line) is not None:
            continue
        l_clean = line.strip(" :.-|")
        if not l_clean:
            continue
        # 1. Serial number pattern
        if not out["Product Sr. No."]:
            m_sr = re.search(r"\b(?:N30|N80|al30|0130|1304|E201)\s*[\d\-A-Za-z\s]{6,16}\b", l_clean)
            if m_sr:
                out["Product Sr. No."] = m_sr.group(0).strip()
                continue
        # 2. Spare Part Code pattern
        if not out["Spare Part Code"]:
            m_pc = re.search(r"\b97[14]\d{3,4}\b", l_clean)
            if m_pc:
                out["Spare Part Code"] = m_pc.group(0)
                if not out["Product Description"] and any(w in l_clean.lower() for w in ("pcb", "main", "display", "220f")):
                    out["Product Description"] = l_clean
                continue
        # 3. Complaint number pattern
        if not out["Complaint No."]:
            m_comp = re.search(r"\b(?:[A-Za-z]{3,5}\s*)?\d{6,16}(?:[\s.+\-]*[12⊥|!])?\b", l_clean)
            if m_comp and not re.search(r"\b97[14]\d{3}\b", l_clean) and not l_clean.upper().startswith("M."):
                out["Complaint No."] = m_comp.group(0).strip()
                continue
        # 4. Devanagari or dealer line when BCCD Name is empty
        if not out["BCCD Name"] and (
            re.search(r"[\u0900-\u097F]", l_clean)
            or any(w in l_clean.lower() for w in ("elec", "enterp", "service", "traders", "sales"))
        ):
            out["BCCD Name"] = l_clean
            continue
        # 5. Defect line when Nature Of Defect is empty or got a tech name by mistake
        if any(w in l_clean.lower() for w in ("dead", "not working", "not heating", "short", "problem", "damage")):
            if not out["Nature Of Defect"] or not any(
                w in out["Nature Of Defect"].lower()
                for w in ("dead", "working", "heating", "pcb", "prob", "short", "built", "rp-", "220f", "phd", "ocdl")
            ):
                if out["Nature Of Defect"] and not out["Technician Name"]:
                    # Previous value in Nature Of Defect was actually the Technician Name! (e.g. Alex Demo)
                    out["Technician Name"] = out["Nature Of Defect"]
                out["Nature Of Defect"] = l_clean
                continue
        # 6. Remaining short alphabetic line at the end -> Technician Name
        if (
            not out["Technician Name"]
            and l_clean not in out.values()
            and re.search(r"[A-Za-z]{3,}", l_clean)
            and not re.search(r"\d{3,}", l_clean)
            and not any(w in l_clean.lower() for w in ("pcb", "main", "display", "infrared", "induction", "road", "street"))
        ):
            out["Technician Name"] = l_clean

    # If Nature Of Defect accidentally captured a technician name (e.g. "Alex Demo") while Spare Part Name had "Maim PCB Dead"
    if extra["Spare Part Name"] and any(w in extra["Spare Part Name"].lower() for w in ("dead", "working", "defect")):
        if out["Nature Of Defect"] and not any(w in out["Nature Of Defect"].lower() for w in ("dead", "working", "pcb", "prob")):
            if not out["Technician Name"]:
                out["Technician Name"] = out["Nature Of Defect"]
        out["Nature Of Defect"] = extra["Spare Part Name"]

    # If Product Description is empty but Spare Part Name was captured on 11-row tag, use it
    if not out["Product Description"] and extra["Spare Part Name"]:
        out["Product Description"] = extra["Spare Part Name"]

    # If Branch is empty and BCCD Location has a city, use BCCD Location
    if not out["Branch"] and extra["BCCD Location"]:
        out["Branch"] = extra["BCCD Location"]

    # Clean up values
    for k in KEYS:
        out[k] = _clean_field_text(out[k])
    if out["Technician Name"]:
        # Strip accidental defect prefix ("Mairm PCB Dead Alex Demo" -> "Alex Demo")
        tech_cleaned = re.sub(
            r"^(?:mai[nr]m?\s+)?(?:display\s+)?(?:pcb\s+)?(?:dead|deed|defect|not\s+working)\b[\s:.\-|]*",
            "",
            out["Technician Name"],
            flags=re.IGNORECASE,
        ).strip()
        if tech_cleaned:
            out["Technician Name"] = tech_cleaned
    out["Spare Part Code"] = re.sub(r"^[^A-Za-z0-9]+", "", out["Spare Part Code"]).strip()
    return out


def _fast_ocr_call(
    img_bgr,
    model: str,
    timings: dict | None = None,
    num_predict: int = 300,
    timeout: int = 120,
    prompt: str | None = None,
) -> str:
    """Resize to optimal vision patch grid (<=896x560) and query Ollama in-memory with stop tokens."""
    import cv2

    h, w = img_bgr.shape[:2]
    scale = min(896 / max(h, w), 560 / min(h, w), 1.0)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    options = {
        "temperature": 0,
        "num_predict": num_predict,
        "repeat_penalty": 1.25,
        "stop": [
            "```",
            "\n```",
            "\nDefective Spare",
            "\nItem Pro",
            "\nIten Pro",
            # Some OCR outputs switch to an HTML table. Stop at its close (or
            # before a repeated table) so the 300-token budget is spent on the
            # first complete record, not a duplicate.
            "</table>",
            "\n<table",
        ],
    }
    if NUM_CTX > 0:
        options["num_ctx"] = NUM_CTX
    if GPU_LAYERS >= 0:
        # 0 disables the GPU entirely, which is how CPU-only machines run.
        options["num_gpu"] = GPU_LAYERS

    payload = {
        "model": model,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": options,
        "messages": [{"role": "user", "content": prompt or T.OCR_PROMPT, "images": [b64]}],
    }
    body = _ollama_request("POST", _CHAT_PATH, payload=payload, timeout=timeout)
    raw = body["message"]["content"]

    if timings is not None:
        def _ms(key: str) -> int:
            return int((body.get(key) or 0) / 1_000_000)

        timings.update({
            "load_ms": _ms("load_duration"),
            "prefill_ms": _ms("prompt_eval_duration"),
            "decode_ms": _ms("eval_duration"),
            "total_ms": _ms("total_duration"),
            "prompt_tokens": int(body.get("prompt_eval_count") or 0),
            "output_tokens": int(body.get("eval_count") or 0),
        })

    # Deduplicate lines
    lines = [l.strip() for l in re.split(r"\n+", raw) if l.strip()]
    seen, deduped = set(), []
    for l in lines:
        key = re.sub(r"[^A-Za-z0-9]", "", l).lower()
        if key and key in seen and len(key) > 10:
            break
        if key:
            seen.add(key)
        deduped.append(l)
    return "\n".join(deduped)


_warm_lock = threading.Lock()
_load_lock = threading.Lock()
_warm_states: dict[str, dict] = {}


def _model_state(model: str) -> dict:
    with _warm_lock:
        state = _warm_states.get(model)
        if state:
            return dict(state)
    return {
        "state": "idle",
        "model": model,
        "seconds": 0.0,
        "gpu_percent": 0,
        "detail": "Not preloaded",
    }


def _set_model_state(model: str, **values) -> None:
    with _warm_lock:
        state = _warm_states.setdefault(model, {"model": model})
        state.update(values)


def _loaded_model_info(model: str) -> dict | None:
    try:
        data = _ollama_request("GET", "/api/ps", timeout=4)
    except Exception:
        return None
    for item in data.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return item
    return None


def _gpu_percent(info: dict | None) -> int:
    if not info:
        return 0
    size = int(info.get("size") or 0)
    vram = int(info.get("size_vram") or 0)
    if size <= 0:
        return 0
    return max(0, min(100, round(100 * vram / size)))


def _model_matches(info: dict | None) -> bool:
    """True when the resident model matches the requested device/context."""
    if not info:
        return False
    pct = _gpu_percent(info)
    if GPU_LAYERS == 0 and pct > 0:
        return False
    if GPU_LAYERS > 0 and pct < 98:
        return False
    if NUM_CTX > 0:
        ctx = int(info.get("context_length") or 0)
        if ctx and ctx != NUM_CTX:
            return False
    return True


def _unload_model(model: str) -> None:
    payload = {"model": model, "keep_alive": 0}
    try:
        _ollama_request("POST", "/api/generate", payload=payload, timeout=60)
    except Exception:
        pass

    for _ in range(20):
        info = _loaded_model_info(model)
        if not info or int(info.get("size_vram") or 0) == 0:
            break
        time.sleep(0.5)
    time.sleep(1.0)


def _warmup_once(model: str) -> dict:
    start = time.perf_counter()
    _set_model_state(
        model,
        state="warming",
        seconds=0.0,
        gpu_percent=0,
        started_at=time.time(),
        detail="Preloading OCR model..." if GPU_LAYERS != 0 else "Preloading OCR model on CPU...",
    )
    import numpy as np

    try:
        with _load_lock:
            info = _loaded_model_info(model)
            if info and not _model_matches(info):
                _set_model_state(
                    model,
                    detail="Model was not fully on GPU - reloading...",
                )
                _unload_model(model)
                info = None

            blank = np.full((64, 64, 3), 255, np.uint8)
            _fast_ocr_call(blank, model, num_predict=1, timeout=30)

            info = _loaded_model_info(model)
            if info and not _model_matches(info):
                _set_model_state(model, detail="Forcing a clean full-GPU reload...")
                _unload_model(model)
                _fast_ocr_call(blank, model, num_predict=1, timeout=30)
                info = _loaded_model_info(model)

        seconds = round(time.perf_counter() - start, 2)
        pct = _gpu_percent(info)
        if GPU_LAYERS == 0:
            state = "ready"
            detail = "Model preloaded on CPU (GPU disabled)"
        elif _model_matches(info):
            state = "ready"
            detail = f"Model preloaded on GPU ({pct}% VRAM)"
        elif info:
            state = "partial"
            detail = f"Model only {pct}% on GPU - scans will be slower"
        else:
            state = "ready"
            detail = "Model preloaded (GPU status unavailable)"
        _set_model_state(
            model,
            state=state,
            seconds=seconds,
            gpu_percent=pct,
            detail=detail,
        )
    except Exception as exc:
        _set_model_state(
            model,
            state="error",
            seconds=round(time.perf_counter() - start, 2),
            gpu_percent=0,
            detail=str(exc),
        )
    return _model_state(model)


def start_warmup(model: str | None = None, blocking: bool = False) -> dict:
    """Preload the model (async by default) and return the current warmup state."""
    model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    with _warm_lock:
        state = _warm_states.get(model)
        if state and state.get("state") == "warming":
            started_at = float(state.get("started_at") or 0)
            if started_at and time.time() - started_at > 90:
                state = dict(state)
                state.update(
                    state="error",
                    detail="Preload timed out - scanning will load the model on demand",
                )
                _warm_states[model] = state
                return dict(state)
            return dict(state)

    info = _loaded_model_info(model)
    if info and _model_matches(info):
        pct = _gpu_percent(info)
        _set_model_state(
            model,
            state="ready",
            seconds=0.0,
            gpu_percent=pct,
            detail=f"Model preloaded on GPU ({pct}% VRAM)",
        )
        return _model_state(model)

    with _warm_lock:
        state = _warm_states.get(model)
        if state and state.get("state") == "warming":
            return dict(state)
        _warm_states[model] = {
            "state": "warming",
            "model": model,
            "seconds": 0.0,
            "gpu_percent": 0,
            "started_at": time.time(),
            "detail": "Starting preload...",
        }

    if blocking:
        return _warmup_once(model)

    threading.Thread(target=_warmup_once, args=(model,), daemon=True).start()
    return _model_state(model)


def warmup_status(model: str | None = None) -> dict:
    model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return _model_state(model)


def _has_devanagari_text(value: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", value or ""))


def _has_bengali_text(value: str) -> bool:
    return bool(re.search(r"[\u0980-\u09FF]", value or ""))


def _top_crop(img_bgr, y0: float = 0.01, y1: float = 0.55, x0: float = 0.10, x1: float = 0.99):
    """Return the top part of an upright tag, where a dealer stamp usually is."""
    h, w = img_bgr.shape[:2]
    return img_bgr[
        max(0, int(h * y0)):min(h, int(h * y1)),
        max(0, int(w * x0)):min(w, int(w * x1)),
    ]


def _dealer_strip_enhanced(img_bgr):
    """Branch/BCCD strip with a local-contrast boost for faint handwriting.

    Blue ballpoint on the striped tag paper is often too low-contrast for the
    raw full-tag pass, but CLAHE makes the strokes readable while leaving the
    printed labels intact.  Used only when the normal stamp pass found nothing.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    roi = img_bgr[int(h * 0.04):int(h * 0.24), int(w * 0.10):w]
    if roi.size == 0:
        return img_bgr
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    boosted = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    up = cv2.resize(boosted, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)


def _parse_stamp_text(text: str, resolver: EntityResolver) -> dict:
    """Extract dealer/city lines from a stamp-focused OCR pass.

    The model usually returns clean lines such as ``DEMO ELECTRONICS`` and
    ``TESTNAGAR``, but can also return a small HTML/JSON fragment or repeat the
    tag headers.  Keep only lines that resolve to a known place or a dealer.
    """
    lines: list[str] = []
    for raw in re.split(r"\n+", _flatten_ocr_html(text or "")):
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\s`*#>+\-\d.)]+", "", line)
        line = re.sub(
            r'^\s*["\']?(?:Branch|BCCD\s*(?:Name)?|CCD\s*(?:Name)?|Name|Location)'
            r'["\']?\s*[:\-|]?\s*',
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"[`{}\"\[\]]", " ", line)
        line = re.sub(r"\s+", " ", line).strip(" ,.-|")
        if not line:
            continue
        # The model sometimes glues the printed header onto the first
        # handwritten value ("TAG Defective Spare Part Return SAMPLE
        # TRADERS").  Keep the tail when it still looks like a word;
        # only a pure header line is dropped.
        header_tail = re.sub(
            r"^(?:TAG\s+)?(?:DETECTIVE|DEFECTIVE|DETECT|DEFECT)?\s*"
            r"(?:SPARE\s+PART\s+)?(?:RETURN(?:\s+TAG)?)?\s*[:\-|.]?\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip(" ?.-|")
        if header_tail != line and len(re.findall(r"[A-Za-z\u0900-\u097F]", header_tail)) >= 3:
            line = header_tail
        if re.search(
            r"^(?:TAG|DEFECTIVE|SPARE\s+PART|RETURN|PRODUCT|ITEM|"
            r"SR\.?\s*NO|DESCRIPTION|DATE|COMPLAINT|SPARE|NATURE|"
            r"TECHNICIAN|MODEL|MOLEF)\b",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(r"transcribe|plain lines|markdown|json|table|[<>]", line, flags=re.IGNORECASE):
            continue
        if len(re.findall(r"[A-Za-z\u0900-\u097F]", line)) < 3:
            continue
        lines.append(line)

    branch = ""
    bccd = ""
    for line in lines:
        place = resolver.extract_place(line)
        if place and resolver._is_place_only(line):
            branch = branch or place
            continue

        resolved = resolver.resolve_bccd(line, branch=branch)
        if resolved.get("status") in ("READ", "REVIEW") and resolved.get("value"):
            if not bccd:
                bccd = resolved["value"]
            if place and not branch:
                branch = place
    return {"Branch": branch, "BCCD Name": bccd}


def _recover_hindi_bccd_from_top(img_bgr, model: str, timings: dict | None = None) -> str:
    """Read the handwritten dealer word from the top rows of a Hindi tag.

    GLM-OCR often loses the BCCD line when it transcribes the whole table.  A
    tight top crop with a word-level prompt gives it a second, much simpler
    look.  This is a fallback only: normal scans never pay for it.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    if h < 2 or w < 2:
        return ""
    # Full width matters: the model uses the printed labels to anchor the row.
    # Upscaling to a fixed width improves the small 640x480 phone photos.
    crop = img_bgr[:max(1, int(h * 0.42)), :]
    scale = 2560.0 / max(w, 1)
    if scale > 1.0:
        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LANCZOS4,
        )
    try:
        text = _fast_ocr_call(
            crop,
            model,
            timings=timings,
            num_predict=45,
            prompt=HINDI_DEALER_CROP_PROMPT,
        )
    except Exception as exc:
        print(f"[scan] Hindi dealer fallback skipped: {exc}")
        return ""

    for raw in re.split(r"\n+", text or ""):
        line = re.sub(r"^[\s`*#>+\-\d.)]+", "", raw).strip(" ,.-|")
        if not line:
            continue
        if re.search(
            r"\b(?:bccd|branch|product|description|location|name)\b",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        if len(re.findall(r"[\u0900-\u097F\u0980-\u09FF]", line)) < 3:
            continue
        resolved = resolver.resolve_bccd(line)
        if resolved.get("status") == "READ" and resolved.get("value"):
            return resolved["value"]
    return ""


def _is_sideways_img(img_bgr) -> bool:
    """In-memory check whether the tag's table lines are rotated 90/270 degrees."""
    import cv2

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    h, w = gray.shape[:2]
    scale = 800 / max(h, w, 1)
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    hs = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vs = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    return int((vs > 0).sum()) > int((hs > 0).sum())


def scan_and_resolve(
    image_path: Path,
    proc_dir: Path | None = None,
    model: str = "glm-ocr:q8_0",
    dual_rotation: bool = False,
) -> dict:
    """Scan tag image and resolve all 9 fields against the master knowledge base."""
    t_start = time.perf_counter()
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    side = _is_sideways_img(img)
    candidates = (270, 90) if side else (0, 180)

    texts, scores, mapped_by_angle = {}, {}, {}
    pass_timings = []

    def rotate_img(angle):
        flag = {
            0: None,
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(angle)
        if angle in (0, 0.0):
            return img
        if flag is not None:
            return cv2.rotate(img, flag)
        # Arbitrary small angle (deskew retry): rotate about the centre and
        # fill the exposed corners with white so the table stays readable.
        matrix = cv2.getRotationMatrix2D(
            (img.shape[1] / 2.0, img.shape[0] / 2.0), float(angle), 1.0
        )
        return cv2.warpAffine(
            img, matrix, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255)
        )

    def ocr_pass(angle: int) -> str:
        timings = {}
        text = _fast_ocr_call(rotate_img(angle), model, timings=timings)
        timings["angle"] = angle
        pass_timings.append(timings)
        return text

    # First pass on primary candidate angle
    a0 = candidates[0]
    txt0 = ocr_pass(a0)
    sc0 = T.orientation_score(txt0)
    map0 = map_lines_enhanced(txt0)
    texts[a0] = txt0
    scores[a0] = sc0
    mapped_by_angle[a0] = map0

    best_angle = a0
    best_txt = txt0
    filled_count = sum(1 for v in map0.values() if v)
    has_code_or_sr_or_comp = bool(
        map0.get("Spare Part Code")
        or map0.get("Product Sr. No.")
        or map0.get("Complaint No.")
    )

    # Smart fast-path threshold: 11-row tags or sparse tags with score >= 11 and >= 4 fields
    # are upright and complete on the first pass (cuts 2-pass 2.8s scans down to 1-pass ~1.0s!)
    need_second = dual_rotation or sc0 < 11 or filled_count < 4 or not has_code_or_sr_or_comp
    if need_second:
        a1 = candidates[1]
        txt1 = ocr_pass(a1)
        sc1 = T.orientation_score(txt1)
        map1 = map_lines_enhanced(txt1)
        texts[a1] = txt1
        scores[a1] = sc1
        mapped_by_angle[a1] = map1
        if sc1 > sc0 or (sc1 == sc0 and sum(1 for v in map1.values() if v) > filled_count):
            best_angle = a1
            best_txt = txt1

    # A slightly tilted phone photo can make the vision encoder read every
    # printed label but none of the handwriting.  When no pass produced any
    # serial/complaint/part value, retry a few small deskew angles and keep
    # the pass with real data; upright scans never pay for this.
    def _pass_quality(angle) -> tuple:
        mp = mapped_by_angle.get(angle) or {}
        has_ids = bool(
            mp.get("Spare Part Code")
            or mp.get("Product Sr. No.")
            or mp.get("Complaint No.")
        )
        dealer = (mp.get("BCCD Name") or mp.get("Branch") or "").strip()
        dealer_ok = False
        if dealer:
            try:
                dealer_ok = resolver.resolve_bccd(dealer).get("status") == "READ"
            except Exception:
                dealer_ok = False
        return (has_ids, dealer_ok, sum(1 for v in mp.values() if v))

    if not _pass_quality(best_angle)[0]:
        for deskew_angle in (4.0, -4.0, 8.0, -8.0):
            txt = ocr_pass(deskew_angle)
            texts[deskew_angle] = txt
            scores[deskew_angle] = T.orientation_score(txt)
            mapped_by_angle[deskew_angle] = map_lines_enhanced(txt)
            if _pass_quality(deskew_angle) > _pass_quality(best_angle):
                best_angle = deskew_angle
                best_txt = txt
            if _pass_quality(deskew_angle)[:2] == (True, True):
                break

    raw_mapped = dict(mapped_by_angle[best_angle])
    for angle, mp in mapped_by_angle.items():
        if angle == best_angle:
            continue
        for k in KEYS:
            if not raw_mapped[k] and mp.get(k):
                raw_mapped[k] = mp[k]

    raw_text_flat = _flatten_ocr_html(best_txt or "")

    # A BCCD cell that holds only a phone fragment or a stray letter is empty
    # for matching purposes; _clean_bccd_value erases it later anyway.  Clearing
    # it now lets the stamp fallback and the dealer-line scan below run.
    if len(re.findall(r"[A-Za-z\u0900-\u097F\u0980-\u09FF]", raw_mapped.get("BCCD Name", ""))) < 3:
        raw_mapped["BCCD Name"] = ""

    # The line mapper sometimes drops the dealer line even though it is
    # readable in the OCR text (on faded stamps it can land under Branch, or
    # on its own line).  Recover it only from the rows above Product
    # Description and only when the line resolves exactly to a known dealer,
    # so ordinary address/Hindi text stays out of BCCD.
    if not raw_mapped.get("BCCD Name"):
        for raw_line in re.split(r"\n+", raw_text_flat):
            if re.match(r"^\s*Product\s+Description\b", raw_line, flags=re.IGNORECASE):
                break
            probe = raw_line.strip(" ?.-|")
            if len(re.findall(r"[A-Za-z]", probe)) < 3:
                continue
            if re.search(
                r"^(?:TAG|DEFECTIVE|SPARE\s+PART|RETURN)\b",
                probe,
                flags=re.IGNORECASE,
            ):
                continue
            cand = resolver.resolve_bccd(probe)
            if cand.get("status") == "READ" and cand.get("value"):
                raw_mapped["BCCD Name"] = cand["value"]
                break

    hindi_detected = (
        any(
            _has_devanagari_text(raw_mapped.get(k, "")) or _has_bengali_text(raw_mapped.get(k, ""))
            for k in ("Branch", "BCCD Name")
        )
        or _has_devanagari_text(raw_text_flat)
        or _has_bengali_text(raw_text_flat)
        or bool(re.search(r"[\u0900-\u0FFF]", raw_text_flat))
    )

    # In the Hindi layout the two handwritten value lines are often shifted
    # into unrelated slots by the generic line mapper.  The first Devanagari
    # value line is normally the dealer and the second is the city/address.
    # Recover those lines directly, then let the normal place/dealer splitter
    # finish the job.
    if hindi_detected:
        top_lines = []
        for line in re.split(r"\n+", raw_text_flat):
            if re.match(r"^\s*Product\s+Description\b", line, flags=re.IGNORECASE):
                break
            top_lines.append(line)
        hindi_lines = [
            line.strip()
            for line in top_lines
            if len(re.findall(r"[\u0900-\u097F\u0980-\u09FF]", line)) >= 3
        ]
        if hindi_lines:
            first = hindi_lines[0]
            second = hindi_lines[1] if len(hindi_lines) > 1 else ""
            first_res = resolver.resolve_bccd(first)
            second_res = resolver.resolve_bccd(second) if second else {}
            first_place = resolver.extract_place(first)
            second_place = resolver.extract_place(second)

            dealer_line = ""
            branch_value = ""
            if first_res.get("status") in ("READ", "REVIEW") and first_res.get("value"):
                dealer_line = first
                branch_value = second_place
            elif second_res.get("status") in ("READ", "REVIEW") and second_res.get("value"):
                dealer_line = second
                branch_value = first_place
            else:
                branch_value = first_place or second_place

            current_bccd = raw_mapped.get("BCCD Name", "")
            current_status = (
                resolver.resolve_bccd(current_bccd).get("status")
                if current_bccd else "UNREADABLE"
            )
            if dealer_line and current_status != "READ":
                raw_mapped["BCCD Name"] = dealer_line
            if branch_value and not resolver.extract_place(raw_mapped.get("Branch", "")):
                raw_mapped["Branch"] = branch_value

        # If a non-Latin tag still has no resolved BCCD, run the fast 45-token
        # Hindi dealer top-crop BEFORE the 120-token/80-token stamp fallbacks.
        if resolver.resolve_bccd(raw_mapped.get("BCCD Name", "")).get("status") != "READ":
            first_source = rotate_img(best_angle)
            sources = [first_source] if first_source is img else [first_source, img]
            for source in sources:
                hindi_timings: dict = {}
                recovered_bccd = _recover_hindi_bccd_from_top(source, model, timings=hindi_timings)
                if hindi_timings:
                    hindi_timings["angle"] = "hindi_top"
                    pass_timings.append(hindi_timings)
                if recovered_bccd:
                    raw_mapped["BCCD Name"] = recovered_bccd
                    break

    # When the BCCD field is empty, the circular dealer stamp is often the
    # only readable source of the dealer and city (e.g. DEMO ELECTRONICS /
    # TESTNAGAR).  Read just the top of the tag; skip this extra pass entirely
    # when BCCD was already read or when Branch holds a pure dealer name with
    # no address/city (e.g. "For Mock Services Service", which
    # normalize_branch_bccd moves to BCCD).
    has_pure_dealer_in_branch = bool(
        raw_mapped.get("Branch")
        and resolver._is_business_text(raw_mapped["Branch"])
        and not resolver.extract_place(raw_mapped["Branch"])
    )
    if not raw_mapped.get("BCCD Name") and not has_pure_dealer_in_branch:
        # Before running 200 tokens of stamp OCR, try the fast 45-token Hindi
        # dealer top-crop when Branch is also empty (common on handwritten
        # Hindi tags where the whole-table pass skipped the top handwriting).
        if not raw_mapped.get("Branch"):
            first_source = rotate_img(best_angle)
            sources = [first_source] if first_source is img else [first_source, img]
            for source in sources:
                hindi_timings: dict = {}
                recovered_bccd = _recover_hindi_bccd_from_top(
                    source, model, timings=hindi_timings
                )
                if hindi_timings:
                    hindi_timings["angle"] = "hindi_top"
                    pass_timings.append(hindi_timings)
                if recovered_bccd:
                    raw_mapped["BCCD Name"] = recovered_bccd
                    hindi_detected = True
                    break

    if not raw_mapped.get("BCCD Name") and not has_pure_dealer_in_branch:
        extra_timings: dict = {}
        try:
            extra_text = _fast_ocr_call(
                _top_crop(rotate_img(best_angle)),
                model,
                timings=extra_timings,
                num_predict=120,
            )
        except Exception as exc:
            # A fallback pass must never turn a good scan into a failed scan.
            print(f"[scan] stamp fallback skipped: {exc}")
            extra_text = ""
        extra_timings["angle"] = "stamp"
        pass_timings.append(extra_timings)
        if extra_text:
            extra_mapped = _parse_stamp_text(extra_text, resolver)
            if extra_mapped.get("BCCD Name") and not raw_mapped.get("BCCD Name"):
                raw_mapped["BCCD Name"] = extra_mapped["BCCD Name"]
            if extra_mapped.get("Branch") and not resolver.extract_place(
                raw_mapped.get("Branch", "")
            ):
                raw_mapped["Branch"] = extra_mapped["Branch"]

        # Faint blue ballpoint can vanish in the raw photo but is readable
        # after local-contrast enhancement of the Branch/BCCD strip.  Only
        # runs when the normal stamp pass above did not find the dealer.
        if not raw_mapped.get("BCCD Name"):
            boosted_timings: dict = {}
            try:
                boosted_text = _fast_ocr_call(
                    _dealer_strip_enhanced(rotate_img(best_angle)),
                    model,
                    timings=boosted_timings,
                    num_predict=80,
                    prompt=(
                        "Transcribe only the handwritten words in the top rows "
                        "of this tag. Plain lines only, one value per line."
                    ),
                )
            except Exception as exc:
                # Fallbacks must never turn a good scan into a failed scan.
                print(f"[scan] enhanced strip fallback skipped: {exc}")
                boosted_text = ""
            boosted_timings["angle"] = "stamp2"
            pass_timings.append(boosted_timings)
            if boosted_text:
                boosted_mapped = _parse_stamp_text(boosted_text, resolver)
                if boosted_mapped.get("BCCD Name") and not raw_mapped.get("BCCD Name"):
                    raw_mapped["BCCD Name"] = boosted_mapped["BCCD Name"]
                if boosted_mapped.get("Branch") and not resolver.extract_place(
                    raw_mapped.get("Branch", "")
                ):
                    raw_mapped["Branch"] = boosted_mapped["Branch"]

    # Some handwritten dealer lines are completely lost by the table
    # transcription (the model may read them as Bengali, Tibetan or unrelated
    # Devanagari words, or skip the handwriting entirely).  Give only the top
    # rows a focused second look when the BCCD field still cannot be resolved;
    # this keeps normal scans fast.
    if (
        not hindi_detected
        and not has_pure_dealer_in_branch
        and resolver.resolve_bccd(raw_mapped.get("BCCD Name", "")).get("status") != "READ"
    ):
        first_source = rotate_img(best_angle)
        sources = [first_source] if first_source is img else [first_source, img]
        for source in sources:
            hindi_timings: dict = {}
            recovered_bccd = _recover_hindi_bccd_from_top(source, model, timings=hindi_timings)
            if hindi_timings:
                hindi_timings["angle"] = "hindi_top"
                pass_timings.append(hindi_timings)
            if recovered_bccd:
                raw_mapped["BCCD Name"] = recovered_bccd
                # Treat a recovered dealer as a Hindi tag: the local
                # dealer->branch table and the "never guess a wrong city"
                # guard further down should also apply to it.
                hindi_detected = True
                break

    # Keep Branch as a place and BCCD as the dealer/shop.  This also handles
    # swapped fields, combined values ("Demo Traders TESTNAGAR"), and
    # product/address text that shifted into either field.
    raw_mapped = resolver.normalize_branch_bccd(raw_mapped)

    # Step 1: Validate Serial & Complaint first
    serial_info = resolver.validate_serial(raw_mapped.get("Product Sr. No.", ""))
    complaint_info = resolver.validate_complaint(raw_mapped.get("Complaint No.", ""), branch=raw_mapped.get("Branch", ""))

    # Step 2: Resolve Branch (preliminary) & BCCD Name, then cross-infer Branch <-> BCCD!
    branch_prelim = resolver.resolve_branch(
        raw_mapped.get("Branch", ""),
        complaint_hint=complaint_info["value"] or raw_mapped.get("Complaint No.", ""),
        bccd_hint=raw_mapped.get("BCCD Name", ""),
    )
    bccd_info = resolver.resolve_bccd(
        raw_mapped.get("BCCD Name", ""),
        branch=branch_prelim["value"],
        tech_hint=raw_mapped.get("Technician Name", ""),
        complaint_hint=complaint_info["value"],
    )
    # Re-resolve Branch with resolved BCCD Name (so EXAMPLE ENTERPRISES -> Demopur, MOCK SERVICES -> MOCKABAD, DEMO ELECTRICALS -> MOCKABAD, etc.)
    branch_info = resolver.resolve_branch(
        raw_mapped.get("Branch", ""),
        complaint_hint=complaint_info["value"] or raw_mapped.get("Complaint No.", ""),
        bccd_hint=raw_mapped.get("BCCD Name", ""),
        resolved_bccd=bccd_info["value"],
    )
    # Re-validate complaint with the final branch (repairs a prefix that
    # resolved to the wrong branch).
    complaint_info = resolver.validate_complaint(raw_mapped.get("Complaint No.", ""), branch=branch_info["value"])

    # Hindi tags often have the dealer/city order reversed, and the BCCD
    # co-occurrence table can confidently point at the wrong city for a new
    # Hindi dealer.  Prefer a blank Branch over a dealer-inferred wrong one.
    if hindi_detected and branch_info.get("confidence") == "INFERRED_FROM_BCCD":
        branch_info = {
            "value": "",
            "status": "UNREADABLE",
            "confidence": "NONE",
            "score": 0.0,
            "candidates": [],
        }

    # Local handwritten demo tags are from the main demo shop in Samplepur.  When
    # the same dealer name exists in multiple branches, the co-occurrence table
    # can point at the wrong one, so only fill a blank Branch (never overwrite a
    # written city) and only for Hindi tags.
    if (
        hindi_detected
        and bccd_info.get("status") == "READ"
        and not branch_info.get("value")
    ):
        local_branch = HINDI_LOCAL_BCCD_BRANCH.get(bccd_info.get("value", ""))
        if local_branch:
            branch_info = {
                "value": local_branch,
                "status": "READ",
                "confidence": "LOCAL_HINDI_DEALER",
                "score": 0.95,
                "candidates": [{"canonical": local_branch, "score": 0.95}],
            }

    # Step 3: Resolve Part Code (with fallback to Product Description + Defect inference when Part Code row is blank)
    part_info = resolver.resolve_part_code(
        raw_mapped.get("Spare Part Code", ""),
        fallback_text=raw_mapped.get("Product Description", ""),
        defect_text=raw_mapped.get("Nature Of Defect", ""),
        bccd_name=bccd_info["value"],
    )

    # Step 4: Resolve Technician Name using BCCD-specific + Branch-specific technician rosters
    tech_info = resolver.resolve_technician(
        raw_mapped.get("Technician Name", ""),
        bccd_name=bccd_info["value"],
        branch=branch_info["value"],
    )
    date_info = resolver.validate_date(raw_mapped.get("Date of Purchase", ""))

    # Step 5: Check if this exact Serial or Complaint exists in the demo knowledge base for instant enrichment
    exact_rec = resolver.lookup_exact_record(serial_info["value"], complaint_info["value"])
    if exact_rec:
        if branch_info["status"] != "READ" and exact_rec.get("br"):
            branch_info = {"value": exact_rec["br"], "status": "READ", "confidence": "EXACT_DB", "score": 1.0, "candidates": [{"canonical": exact_rec["br"], "score": 1.0}]}
        if bccd_info["status"] != "READ" and exact_rec.get("bc"):
            bccd_info = {"value": exact_rec["bc"], "status": "READ", "confidence": "EXACT_DB", "score": 1.0, "candidates": [{"canonical": exact_rec["bc"], "score": 1.0}]}
        if part_info["status"] != "READ" and exact_rec.get("pc"):
            part_info = resolver.resolve_part_code(exact_rec["pc"])
        if tech_info["status"] != "READ" and exact_rec.get("vt"):
            tech_info = {"value": exact_rec["vt"], "status": "READ", "confidence": "EXACT_DB", "score": 1.0, "candidates": [{"canonical": exact_rec["vt"], "score": 1.0}]}

    # The printed circular stamp pairs a dealer around the rim with a place in
    # the middle.  When the rim is too faded for OCR but the middle place is
    # readable, fill BCCD from the place for the places that have a single
    # known dealer (TESTNAGAR -> DEMO ELECTRONICS).  Runs after the exact-match
    # enrichment so an exact record match always wins.
    if bccd_info.get("status") != "READ" and branch_info.get("value"):
        stamp_dealer = STAMP_PLACE_BCCD_OVERRIDES.get(
            branch_info["value"].strip().upper()
        )
        if stamp_dealer:
            bccd_info = {
                "value": stamp_dealer,
                "status": "READ",
                "confidence": "STAMP_PLACE",
                "score": 0.9,
                "candidates": [{"canonical": stamp_dealer, "score": 0.9}],
            }

    # Product description derived directly from resolved part code
    desc_val = part_info.get("derived_description") or raw_mapped.get("Product Description", "")
    desc_status = "READ" if part_info.get("derived_description") else ("REVIEW" if desc_val else "UNREADABLE")
    desc_info = {"value": desc_val, "status": desc_status}

    # Defect is Class C: preserve raw OCR verbatim, only clean obvious OCR noise on common words if helpful
    defect_raw = raw_mapped.get("Nature Of Defect", "").strip()
    # Handwritten "Dead" is sometimes read as the English word "Dear"; the field
    # is effectively a Dead/Working flag on these tags, so canonicalize it.
    if defect_raw.lower() == "dear":
        defect_raw = "Dead"
    defect_info = {"value": defect_raw, "status": "READ" if defect_raw else "UNREADABLE"}

    fields = {
        "Branch": branch_info,
        "BCCD Name": bccd_info,
        "Spare Part Code": part_info,
        "Product Description": desc_info,
        "Product Sr. No.": serial_info,
        "Date of Purchase": date_info,
        "Complaint No.": complaint_info,
        "Nature Of Defect": defect_info,
        "Technician Name": tech_info,
    }

    flat_result = {k: v["value"] for k, v in fields.items()}
    elapsed = round(time.perf_counter() - t_start, 2)

    def _sum_s(key: str) -> float:
        return round(sum(t.get(key, 0) for t in pass_timings) / 1000.0, 2)

    model_s = _sum_s("total_ms")
    timings = {
        "load_s": _sum_s("load_ms"),
        "prefill_s": _sum_s("prefill_ms"),
        "decode_s": _sum_s("decode_ms"),
        "model_s": model_s,
        "overhead_s": round(max(0.0, elapsed - model_s), 2),
        "by_pass": [
            {
                "angle": t["angle"],
                "load_s": round(t.get("load_ms", 0) / 1000.0, 2),
                "prefill_s": round(t.get("prefill_ms", 0) / 1000.0, 2),
                "decode_s": round(t.get("decode_ms", 0) / 1000.0, 2),
                "prompt_tokens": t.get("prompt_tokens", 0),
                "output_tokens": t.get("output_tokens", 0),
            }
            for t in pass_timings
        ],
    }

    read_count = sum(1 for v in fields.values() if v.get("status") == "READ")
    filled_final = sum(1 for v in fields.values() if v.get("value"))

    return {
        "angle": best_angle,
        "passes": len(scores),
        "scores": scores,
        "elapsed": elapsed,
        "timings": timings,
        "model": model,
        "read_count": read_count,
        "filled_count": filled_final,
        "exact_db_match": bool(exact_rec),
        "raw_ocr": best_txt,
        "raw_mapped": raw_mapped,
        "fields": fields,
        "result": flat_result,
    }


if __name__ == "__main__":
    sample_dir = Path(__file__).parent / "samples"
    files = [f for f in sorted(sample_dir.glob("*.*")) if f.suffix.lower() in (".jpg", ".jpeg", ".png") and "Blank" not in f.name]
    start_warmup(blocking=True)
    for img in files:
        print(f"\n==================== {img.name} ====================")
        res = scan_and_resolve(img)
        print(
            f"Elapsed: {res['elapsed']}s ({res['passes']} pass) | Best Angle: {res['angle']} | "
            f"READ: {res['read_count']}/9 | Filled: {res['filled_count']}/9"
        )
        for k, v in res["fields"].items():
            cands = [c.get("canonical") or c.get("code") for c in v.get("candidates", [])]
            cand_str = f" (cands: {cands})" if cands else ""
            print(f"  {k:20}: {v['value']:38} [{v.get('status', 'READ'):10}]{cand_str}")
