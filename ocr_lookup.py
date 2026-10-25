#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_lookup.py - Fast, free, 100% offline OCR field corrector for the two
OCR-critical fields of the repair datasheet:

    branch     (Branch)
    bccd_name  (BCCD)

Instead of letting the OCR model GUESS between look-alike words, every OCR
output is resolved against a canonical lexicon built from your own records.
Zero dependencies (pure Python stdlib), so it is free to
run locally next to any OCR pipeline (e.g. a 2.2B GLM vision model at max
config).  Typical latency: exact hits in <50 us, fuzzy hits in ~1-3 ms.

Matching pipeline per field:
    1. sentinel/empty check            -> EMPTY
    2. exact raw match                  -> EXACT
    3. case-insensitive match           -> HIGH   (method=case)
    4. punctuation/spacing-insensitive  -> HIGH   (method=norm / tight)
    5. fuzzy: bigram-candidate retrieval + banded OSA (Damerau) distance,
       adaptive edit budget by length, frequency prior for tie-breaks,
       optional branch<->bccd co-occurrence context to disambiguate
       -> HIGH / MEDIUM / LOW, or NONE (flagged for review, never guessed)

Accept policy (default): EXACT/HIGH/MEDIUM accepted, LOW/NONE flagged for
review (--strict: only EXACT/HIGH).  A LOW/NONE result always returns the
nearest dictionary value as a suggestion plus the alternatives, so a human
(or a second pass) decides - the script never silently invents values.

Usage:
    python ocr_lookup.py correct branch "testnagar"
    python ocr_lookup.py correct bccd "DEMO ELECTRONIC" --branch TESTNAGAR
    python ocr_lookup.py csv   --in data.csv --out corrected.csv
    python ocr_lookup.py batch < in.jsonl > out.jsonl
    python ocr_lookup.py selftest [--samples 3000]

Library:
    from ocr_lookup import Lexicon
    lex = Lexicon.load("data")
    r = lex.correct_bccd("demo electronic", branch="TESTNAGAR")
    print(r.value, r.confidence, r.score)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict, namedtuple

csv.field_size_limit(10 ** 9)

# Windows consoles default to cp1252; lexicon values may contain non-ASCII
# characters, so always emit UTF-8 instead of crashing mid-stream.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DICT_DIR = os.path.join(HERE, "data")

SENTINELS = {"", "NULL", "NA", "N/A", "NONE", "NAN", "-", "--", "\u2014", "."}
_WS = re.compile(r"[\W_]+", re.UNICODE)
_CENTRE = re.compile(r"\bCENTRE\b")

Correction = namedtuple(
    "Correction", "input value confidence method score accept alternatives"
)


def norm_key(s: str) -> str:
    """Uppercase, & -> AND, collapse punctuation/spaces, CENTRE -> CENTER."""
    s = _WS.sub(" ", s.upper().replace("&", " AND ")).strip()
    return _CENTRE.sub("CENTER", s)


def tight_key(s: str) -> str:
    """Spacing-insensitive key (norm_key with all spaces removed)."""
    return "".join(norm_key(s).split())


def allowed_edits(n: int) -> int:
    """Adaptive edit budget by string length (tight form)."""
    if n <= 2:
        return 0
    if n <= 4:
        return 1
    if n <= 8:
        return 2
    if n <= 14:
        return 3
    return 4


def osa_distance(a: str, b: str, cap: int) -> int:
    """Optimal-string-alignment (Damerau, adjacent transposition) distance
    with band + early exit.  Returns min(distance, cap+1)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    if la == 0:
        return lb if lb <= cap else cap + 1
    if lb == 0:
        return la if la <= cap else cap + 1
    OVER = cap + 2  # clamp for "beyond cap"
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        j_lo = max(1, i - cap - 1)
        j_hi = min(lb, i + cap + 1)
        cur = [OVER] * (lb + 1)
        cur[0] = i if i <= cap else OVER
        row_min = cur[0]
        ai = a[i - 1]
        for j in range(j_lo, j_hi + 1):
            cost = 0 if ai == b[j - 1] else 1
            v = prev[j - 1] + cost
            d = prev[j] + 1
            if d < v:
                v = d
            ins = cur[j - 1] + 1
            if ins < v:
                v = ins
            if i > 1 and j > 1 and ai == b[j - 2] and a[i - 2] == b[j - 1]:
                t = prev[j - 2] + 1
                if t < v:
                    v = t
            if v > cap:
                v = OVER
            cur[j] = v
            if v < row_min:
                row_min = v
        if row_min > cap:
            return cap + 1
        prev = cur
    return prev[lb] if prev[lb] <= cap else cap + 1


class FieldIndex:
    """In-memory index for one field's canonical entries."""

    def __init__(self, entries: list[dict], pair_counts: dict | None = None):
        # entries: [{"value": str, "count": int, "variants": [...]}]
        self.values = [e["value"] for e in entries]
        self.counts = [int(e["count"]) for e in entries]
        self.total = sum(self.counts) or 1
        self.n = len(self.values)

        self.case_map: dict[str, list[int]] = defaultdict(list)
        self.norm_map: dict[str, list[int]] = defaultdict(list)
        self.tight_map: dict[str, list[int]] = defaultdict(list)
        self.tight_of: list[str] = []
        self.bigram_idx: dict[str, list[int]] = defaultdict(list)
        self.by_len: dict[int, list[int]] = defaultdict(list)

        for i, v in enumerate(self.values):
            tk = tight_key(v)
            self.tight_of.append(tk)
            self.case_map[v.upper()].append(i)
            self.norm_map[norm_key(v)].append(i)
            self.tight_map[tk].append(i)
            self.by_len[len(tk)].append(i)
            for g in set(tk[p : p + 2] for p in range(len(tk) - 1)):
                self.bigram_idx[g].append(i)

        # branch <-> bccd co-occurrence, provided externally per field
        self.pair_counts: dict = pair_counts or {}

    def _pick_best(self, idxs: list[int]) -> int:
        return max(idxs, key=lambda i: self.counts[i])

    def _by_tight(self, tk: str) -> int | None:
        idxs = self.tight_map.get(tk)
        return self._pick_best(idxs) if idxs else None

    def _exact_stage(self, raw: str):
        """Stages 1-4.  Returns (idx, method, confidence) or None."""
        u = raw.upper()
        idxs = self.case_map.get(u)
        if idxs:
            m = "raw" if any(self.values[i] == raw for i in idxs) else "case"
            c = "EXACT" if m == "raw" else "HIGH"
            return self._pick_best(idxs), m, c
        nk = norm_key(raw)
        idxs = self.norm_map.get(nk)
        if idxs:
            return self._pick_best(idxs), "norm", "HIGH"
        tk = tight_key(raw)
        if tk:
            i = self._by_tight(tk)
            if i is not None:
                return i, "tight", "HIGH"
        return None

    def _fuzzy_stage(self, raw: str):
        """Stages 5: bigram retrieval + banded OSA.  Returns sorted cand list."""
        tk = tight_key(raw)
        if not tk:
            return []
        cap = allowed_edits(len(tk))
        if cap == 0:
            return []

        qgrams = set(tk[p : p + 2] for p in range(len(tk) - 1))
        if not qgrams:
            return []
        overlap: Counter = Counter()
        for g in qgrams:
            for i in self.bigram_idx.get(g, ()):
                overlap[i] += 1

        lo, hi = len(tk) - cap, len(tk) + cap
        cand = []
        for i, ov in overlap.items():
            lc = len(self.tight_of[i])
            if not (lo <= lc <= hi):
                continue
            need = max(1, min(len(tk), lc) - 2 * cap - 2)
            if ov >= need:
                cand.append((ov, i))
        cand.sort(key=lambda oi: (-oi[0], -self.counts[oi[1]]))
        cand = [i for _ov, i in cand[:60]]

        if not cand:  # safety fallback: most frequent entries in length window
            for lc in range(lo, hi + 1):
                cand.extend(self.by_len.get(lc, ())[:15])

        scored = []
        for i in cand:
            d = osa_distance(tk, self.tight_of[i], cap + 1)
            if d <= cap + 1:
                scored.append((i, d))
        scored.sort(key=lambda id_: (id_[1], -self.counts[id_[0]]))
        return scored

    def correct(
        self,
        raw: str,
        context_key: str | None = None,
        context_pairs: dict | None = None,
    ) -> Correction:
        """Resolve one OCR string.  context_* re-rank candidates with
        branch<->bccd co-occurrence counts (for the bccd field)."""
        if raw is None:
            return Correction(raw, None, "EMPTY", "sentinel", 0.0, True, [])
        s = raw.strip()
        if s.upper() in SENTINELS:
            return Correction(raw, None, "EMPTY", "sentinel", 0.0, True, [])

        hit = self._exact_stage(s)
        if hit:
            i, method, conf = hit
            return Correction(raw, self.values[i], conf, method, 1.0, True, [])

        scored = self._fuzzy_stage(s)
        tk = tight_key(s)
        cap = allowed_edits(len(tk))

        def pair_count(i: int) -> int:
            if context_pairs and context_key:
                return context_pairs.get((context_key, self.values[i]), 0)
            return 0

        if scored:
            # context re-rank: candidates seen together with the given branch
            # get a 1-edit bonus, but never overtake a clearly better distance
            def sort_key(idd):
                i, d = idd
                pc = pair_count(i)
                return (d - (1 if pc else 0), d, -pc, -self.counts[i])

            ranked = sorted(scored, key=sort_key)
            bi, bd = ranked[0]
            maxl = max(len(tk), len(self.tight_of[bi])) or 1
            ratio = 1.0 - bd / maxl
            if bd <= cap and (bd == 0 or ratio >= 0.65):
                # ambiguity: different value at the same effective distance?
                tied = [
                    i
                    for i, d in ranked[1:]
                    if (d - (1 if pair_count(i) else 0), d)
                    == (bd - (1 if pair_count(bi) else 0), bd)
                    and i != bi
                ]
                conf = "HIGH" if ratio >= 0.85 else ("MEDIUM" if ratio >= 0.75 else "LOW")
                if tied and conf != "LOW":
                    # frequency prior resolves the tie unless a rival is also
                    # frequent (then the read is genuinely ambiguous)
                    resolved = all(
                        self.counts[bi] >= 5 * self.counts[i] for i in tied
                    )
                    if not resolved:
                        conf = "MEDIUM" if conf == "HIGH" else "LOW"
                method = "pair" if pair_count(bi) and bd > 0 else "fuzzy"
                alts = [self.values[i] for i in tied[:3]]
                accept = conf in ("HIGH", "MEDIUM")
                return Correction(
                    raw, self.values[bi], conf, method, round(ratio, 4), accept, alts
                )
            # outside budget: still return nearest as a REVIEW SUGGESTION
            sug = self.values[bi]
            return Correction(
                raw, sug, "NONE", "fuzzy-out-of-budget", round(ratio, 4), False, []
            )

        # no fuzzy candidate at all
        return Correction(raw, None, "NONE", "unknown", 0.0, False, [])


# Placeholder dealer values ("BCCD-001", "BCCD-02", ...) are not real shops.
# Dropping them at load time means fuzzy matching
# can never surface a placeholder when a handwritten dealer is unreadable;
# the field stays empty and the pipeline's fallbacks get a chance to run.
_PLACEHOLDER_BCCD_RE = re.compile(r"^BCCD[\s\-_]*0*\d+$", re.IGNORECASE)


class Lexicon:
    """Both fields + co-occurrence pair table."""

    def __init__(self, branch_entries, bccd_entries, pair_counts):
        self.branch = FieldIndex(branch_entries)
        self.bccd = FieldIndex(bccd_entries, pair_counts=pair_counts)
        self.pair_counts = pair_counts
        # canonical branch value -> {bccd value: count}
        self.bccd_by_branch: dict[str, dict[str, int]] = defaultdict(dict)
        for (b, n), c in pair_counts.items():
            self.bccd_by_branch[b][n] = c

    @classmethod
    def load(cls, dict_dir: str = DEFAULT_DICT_DIR) -> "Lexicon":
        with open(os.path.join(dict_dir, "branch_lexicon.json"), encoding="utf-8") as f:
            branch = json.load(f)["entries"]
        with open(os.path.join(dict_dir, "bccd_lexicon.json"), encoding="utf-8") as f:
            bccd = json.load(f)["entries"]
        bccd = [
            e
            for e in bccd
            if not _PLACEHOLDER_BCCD_RE.match(str(e.get("value", "")).strip())
        ]
        with open(os.path.join(dict_dir, "pairs.json"), encoding="utf-8") as f:
            pj = json.load(f)["pairs"]
        pairs = {(b, n): int(c) for b, n, c in pj}
        return cls(branch, bccd, pairs)

    def correct_branch(self, raw: str) -> Correction:
        return self.branch.correct(raw)

    def correct_bccd(self, raw: str, branch: str | None = None) -> Correction:
        ctx = None
        ctx_pairs = None
        if branch:
            b = norm_key(branch)
            ctx = b
            # map the (possibly OCR'd) branch to canonical, then use its pairs
            bcorr = self.correct_branch(branch)
            canon = bcorr.value if bcorr.accept and bcorr.value else None
            if canon:
                ctx = canon
            by_br = self.bccd_by_branch.get(ctx)
            if by_br:
                ctx_pairs = {(ctx, nn): c for nn, c in by_br.items()}
            else:
                ctx_pairs = {}
        return self.bccd.correct(raw, context_key=ctx, context_pairs=ctx_pairs)

    def correct_row(self, branch_raw: str | None, bccd_raw: str | None):
        b = self.correct_branch(branch_raw) if branch_raw is not None else None
        branch_for_ctx = b.value if (b and b.accept and b.value) else branch_raw
        n = self.correct_bccd(bccd_raw, branch=branch_for_ctx) if bccd_raw is not None else None
        return b, n


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _print_correction(r: Correction, as_json: bool) -> None:
    if as_json:
        print(json.dumps(r._asdict(), ensure_ascii=False))
    else:
        print(f"  input       : {r.input!r}")
        print(f"  value       : {r.value!r}")
        print(f"  confidence  : {r.confidence}   accept={r.accept}")
        print(f"  method      : {r.method}   score={r.score}")
        if r.alternatives:
            print(f"  alternatives: {r.alternatives}")


def cmd_correct(lex: Lexicon, args) -> int:
    if args.field == "branch":
        r = lex.correct_branch(args.text)
    else:
        r = lex.correct_bccd(args.text, branch=args.branch)
    if args.strict:
        r = r._replace(accept=r.confidence in ("EXACT", "HIGH"))
    _print_correction(r, args.json)
    return 0 if r.accept else 2


def cmd_batch(lex: Lexicon, args) -> int:
    """JSONL stdin -> JSONL stdout.  Each line: {"branch": ..., "bccd": ...}."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        b, n = lex.correct_row(obj.get("branch"), obj.get("bccd"))
        out = {
            "branch_ocr": obj.get("branch"),
            "branch": b.value if b else None,
            "branch_confidence": b.confidence if b else None,
            "bccd_ocr": obj.get("bccd"),
            "bccd": n.value if n else None,
            "bccd_confidence": n.confidence if n else None,
            "review": bool((b and not b.accept) or (n and not n.accept)),
        }
        print(json.dumps(out, ensure_ascii=False))
    return 0


def cmd_csv(lex: Lexicon, args) -> int:
    """Correct the branch / bccd_name columns of a CSV, keeping originals."""
    br_col, bn_col = args.branch_col, args.bccd_col
    stats = Counter()
    t0 = time.perf_counter()
    n_rows = 0
    with open(args.input, newline="", encoding="utf-8-sig", errors="replace") as fin, \
         open(args.output, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        for extra in (
            "branch__corrected", "branch__confidence",
            "bccd_name__corrected", "bccd_name__confidence", "needs_review",
        ):
            if extra not in fieldnames:
                fieldnames.append(extra)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            n_rows += 1
            b, n = lex.correct_row(row.get(br_col), row.get(bn_col))
            if b:
                row["branch__corrected"] = b.value or ""
                row["branch__confidence"] = b.confidence
                stats["branch_" + b.confidence] += 1
            if n:
                row["bccd_name__corrected"] = n.value or ""
                row["bccd_name__confidence"] = n.confidence
                stats["bccd_" + n.confidence] += 1
            review = bool((b and not b.accept) or (n and not n.accept))
            row["needs_review"] = "yes" if review else ""
            if review:
                stats["needs_review_rows"] += 1
            writer.writerow(row)
    dt = time.perf_counter() - t0
    print(f"rows processed: {n_rows} in {dt:.2f}s "
          f"({n_rows / dt:,.0f} rows/s, {(dt / max(n_rows,1)) * 1000:.2f} ms/row)")
    for k in sorted(stats):
        print(f"  {k:22s} {stats[k]}")
    return 0


# ----------------------------------------------------------------------------
# Selftest: simulate OCR corruption of the known ground truth
# ----------------------------------------------------------------------------

CONFUSABLES = {
    "O": "0", "0": "O", "I": "1", "1": "I", "l": "1", "S": "5", "5": "S",
    "B": "8", "8": "B", "Z": "2", "2": "Z", "G": "6", "6": "G", "E": "F",
    "F": "E", "A": "4", "4": "A", "D": "0", "C": "(", "(": "C", ".": ",",
    ",": ".", "-": "", " ": "", "&": "8",
}
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .-"


def ocr_corrupt(v: str, rng: random.Random) -> str:
    """Produce a plausible OCR misread of a reference value."""
    s = v
    n_ops = 1 + rng.randint(0, 1) + (1 if len(s) > 10 else 0)
    for _ in range(n_ops):
        if not s:
            break
        op = rng.random()
        i = rng.randrange(len(s))
        if op < 0.25:                                   # deletion
            s = s[:i] + s[i + 1 :]
        elif op < 0.45:                                 # insertion
            s = s[:i] + rng.choice(ALPHA) + s[i:]
        elif op < 0.75:                                 # substitution (OCR confusable)
            c = s[i]
            s = s[:i] + CONFUSABLES.get(c, rng.choice(ALPHA)) + s[i + 1 :]
        elif op < 0.9 and i + 1 < len(s):               # adjacent transposition
            s = s[:i] + s[i + 1] + s[i] + s[i + 2 :]
        else:                                           # case flip
            s = s[:i] + (s[i].lower() if s[i].isupper() else s[i].upper()) + s[i + 1 :]
    return s


def cmd_selftest(lex: Lexicon, args) -> int:
    rng = random.Random(args.seed)
    results = []

    for field, index in (("branch", lex.branch), ("bccd", lex.bccd)):
        # sample canonical entries by frequency tier
        entries = list(zip(index.values, index.counts))
        hi = [e for e in entries if e[1] >= 10]
        mid = [e for e in entries if 3 <= e[1] < 10]
        lo = [e for e in entries if e[1] < 3]
        picks = []
        for tier, pool in (("high", hi), ("mid", mid), ("low", lo)):
            k = max(1, args.samples // 3)
            picks += [(tier, e) for e in rng.sample(pool, min(k, len(pool)))]
        for tier, (truth, _cnt) in picks:
            noisy = ocr_corrupt(truth, rng)
            if field == "branch":
                r = lex.correct_branch(noisy)
            else:
                # realistic: correct the branch first, then use it as context
                r = lex.correct_bccd(noisy, branch=None)
            ok = bool(r.value) and tight_key(r.value) == tight_key(truth)
            wrong = bool(r.value) and not ok
            results.append((field, tier, r, ok, wrong, truth, noisy))

    def report(rows):
        n = len(rows)
        acc = sum(1 for x in rows if x[3]) / n if n else 0
        accepted = [x for x in rows if x[2].accept]
        prec = sum(1 for x in accepted if x[3]) / len(accepted) if accepted else 0
        wrong = sum(1 for x in rows if x[4])
        return n, acc, len(accepted), prec, wrong

    print(f"selftest: {len(results)} corrupted samples (seed={args.seed})")
    print(f"{'field':8s}{'tier':7s}{'n':>6s}{'accuracy':>10s}{'accepted':>10s}"
          f"{'acc-rate':>10s}{'precision':>11s}{'wrong-fixes':>12s}")
    for field in ("branch", "bccd"):
        for tier in ("high", "mid", "low"):
            rows = [x for x in results if x[0] == field and x[1] == tier]
            if not rows:
                continue
            n, acc, na, prec, wrong = report(rows)
            print(f"{field:8s}{tier:7s}{n:6d}{acc * 100:9.1f}%{na:10d}"
                  f"{na / n * 100:9.1f}%{prec * 100:10.1f}%{wrong:12d}")
        rows = [x for x in results if x[0] == field]
        n, acc, na, prec, wrong = report(rows)
        print(f"{field:8s}{'ALL':7s}{n:6d}{acc * 100:9.1f}%{na:10d}"
              f"{na / n * 100:9.1f}%{prec * 100:10.1f}%{wrong:12d}")

    # latency on the full corpus values (exact-path dominant)
    all_branch = [v for v in lex.branch.values] * max(1, args.samples // len(lex.branch.values))
    t0 = time.perf_counter()
    for v in all_branch:
        lex.correct_branch(v)
    dt = time.perf_counter() - t0
    print(f"\nlatency (exact hits): {dt / max(len(all_branch),1) * 1e6:.0f} us/query, "
          f"{len(all_branch) / dt:,.0f} q/s")

    # latency on fully-corrupted inputs (fuzzy-path worst case)
    noisy_batch = [ocr_corrupt(v, rng) for v in all_branch[:2000]]
    t0 = time.perf_counter()
    for v in noisy_batch:
        lex.correct_branch(v)
    dt = time.perf_counter() - t0
    print(f"latency (fuzzy hits): {dt / max(len(noisy_batch),1) * 1000:.2f} ms/query, "
          f"{len(noisy_batch) / dt:,.0f} q/s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict-dir", default=DEFAULT_DICT_DIR, help="directory with *_lexicon.json / pairs.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("correct", help="correct one OCR string")
    c.add_argument("field", choices=["branch", "bccd"])
    c.add_argument("text")
    c.add_argument("--branch", help="known branch (context for bccd disambiguation)")
    c.add_argument("--json", action="store_true")
    c.add_argument("--strict", action="store_true", help="exit 2 unless EXACT/HIGH")

    b = sub.add_parser("batch", help="JSONL stdin -> JSONL stdout")
    v = sub.add_parser("vocab", help="print canonical vocabulary (prompt lexicon)")
    v.add_argument("field", choices=["branch", "bccd"])

    cs = sub.add_parser("csv", help="correct branch/bccd columns of a CSV")
    cs.add_argument("--in", dest="input", required=True)
    cs.add_argument("--out", dest="output", required=True)
    cs.add_argument("--branch-col", default="branch")
    cs.add_argument("--bccd-col", default="bccd_name")

    st = sub.add_parser("selftest", help="benchmark against simulated OCR errors")
    st.add_argument("--samples", type=int, default=3000)
    st.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    lex = Lexicon.load(args.dict_dir)

    if args.cmd == "correct":
        return cmd_correct(lex, args)
    if args.cmd == "batch":
        return cmd_batch(lex, args)
    if args.cmd == "csv":
        return cmd_csv(lex, args)
    if args.cmd == "vocab":
        idx = lex.branch if args.field == "branch" else lex.bccd
        sys.stdout.write("\n".join(idx.values) + "\n")
        return 0
    if args.cmd == "selftest":
        return cmd_selftest(lex, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
