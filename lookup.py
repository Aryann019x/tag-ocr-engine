"""Entity resolution for handwritten form tags.

Layers:
1. Lexicon (`data/*_lexicon.json`) - canonical branch/dealer values plus
   co-occurrence counts.
2. Knowledge base (`data/master_knowledge.json`) - technician rosters,
   dealer->branch pairs, complaint prefixes and record indexes.
3. Closed-set part-code resolver + validators for serial, complaint and
   date fields.

The files in `data/` are examples. Point the resolver at your own export to use
it on real forms.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

TOOLKIT_DIR = Path(__file__).parent / "data"
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

try:
    from ocr_lookup import Lexicon, norm_key, osa_distance, tight_key
except ImportError:
    Lexicon = None
    norm_key = None
    osa_distance = None
    tight_key = None

# Demo closed-set part codes and their canonical product descriptions
PART_CODE_MASTER = {
    # Closed-set demo catalogue. Replace with your own part-code master.
    "100001": "MAIN PCB ASSY DEMO 100",
    "100002": "DISPLAY PCB DEMO 200",
    "100003": "MOTOR ASSY DEMO 300",
    "100004": "HEATER COIL DEMO 400",
    "100005": "CONTROL PANEL DEMO 500",
}

# Demo technician roster
KNOWN_ENGINEERS = [
    # Demo roster. Add your own visiting engineers here.
    "Alex Demo", "Priya Sample", "Rahul Test",
]

# Digit confusion map for OCR errors
DIGIT_FIXES = {
    "O": "0", "o": "0", "Q": "0", "D": "0", "U": "0", "u": "0",
    "I": "1", "i": "1", "l": "1", "|": "1", "!": "1", "+": "1", "t": "1", "⊥": "1",
    "Z": "2", "z": "2",
    "E": "3",
    "A": "4", "y": "4", "w": "4",
    "S": "5", "s": "5", "$": "5",
    "G": "6", "b": "6",
    "T": "7", "f": "7", "F": "7", "J": "7",
    "B": "8",
    "g": "9", "q": "9",
}

# Common prefixes in Tag Complaint Numbers -> Canonical Branch
COMPLAINT_PREFIX_TO_BRANCH = {
    # Demo complaint prefixes -> demo branches.
    "TST": "Testnagar",
    "DMO": "Demopur",
    "SMP": "Samplepur",
    "MCK": "Mockabad",
}

BRANCH_DISPLAY_NORMALIZATION = {
    "TESTNAGAR": "Testnagar",
    "DEMOPUR": "Demopur",
    "SAMPLEPUR": "Samplepur",
    "MOCKABAD": "Mockabad",
}

GENERIC_BCCD_WORDS = {
    "enterprises", "enterprise", "ent", "ents", "electricals", "electrical",
    "elec", "elc", "electronics", "electronic", "service", "services",
    "center", "centre", "care", "solutions", "solution", "traders", "trader",
    "sales", "sals", "agency", "agencies", "works", "workshop", "associate",
    "associates", "point", "house", "world", "corp", "corporation", "pvt",
    "ltd", "for", "m", "s", "sri", "shri", "jai", "shree", "group", "co",
}

# Explicit high-frequency OCR / Devanagari / abbreviated BCCD aliases observed on tags
BCCD_OCR_ALIASES = {
    # OCR spelling -> canonical dealer. Ships empty; add your own aliases.
}

BCCD_BRANCH_OVERRIDES = {
    # Dealer-specific branch overrides, e.g. "DEMOELECTRONICS": "TESTNAGAR".
}

# Branch must contain a place, never a dealer/shop name.  These are the
# business words that distinguish BCCD/shop names from city names.
BUSINESS_WORD_RE = re.compile(
    r"\b(?:"
    r"enterprises?|ents?|electricals?|elec|electronics?|services?|servicing|"
    r"traders?|trading|sales|sals|shops?|stores?|centres?|centers?|"
    r"agenc(?:y|ies)|works|workshop|associates?|solutions?|"
    r"communication|telecom|rewinders?|distributors?|suppliers?|marketing|"
    r"appliances?|customer\s+care|care|group|pvt|ltd|corp|corporation"
    r")\b",
    re.IGNORECASE,
)

# Product/PCB text sometimes lands in Branch or BCCD because of a vertical
# stamp shift.  It must not be treated as a dealer or a city.
PRODUCT_WORD_RE = re.compile(
    r"\b(?:"
    r"pcb|infrared|infra\s*red|"
    r"induction|cooktop|main\s+pcb|display\s+pcb|spare\s+part|"
    r"model|molef|product\s+description"
    r")\b",
    re.IGNORECASE,
)

# Labels that can remain glued to a value when the model writes two fields on
# one line (e.g. "BCCIName Demo Electronics..." or "CD Name ... CCD Location ...").
BCCD_LABEL_RE = re.compile(
    r"\b(?:bcc[i]?|bccd|a?ccd|cd|rccd|sccd)\s*(?:name|location)?\b",
    re.IGNORECASE,
)

BCCD_NOISE = {
    "LOCALION", "LOCALLION", "LOCATION", "NAME", "BCCD", "CCD", "BRANCH",
    "ITEM", "NIL", "NULL", "N/A",
}

HEADER_NOISE_RE = re.compile(
    r"^(?:TAG\s+)?(?:DETECTIVE|DEFECTIVE|DETECT|DEFECT|SPARE\s+PART|RETURN)\b",
    re.IGNORECASE,
)

ADDRESS_TAIL_RE = re.compile(
    r"\b(?:"
    r"housing|colony|road|street|nagar|ward|near|behind|"
    r"opp|opposite|shop|plot|sector|block|floor|complex|mlp"
    r")\b.*$",
    re.IGNORECASE,
)

# Extra place spellings that are not present in the demo branch lexicon (or are
# common OCR variants). The demo lexicon can contain noisy entries, so these
# aliases are kept separately and applied with high priority.
EXTRA_PLACE_ALIASES = {
    # Place aliases -> canonical branch name.
    "TEST NAGAR": "Testnagar",
    "DEMO PUR": "Demopur",
}


# Transliterated spellings produced by the Devanagari reader below.  These
# are kept separate from EXTRA_PLACE_ALIASES so the existing Latin spellings
# remain untouched.
HINDI_TRANSLIT_PLACE_ALIASES = {
    # Devanagari/transliterated place spellings -> canonical branch.
}

DEVANAGARI_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o",
    "ऍ": "e",
}
DEVANAGARI_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क़": "q",
    "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh",
    "फ़": "f", "य़": "y",
}
DEVANAGARI_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
    "्": "", "ं": "n", "ँ": "n", "ः": "h",
}
DEVANAGARI_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

HINDI_BUSINESS_RE = re.compile(
    r"इलेक्ट|इलेक|इलेव|एलेक|सर्विस|एंटर|ट्रेड|शॉप|स्टोर|एजेंस|"
    r"वर्क|सेंटर|अप्ला|डिस्ट्री|मार्केट"
)
TRANSLIT_BUSINESS_RE = re.compile(
    r"\b(?:ilekt|elekt|ilek|servis|entar|entr|tred|shop|stor|agen|"
    r"werk|sent|applian|distrib|market|sals|sales)\w*",
    re.IGNORECASE,
)

# Whole-line handwritten variants that the OCR model has returned for the
# local Demo Electronics tag.  These are kept exact so that ordinary Hindi
# text in a product field is not turned into a dealer name.  Keys contain only
# Devanagari/Bengali letters, matras and digits (spaces/punctuation removed).
HINDI_BCCD_SCRIPT_ALIASES = {
    # Handwritten dealer names in Devanagari/Bengali script -> canonical dealer.
    # Ships empty: add your own variants (see docs/method.md).
}


def _script_only(text: str) -> str:
    """Keep only Devanagari/Bengali letters, matras and digits."""
    return re.sub(r"[^\u0900-\u097F\u0980-\u09FF]", "", text or "")


def _known_script_bccd_alias(text: str) -> str:
    """Return a curated dealer name for an exact observed script reading."""
    key = _script_only(text)
    return HINDI_BCCD_SCRIPT_ALIASES.get(key, "")


def _has_devanagari(s: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", s or ""))


def _has_bengali(s: str) -> bool:
    return bool(re.search(r"[\u0980-\u09FF]", s or ""))


def transliterate_devanagari(text: str) -> str:
    """Best-effort Hindi -> Latin transliteration for tag values.

    This is deliberately small and deterministic: it handles the printed and
    handwritten Hindi dealer/city words seen on these tags without adding a
    second OCR model or a heavyweight dependency.
    """
    if not text:
        return ""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in DEVANAGARI_DIGITS:
            out.append(DEVANAGARI_DIGITS[ch])
            i += 1
            continue
        if ch in DEVANAGARI_VOWELS:
            out.append(DEVANAGARI_VOWELS[ch])
            i += 1
            continue
        if ch in DEVANAGARI_CONSONANTS:
            out.append(DEVANAGARI_CONSONANTS[ch])
            if i + 1 < n and text[i + 1] == "्":
                i += 2
                continue
            if i + 1 < n and text[i + 1] in ("ं", "ँ", "ः"):
                # Anusvara/visarga follow the inherent vowel: पं -> pan,
                # not pn.  This matters for handwritten place names.
                out.append("a")
                out.append(DEVANAGARI_MATRAS[text[i + 1]])
                i += 2
                continue
            if i + 1 < n and text[i + 1] in DEVANAGARI_MATRAS:
                out.append(DEVANAGARI_MATRAS[text[i + 1]])
                i += 2
                continue
            # Keep the inherent vowel only inside a word.  Word-final
            # consonants are usually silent enough for matching purposes.
            if i + 1 < n and (
                text[i + 1] in DEVANAGARI_CONSONANTS
                or text[i + 1] in DEVANAGARI_VOWELS
            ):
                out.append("a")
            i += 1
            continue
        if ch in DEVANAGARI_MATRAS:
            out.append(DEVANAGARI_MATRAS[ch])
            i += 1
            continue
        if ch in ("।", "|"):
            out.append(" ")
            i += 1
            continue
        out.append(ch)
        i += 1

    value = "".join(out)
    value = re.sub(r"[^A-Za-z0-9]+", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def _transliterate_devanagari_bccd(s: str) -> str:
    """Return a curated dealer name for a Devanagari-script reading.

    Observed OCR spellings live in ``HINDI_BCCD_SCRIPT_ALIASES`` (data, not
    code), so a deployment extends the table instead of editing regexes.
    See docs/method.md for how these variants are built.
    """
    return _known_script_bccd_alias(s)


def _transliterate_bengali_bccd(s: str) -> str:
    """Same mechanism for readings that arrive in Bengali script."""
    return _known_script_bccd_alias(s)


class EntityResolver:
    def __init__(self, dict_dir: Path | None = None):
        if dict_dir is None:
            dict_dir = TOOLKIT_DIR
        self.dict_dir = dict_dir
        self.lexicon = Lexicon.load(str(dict_dir)) if Lexicon else None

        self.kb = {
            "tech_entries": [],
            "bccd_tech_map": {},
            "branch_tech_map": {},
            "bccd_branch_map": {},
            "comp_prefix_map": {},
            "serial_index": {},
            "complaint_index": {},
        }
        kb_path = dict_dir / "master_knowledge.json"
        if kb_path.is_file():
            try:
                self.kb = json.loads(kb_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Fast exact lookup map for technicians
        self.tech_exact: dict[str, dict] = {}
        for entry in self.kb.get("tech_entries", []):
            self.tech_exact[entry["tk"]] = entry

        self.pair_counts_upper = {(b.upper(), n): c for (b, n), c in (self.lexicon.pair_counts if self.lexicon else {}).items()}

        # Precompute core-token BCCD index for instant O(1)-filtered fallback
        self._bccd_core_entries: list[tuple[int, str, str, int]] = []
        if self.lexicon:
            for i, val in enumerate(self.lexicon.bccd.values):
                cnt = self.lexicon.bccd.counts[i]
                if cnt < 2:
                    continue
                v_words = [w for w in re.findall(r"[a-z0-9]+", val.lower()) if w not in GENERIC_BCCD_WORDS and len(w) >= 2]
                if v_words:
                    self._bccd_core_entries.append((i, val, v_words[0], cnt))

        # Place index used to keep Branch free of dealer/shop names and to split
        # values such as "Demo Traders TESTNAGAR" or
        # "Demo Electronics12 Main Road,Demopur".
        self._place_aliases: dict[str, tuple[str, int, int]] = {}
        self._build_place_index()

    @staticmethod
    def _place_key(value: str) -> str:
        """Uppercase, punctuation-insensitive key used for place matching."""
        if norm_key:
            value = norm_key(value)
        else:
            value = (value or "").upper().replace("&", " AND ")
        return " ".join(re.findall(r"[A-Z0-9]+", value or ""))

    def _build_place_index(self) -> None:
        aliases: dict[str, tuple[str, int, int]] = {}

        def add(alias: str, canonical: str, priority: int = 10, count: int = 0) -> None:
            key = self._place_key(alias)
            if not key or len(key.replace(" ", "")) < 3:
                return
            prev = aliases.get(key)
            if prev is None or priority > prev[1] or (priority == prev[1] and count > prev[2]):
                aliases[key] = (canonical, priority, count)

        if self.lexicon:
            for i, value in enumerate(self.lexicon.branch.values):
                count = self.lexicon.branch.counts[i]
                if count < 3 or not self._is_place_candidate(value):
                    continue
                canonical = BRANCH_DISPLAY_NORMALIZATION.get(value.upper(), value)
                add(value, canonical, priority=10, count=count)
                # Comma/slash-separated places should expose both tokens.
                for part in re.split(r"[,/]", value):
                    if part.strip() and part.strip().upper() != value.upper():
                        add(part.strip(), part.strip(), priority=8, count=count)

        for alias, canonical in EXTRA_PLACE_ALIASES.items():
            add(alias, canonical, priority=100, count=9999)
            add(canonical, canonical, priority=100, count=9999)

        # Latin spellings produced by transliterate_devanagari().
        for alias, canonical in HINDI_TRANSLIT_PLACE_ALIASES.items():
            add(alias, canonical, priority=100, count=9999)
            add(canonical, canonical, priority=100, count=9999)

        self._place_aliases = aliases
        ordered = sorted(aliases, key=lambda item: (len(item), aliases[item][1], aliases[item][2]), reverse=True)
        self._place_direct_re = (
            re.compile(
                r"(?<![A-Z0-9])(" + "|".join(re.escape(alias) for alias in ordered) + r")(?![A-Z0-9])",
                re.IGNORECASE,
            )
            if ordered
            else None
        )
        self._place_suffix_aliases = [alias for alias in ordered if " " not in alias and len(alias) >= 4]

    @staticmethod
    def _is_place_candidate(value: str) -> bool:
        value = (value or "").strip()
        if not value or len(value) > 40 or _has_devanagari(value):
            return False
        if len(re.findall(r"[A-Za-z]", value)) < 3:
            return False
        if value.upper() in {"INDIA", "UNKNOWN"}:
            return False
        if BUSINESS_WORD_RE.search(value):
            return False
        return True

    @lru_cache(maxsize=4096)
    def _find_place_match(self, text: str) -> tuple[str, str]:
        """Return (canonical place, matched alias) for the best place in text."""
        if not text or not self._place_aliases:
            return "", ""
        if _has_devanagari(text):
            text = transliterate_devanagari(text)
        norm = self._place_key(text)
        if not norm:
            return "", ""

        direct: list[tuple[int, int, int, str, str]] = []
        if self._place_direct_re:
            for match in self._place_direct_re.finditer(norm):
                alias = match.group(1)
                data = self._place_aliases.get(alias)
                if data:
                    canonical, priority, count = data
                    direct.append((len(alias), priority, count, canonical, alias))
        if direct:
            best = max(direct, key=lambda item: (item[0], item[1], item[2]))
            return best[3], best[4]

        # Handles glued forms such as "DEMO ELECTRONICSSAMPLEPUR" and
        # "TSAMPLEPUR" where the place is the tail of a longer OCR token.
        suffix: list[tuple[int, int, int, str, str]] = []
        suffix_set = set(self._place_suffix_aliases)
        for token in re.findall(r"[A-Z0-9]+", norm):
            if len(token) < 5:
                continue
            for start in range(1, len(token) - 3):
                alias = token[start:]
                if alias in suffix_set:
                    canonical, priority, count = self._place_aliases[alias]
                    suffix.append((len(alias), priority, count, canonical, alias))
        if suffix:
            best = max(suffix, key=lambda item: (item[0], item[1], item[2]))
            return best[3], best[4]
        return "", ""

    def extract_place(self, text: str) -> str:
        """Extract the strongest known city/place from free text, else ''."""
        return self._find_place_match(text)[0]

    @lru_cache(maxsize=4096)
    def _clean_bccd_value(self, text: str) -> str:
        """Remove labels and phone numbers that stick to Branch/BCCD values."""
        value = (text or "").strip(" ?.-|")
        if not value:
            return ""
        value = BCCD_LABEL_RE.sub(" ", value)
        value = re.sub(
            r"\b(?:M|MOB|MOBILE|PHONE|TEL)\s*[.:]*\s*\d{8,12}\b",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"\b(?:NAME|LOCATION)\b", " ", value, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", value).strip(" ,.-|")

    def _remove_place_aliases(self, text: str) -> str:
        """Remove known place tokens from a combined field value."""
        value = text or ""
        if self._place_direct_re:
            value = self._place_direct_re.sub(" ", value)
        # One glued suffix such as "ELECTRONICSTESTNAGAR" or "TSAMPLEPUR".
        alias = self._find_place_match(value)[1]
        if alias and " " not in alias and len(alias) >= 4:
            value = re.sub(
                rf"(?<=[A-Z0-9]){re.escape(alias)}(?![A-Z0-9])",
                " ",
                value,
                flags=re.IGNORECASE,
            )
        return value

    @lru_cache(maxsize=4096)
    def _extract_business_text(self, text: str) -> str:
        """Return the dealer/shop part of a combined Branch/BCCD value."""
        value = self._clean_bccd_value(text)
        if not value:
            return ""
        value = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", value)
        value = self._remove_place_aliases(value)
        value = ADDRESS_TAIL_RE.sub(" ", value)
        value = re.sub(
            r"\b(?:DESCRIPTION|PRODUCT|MODEL|MOLEF)\b.*$",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"\b(?:INFRARED|INFRA\s*RED|"
            r"INDUCTION|COOKTOP|PCB)\b.*$",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"[()\[\]]", " ", value)
        value = re.sub(r"\b(?:M|MOB|MOBILE|PHONE|TEL)\s*[.:]*\s*\d{8,12}\b", " ", value, flags=re.IGNORECASE)
        value = re.sub(r"\b\d+\b", " ", value)
        value = re.sub(
            r"^(?:APPLICATION|FOR|M/S|MS)\b[\s:.\-|]*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", value).strip(" ,.-|&")

    @lru_cache(maxsize=4096)
    def _is_place_only(self, text: str) -> bool:
        return bool(self.extract_place(text)) and not bool(self._extract_business_text(text))

    @lru_cache(maxsize=4096)
    def _has_business_keyword(self, text: str) -> bool:
        value = (text or "").strip()
        if not value:
            return False
        if _has_devanagari(value):
            if HINDI_BUSINESS_RE.search(value):
                return True
            value = transliterate_devanagari(value)
        # "ENTERPRISES46" and "ELECTRONICSTESTNAGAR" are common glued OCR forms.
        probe = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", value)
        if BUSINESS_WORD_RE.search(probe) or TRANSLIT_BUSINESS_RE.search(probe):
            return True
        residual = self._extract_business_text(value)
        return bool(
            residual
            and (BUSINESS_WORD_RE.search(residual) or TRANSLIT_BUSINESS_RE.search(residual))
        )

    @lru_cache(maxsize=4096)
    def _is_business_text(self, text: str) -> bool:
        value = (text or "").strip()
        if not value:
            return False
        if self._has_business_keyword(value):
            return True
        residual = self._extract_business_text(value)
        if residual:
            words = re.findall(r"[A-Za-z]{2,}", residual)
            if len(words) >= 2:
                return True
        if self._is_place_only(value):
            return False
        if self.lexicon:
            try:
                corr = self.lexicon.correct_bccd(value)
            except Exception:
                return False
            if corr.confidence in ("EXACT", "HIGH") and not self.extract_place(value):
                return True
        return False

    @lru_cache(maxsize=4096)
    def _is_product_only(self, text: str) -> bool:
        value = (text or "").strip()
        if not value or BUSINESS_WORD_RE.search(value):
            return False
        if self.extract_place(value):
            return False
        return bool(PRODUCT_WORD_RE.search(value))

    def _is_known_bccd(self, text: str) -> bool:
        if not text or not self.lexicon:
            return False
        try:
            corr = self.lexicon.correct_bccd(text)
        except Exception:
            return False
        return corr.confidence in ("EXACT", "HIGH") and bool(corr.value)

    def _place_at_end(self, text: str, alias: str) -> bool:
        """True when the place is the trailing part of a dealer value."""
        norm = self._place_key(text)
        if not norm or not alias:
            return False
        tail = re.sub(r"\b(?:MLP|M|MOB|MOBILE|PHONE|TEL)\b", " ", norm)
        tail = re.sub(r"\b\d+\b", " ", tail)
        tail = re.sub(r"\s+", " ", tail).strip()
        return bool(tail) and tail.endswith(alias)

    def normalize_branch_bccd(self, raw_mapped: dict) -> dict:
        """Separate place (Branch) from dealer/shop (BCCD) before resolution.

        Handles the common stamp-shift cases:
        - "Sample Traders" / "Testnagar" -> Branch Testnagar, BCCD Sample Traders
        - "Demo Traders" / "TESTNAGAR" -> Branch Testnagar, BCCD Demo Traders
        - "DEMO ELECTRONICSTESTNAGAR (MLP) 000000" -> Branch Testnagar, BCCD Demo Electronics
        - "Demo Electronics12 Main Road,Demopur" -> Branch Demopur, BCCD Demo Electronics
        - product text in BCCD -> discarded/rescued to Product Description
        """
        out = dict(raw_mapped)
        orig_branch = out.get("Branch", "")
        orig_bccd = out.get("BCCD Name", "")
        hindi_source = _has_devanagari(orig_branch) or _has_devanagari(orig_bccd)
        bengali_source = _has_bengali(orig_branch) or _has_bengali(orig_bccd)

        branch = self._clean_bccd_value(orig_branch)
        bccd = self._clean_bccd_value(orig_bccd)
        # Convert Hindi values before the normal place/dealer split.  Known
        # dealer patterns map to their English canonical name; everything else
        # is transliterated so the existing place index and lexicon can work.
        if _has_devanagari(branch):
            branch = (
                _transliterate_devanagari_bccd(branch)
                or transliterate_devanagari(branch)
            )
        elif _has_bengali(branch):
            branch = _transliterate_bengali_bccd(branch) or branch
        if _has_devanagari(bccd):
            bccd = (
                _transliterate_devanagari_bccd(bccd)
                or transliterate_devanagari(bccd)
            )
        elif _has_bengali(bccd):
            bccd = _transliterate_bengali_bccd(bccd) or bccd
        product = (out.get("Product Description", "") or "").strip()
        rescued_product = ""

        # Printed tag headers are not Branch/BCCD values.
        if branch and HEADER_NOISE_RE.match(branch):
            branch = ""
        if bccd and HEADER_NOISE_RE.match(bccd):
            bccd = ""

        # A product/PCB line shifted into either field is not a place or dealer.
        if branch and self._is_product_only(branch):
            rescued_product = rescued_product or branch
            branch = ""
        if bccd and self._is_product_only(bccd):
            rescued_product = rescued_product or bccd
            bccd = ""

        # 1) BCCD is only a place/address (e.g. "TESTNAGAR" or "SAMPLEPUR (MLP) 000000").
        #    If Branch held the dealer name, this is a swap, not a deletion.
        bccd_place, bccd_alias = self._find_place_match(bccd)
        if bccd and bccd_place and self._is_place_only(bccd):
            branch_business = ""
            if self._is_business_text(branch):
                branch_business = self._extract_business_text(branch)
            if not self.extract_place(branch) or branch_business:
                branch = bccd_place
            bccd = branch_business

        # 2) Branch contains both a place and a dealer.  Keep the place and move
        #    the dealer text to BCCD unless BCCD already has a dealer.
        branch_place, branch_alias = self._find_place_match(branch)
        if branch_place and self._is_business_text(branch):
            business = self._extract_business_text(branch)
            branch = branch_place
            if business and (not bccd or not self._is_business_text(bccd)):
                bccd = business
        elif branch_place:
            branch = branch_place

        # 3) BCCD can itself be "Dealer + Place" (e.g. "Demo Traders TESTNAGAR").
        bccd_place, bccd_alias = self._find_place_match(bccd)
        if bccd_place and self._is_business_text(bccd):
            business = self._extract_business_text(bccd)
            if business:
                if not branch:
                    branch = bccd_place
                if self._place_at_end(bccd, bccd_alias) or not self._is_known_bccd(bccd):
                    bccd = business
        elif bccd and bccd_place and self._is_place_only(bccd):
            if not branch:
                branch = bccd_place
            bccd = ""

        # 4) Branch is still a dealer with no place (e.g. "Mock Services
        #    Service" or "Demo Sales").  Move it to BCCD and leave Branch
        #    empty so it can be inferred from BCCD/complaint.
        branch_place, _ = self._find_place_match(branch)
        if branch and not branch_place and self._is_business_text(branch):
            business = self._extract_business_text(branch)
            bccd_has_identity = bool(
                bccd
                and (
                    self._is_known_bccd(bccd)
                    or self._has_business_keyword(bccd)
                )
            )
            if business and not bccd_has_identity:
                bccd = business
            branch = ""

        # 5) If BCCD is still empty, rescue a dealer name that shifted into a
        #    neighbouring field.  This is common on short tags: the BCCD line is
        #    read as Technician Name or Nature Of Defect.
        if not bccd:
            label_re = re.compile(
                r"\b(?:bcc[i]?|a?ccd|cd)\s*(?:name|location)\b",
                re.IGNORECASE,
            )
            for key in ("Technician Name", "Nature Of Defect", "Product Description"):
                source = (out.get(key, "") or "").strip()
                if not source:
                    continue
                label_hit = bool(label_re.search(source))
                candidate = self._clean_bccd_value(source)
                if not candidate:
                    continue
                if label_hit or self._has_business_keyword(candidate):
                    if self._is_place_only(candidate) or self._is_product_only(candidate):
                        continue
                    bccd = self._extract_business_text(candidate) or candidate
                    out[key] = ""
                    break

        # 6) Final cleanup.  Never leave a dealer/product/phone in Branch.
        branch_place, _ = self._find_place_match(branch)
        if branch_place:
            branch = branch_place
        if bccd and (self._is_place_only(bccd) or self._is_product_only(bccd)):
            bccd = ""
        branch = self._clean_bccd_value(branch)
        bccd = self._clean_bccd_value(bccd)

        # Hindi OCR often lands within one edit of a known BCCD.  Canonicalize
        # it here so the UI shows the normal English dealer name, not the raw
        # transliteration.
        if (hindi_source or bengali_source) and bccd and self.lexicon:
            corr = self.lexicon.correct_bccd(bccd)
            if (
                corr.value
                and corr.confidence in ("EXACT", "HIGH", "MEDIUM")
                and corr.score >= 0.78
            ):
                bccd = corr.value

        if rescued_product and not product:
            out["Product Description"] = rescued_product
        out["Branch"] = branch
        out["BCCD Name"] = bccd
        return out

    def lookup_exact_record(self, raw_serial: str, raw_complaint: str) -> dict | None:
        """Check if this exact Serial No or Complaint No exists in the demo knowledge base."""
        sr_key = re.sub(r"[^A-Za-z0-9]", "", raw_serial or "").upper()
        cp_key = re.sub(r"[^A-Za-z0-9]", "", raw_complaint or "").upper()
        if sr_key and sr_key in self.kb.get("serial_index", {}):
            return {"matched_by": "serial", "key": sr_key, **self.kb["serial_index"][sr_key]}
        if cp_key and cp_key in self.kb.get("complaint_index", {}):
            return {"matched_by": "complaint", "key": cp_key, **self.kb["complaint_index"][cp_key]}
        return None

    def infer_branch_from_text(self, text: str) -> str:
        """Detect known branch city name or complaint prefix inside text."""
        if not text:
            return ""
        up = text.strip().upper()
        # Check complaint prefix (e.g., TST120..., DMO120..., SMP120..., MCK120...)
        m = re.match(r"^\s*([A-Z]{3,6})\s*\d{5,}", up)
        if m:
            pref4 = m.group(1)[:4]
            pref3 = m.group(1)[:3]
            if pref4 in COMPLAINT_PREFIX_TO_BRANCH:
                return COMPLAINT_PREFIX_TO_BRANCH[pref4]
            if pref3 in COMPLAINT_PREFIX_TO_BRANCH:
                return COMPLAINT_PREFIX_TO_BRANCH[pref3]
            kb_prefs = self.kb.get("comp_prefix_map", {})
            if pref4 in kb_prefs:
                return kb_prefs[pref4]
            if pref3 in kb_prefs:
                return kb_prefs[pref3]

        # Check place names inside address text (e.g. "12 Main Road,
        # Demopur", "Demo Traders TESTNAGAR", "CD Name ... CCD Location Samplepur").
        place = self.extract_place(text)
        if place:
            return place
        return ""

    def infer_branch_from_bccd(self, bccd_canonical: str) -> tuple[str, float]:
        """Infer canonical Branch from the resolved dealer using co-occurrence counts."""
        if not bccd_canonical or not tight_key:
            return "", 0.0
        tk = tight_key(bccd_canonical)
        if tk in BCCD_BRANCH_OVERRIDES:
            return BCCD_BRANCH_OVERRIDES[tk], 0.95
        branches = self.kb.get("bccd_branch_map", {}).get(tk, [])
        if not branches:
            return "", 0.0
        top_br, top_cnt = branches[0]
        if top_br.upper() in BRANCH_DISPLAY_NORMALIZATION:
            top_br = BRANCH_DISPLAY_NORMALIZATION[top_br.upper()]
        total = sum(c for _, c in branches)
        ratio = top_cnt / max(total, 1)
        score = round(min(0.96, 0.80 + min(top_cnt, 50) * 0.003 + ratio * 0.1), 3)
        return top_br, score

    def resolve_branch(
        self,
        raw_branch: str,
        complaint_hint: str = "",
        bccd_hint: str = "",
        resolved_bccd: str = "",
    ) -> dict:
        raw_clean = (raw_branch or "").strip(" ?.-|")
        if _has_devanagari(raw_clean):
            raw_clean = (
                _transliterate_devanagari_bccd(raw_clean)
                or transliterate_devanagari(raw_clean)
            )
        up_clean = raw_clean.upper()

        comp_branch = self.infer_branch_from_text(complaint_hint)
        bccd_inferred_br, bccd_br_sc = self.infer_branch_from_bccd(resolved_bccd or bccd_hint)
        bccd_branch = bccd_inferred_br if bccd_br_sc >= 0.75 else ""

        # Known cursive / handwritten city misreads.
        if up_clean in ("SMPL", "SMPL.", "SMP"):
            return {
                "value": "Samplepur",
                "status": "READ",
                "confidence": "HIGH",
                "score": 0.94,
                "candidates": [{"canonical": "Samplepur", "score": 0.94}],
            }

        # Branch is a place only.  This deliberately runs before the fuzzy
        # branch lexicon: that lexicon also contains rare contaminated entries
        # such as dealer names and "India".
        place = self.extract_place(raw_clean)
        if place:
            val = BRANCH_DISPLAY_NORMALIZATION.get(place.upper(), place)
            # When both the complaint prefix AND the resolved BCCD independently
            # agree on a different branch (e.g. a place alias on a dealer tag /
            # TST120... tag), let that unanimous cross-field agreement win.
            if (
                comp_branch
                and bccd_branch
                and comp_branch.upper() == bccd_branch.upper()
                and comp_branch.upper() != val.upper()
                and bccd_br_sc >= 0.90
            ):
                agreed = BRANCH_DISPLAY_NORMALIZATION.get(comp_branch.upper(), comp_branch)
                return {
                    "value": agreed,
                    "status": "READ",
                    "confidence": "CROSS_VERIFIED",
                    "score": 0.96,
                    "candidates": [
                        {"canonical": agreed, "score": 0.96},
                        {"canonical": val, "score": 0.88},
                    ],
                }
            candidates = [{"canonical": val, "score": 0.96}]
            if comp_branch and comp_branch.upper() != val.upper():
                candidates.append({"canonical": comp_branch, "score": 0.92})
            if bccd_branch and bccd_branch.upper() != val.upper():
                candidates.append({"canonical": bccd_branch, "score": bccd_br_sc})
            return {
                "value": val,
                "status": "READ",
                "confidence": "PLACE",
                "score": 0.96,
                "candidates": candidates[:4],
            }

        is_noise = (
            not raw_clean
            or up_clean in ("?", "NA", "NULL", "ITEM", "DESCRIPTION", "LOCATION")
            or bool(re.fullmatch(r"[\d\s\-]+", raw_clean))
            or up_clean.startswith("DEFECT")
            or bool(HEADER_NOISE_RE.match(raw_clean))
            or _has_devanagari(raw_clean)
        )
        is_business = bool(raw_clean) and self._is_business_text(raw_clean)

        def inferred_result(source: str) -> dict | None:
            if bccd_branch:
                return {
                    "value": bccd_branch,
                    "status": "READ",
                    "confidence": "INFERRED_FROM_BCCD",
                    "score": bccd_br_sc,
                    "candidates": [{"canonical": bccd_branch, "score": bccd_br_sc}],
                }
            if comp_branch:
                return {
                    "value": comp_branch,
                    "status": "READ",
                    "confidence": "INFERRED",
                    "score": 0.94,
                    "candidates": [{"canonical": comp_branch, "score": 0.94}],
                }
            return None

        # Dealer text, product text, phone numbers, labels and empty cells are
        # never valid Branch values.
        if is_noise or is_business:
            result = inferred_result("noise" if is_noise else "business")
            if result:
                return result
            return {"value": "", "status": "UNREADABLE", "confidence": "NONE", "score": 0.0, "candidates": []}

        # Strip trailing stray digits like "For Mock Services Service 24".
        raw_for_lookup = re.sub(r"\s+\d+$", "", raw_clean).strip() or raw_clean

        if self.lexicon:
            corr = self.lexicon.correct_branch(raw_for_lookup)
            candidate = corr.value if corr.confidence in ("EXACT", "HIGH", "MEDIUM") else ""
            candidate_is_place = bool(candidate) and bool(self.extract_place(candidate))
            if candidate_is_place and not self._is_business_text(candidate):
                val = BRANCH_DISPLAY_NORMALIZATION.get(candidate.upper(), candidate)
                status = "READ" if corr.confidence in ("EXACT", "HIGH") else "REVIEW"
                score = round(corr.score, 3)
                candidates = [{"canonical": val, "score": score}]
                if comp_branch and comp_branch.upper() != val.upper():
                    candidates.append({"canonical": comp_branch, "score": 0.92})
                if bccd_branch and bccd_branch.upper() != val.upper():
                    candidates.append({"canonical": bccd_branch, "score": bccd_br_sc})
                for item in (corr.alternatives or [])[:3]:
                    alt_val = item if isinstance(item, str) else item[0]
                    alt_sc = round(corr.score * 0.9, 3) if isinstance(item, str) else round(float(item[1]), 3)
                    if self.extract_place(alt_val) and not self._is_business_text(alt_val):
                        alt_val = BRANCH_DISPLAY_NORMALIZATION.get(alt_val.upper(), alt_val)
                        if all(c["canonical"] != alt_val for c in candidates):
                            candidates.append({"canonical": alt_val, "score": alt_sc})
                return {
                    "value": val,
                    "status": status,
                    "confidence": corr.confidence,
                    "score": score,
                    "candidates": candidates[:4],
                }

        # Unknown text: a resolved BCCD is a much better branch hint than
        # echoing an unrecognised handwritten word into Branch.
        result = inferred_result("unknown")
        if result:
            return result
        return {"value": "", "status": "UNREADABLE", "confidence": "NONE", "score": 0.0, "candidates": []}

    def _core_token_bccd_candidates(self, raw_clean: str, branch: str = "") -> list[dict]:
        """Fallback candidate finder when the second word of a BCCD name is corrupted by handwriting."""
        if not self.lexicon or not osa_distance:
            return []
        words = [w for w in re.findall(r"[a-z0-9]+", raw_clean.lower()) if w not in GENERIC_BCCD_WORDS and len(w) >= 3]
        if not words:
            return []
        first_tok = words[0]
        ft_len = len(first_tok)
        br_up = branch.upper() if branch else ""
        scored = []
        for _i, val, v_w0, cnt in self._bccd_core_entries:
            if abs(ft_len - len(v_w0)) > 2:
                continue
            d0 = osa_distance(first_tok, v_w0, 2)
            if d0 > 2:
                continue
            sim0 = 1.0 - d0 / max(ft_len, len(v_w0), 1)
            if sim0 >= 0.72:
                pair_boost = 0.10 if br_up and (br_up, val) in self.pair_counts_upper else 0.0
                freq_boost = min(cnt, 100) * 0.0018
                sc = round(min(0.96, sim0 * 0.85 + pair_boost + freq_boost), 3)
                scored.append((sc, cnt, val))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        out = []
        seen = set()
        for sc, _, val in scored[:4]:
            if val not in seen:
                seen.add(val)
                out.append({"canonical": val, "score": sc})
        return out

    def resolve_bccd(self, raw_bccd: str, branch: str = "", tech_hint: str = "", complaint_hint: str = "") -> dict:
        from_devanagari = _has_devanagari(raw_bccd)
        from_bengali = _has_bengali(raw_bccd)
        # Dataset placeholders ("BCCD-001", ...) are never a dealer name.
        if re.match(r"^BCCD[\s\-_]*0*\d+$", (raw_bccd or "").strip(), flags=re.IGNORECASE):
            return {"value": "", "status": "UNREADABLE", "confidence": "NONE", "score": 0.0, "candidates": []}
        raw_clean = self._clean_bccd_value(raw_bccd)
        if (
            not raw_clean
            or raw_clean.upper() in BCCD_NOISE
            or raw_clean.upper() in ("?", "NA")
            or HEADER_NOISE_RE.match(raw_clean)
            or re.match(r"^BCCD[\s\-_]*0*\d+$", raw_clean, flags=re.IGNORECASE)
        ):
            return {"value": "", "status": "UNREADABLE", "confidence": "NONE", "score": 0.0, "candidates": []}

        # A pure city/address is Branch material, never a dealer name.
        place, place_alias = self._find_place_match(raw_clean)
        if place and self._is_place_only(raw_clean):
            return {"value": "", "status": "UNREADABLE", "confidence": "NONE", "score": 0.0, "candidates": []}

        # A dealer written as "Demo Traders TESTNAGAR" is split before lookup.
        if place and self._is_business_text(raw_clean) and self._place_at_end(raw_clean, place_alias):
            business = self._extract_business_text(raw_clean)
            if business:
                raw_clean = business

        # Product/PCB text shifted into BCCD is not a dealer.
        if self._is_product_only(raw_clean):
            return {"value": "", "status": "UNREADABLE", "confidence": "NONE", "score": 0.0, "candidates": []}

        # 1. Check known Hindi dealer patterns, then transliterate the rest so
        # the normal BCCD lexicon can match it.
        dev_hit = _transliterate_devanagari_bccd(raw_clean) or _transliterate_bengali_bccd(raw_clean)
        if dev_hit:
            return {
                "value": dev_hit,
                "status": "READ",
                "confidence": "HIGH",
                "score": 0.95,
                "candidates": [{"canonical": dev_hit, "score": 0.95}],
            }
        if _has_devanagari(raw_clean):
            raw_clean = transliterate_devanagari(raw_clean)

        # 2. Check the explicit OCR-alias table (BCCD_OCR_ALIASES).
        tk_raw = tight_key(raw_clean) if tight_key else re.sub(r"[^A-Z0-9]", "", raw_clean.upper())
        if tk_raw in BCCD_OCR_ALIASES:
            canon = BCCD_OCR_ALIASES[tk_raw]
            return {
                "value": canon,
                "status": "READ",
                "confidence": "HIGH",
                "score": 0.95,
                "candidates": [{"canonical": canon, "score": 0.95}],
            }

        if not self.lexicon:
            return {"value": raw_clean, "status": "REVIEW", "confidence": "RAW", "score": 0.5, "candidates": []}

        corr = self.lexicon.correct_bccd(raw_clean, branch=branch or None)
        # Placeholder values ("BCCD-001", ...) are dataset artifacts, never a
        # real shop.  Treat them as unreadable even if the lexicon still holds
        # one, so the field stays blank instead of showing a fake dealer.
        if corr.value and re.match(
            r"^BCCD[\s\-_]*0*\d+$", str(corr.value).strip(), flags=re.IGNORECASE
        ):
            return {
                "value": "",
                "status": "UNREADABLE",
                "confidence": "NONE",
                "score": 0.0,
                "candidates": [],
            }
        status = "READ" if corr.confidence in ("EXACT", "HIGH") else ("REVIEW" if corr.confidence == "MEDIUM" else "UNREADABLE")
        if (from_devanagari or from_bengali) and status == "REVIEW" and corr.value and corr.score >= 0.78:
            status = "READ"

        candidates = []
        if corr.value:
            candidates.append({"canonical": corr.value, "score": round(corr.score, 3)})
        if corr.alternatives:
            for item in corr.alternatives[:3]:
                alt_val = item if isinstance(item, str) else item[0]
                alt_sc = round(corr.score * 0.9, 3) if isinstance(item, str) else round(float(item[1]), 3)
                if alt_val != corr.value:
                    candidates.append({"canonical": alt_val, "score": alt_sc})

        # Core-token fallback for multi-word handwritten BCCD names
        if status in ("UNREADABLE", "REVIEW") or corr.score < 0.86:
            tok_cands = self._core_token_bccd_candidates(raw_clean, branch=branch)
            for tc in tok_cands:
                if all(c["canonical"] != tc["canonical"] for c in candidates):
                    candidates.append(tc)
            if tok_cands:
                best_tc = tok_cands[0]
                # If the core token and branch pair agree strongly, promote to READ.
                if best_tc["score"] >= 0.85 or (status == "UNREADABLE" and best_tc["score"] >= 0.72):
                    new_status = "READ" if best_tc["score"] >= 0.85 else "REVIEW"
                    candidates.sort(key=lambda x: -x["score"])
                    return {
                        "value": best_tc["canonical"],
                        "status": new_status,
                        "confidence": "HIGH" if new_status == "READ" else "MEDIUM",
                        "score": best_tc["score"],
                        "candidates": candidates[:4],
                    }

        # If a valid multi-word dealer name (e.g., "Demo Group Centre") is not yet in the demo lexicon, preserve cleanly as REVIEW
        if status == "UNREADABLE" and any(w in raw_clean.lower() for w in ("centre", "center", "enterprises", "electrical", "electronics", "service", "sales", "group", "traders")):
            titled = " ".join(w.upper() if len(w) <= 2 else w.capitalize() for w in raw_clean.split())
            return {
                "value": titled,
                "status": "REVIEW",
                "confidence": "NEW_ENTITY",
                "score": 0.75,
                "candidates": [{"canonical": titled, "score": 0.75}] + candidates[:3],
            }

        return {
            "value": corr.value if status in ("READ", "REVIEW") else raw_clean,
            "status": status,
            "confidence": corr.confidence,
            "score": round(corr.score, 3),
            "candidates": candidates[:4],
        }

    def resolve_part_code(
        self,
        raw_code: str,
        fallback_text: str = "",
        defect_text: str = "",
        bccd_name: str = "",
    ) -> dict:
        """Resolve against 17 known part codes with digit confusion mapping + model/defect inference."""
        combined = f"{raw_code or ''} {fallback_text or ''}".strip()
        clean = (raw_code or "").strip(" ?.-|")
        if clean.upper() in ("NAME", "CODE", "SPARE", "PART"):
            clean = ""

        # 1. Check exact hit in raw_code or fallback_text
        for pc in PART_CODE_MASTER:
            if pc in combined:
                return {
                    "value": pc,
                    "status": "READ",
                    "score": 1.0,
                    "candidates": [{"code": pc, "desc": PART_CODE_MASTER[pc], "score": 1.0}],
                    "derived_description": PART_CODE_MASTER[pc],
                }

        # 2. Translate confusable characters to digits.
        # 3. Translate confusable characters to digits token-by-token
        tokens = re.findall(r"[A-Za-z0-9+$!|]{5,8}", combined)
        best_match = None
        best_score = 0.0
        scored_list = []

        for tok in tokens:
            trans_chars = []
            for ch in tok:
                if ch.isdigit():
                    trans_chars.append(ch)
                elif ch in DIGIT_FIXES:
                    trans_chars.append(DIGIT_FIXES[ch])
            cand = "".join(trans_chars)
            if len(cand) < 5 or len(cand) > 7:
                continue
            for pc in PART_CODE_MASTER:
                if cand == pc:
                    score = 0.98
                elif len(cand) == 6:
                    matches = sum(1 for a, b in zip(cand, pc) if a == b)
                    score = matches / 6.0
                else:
                    d = osa_distance(cand, pc, 2) if osa_distance else 3
                    score = max(0.0, 1.0 - d / 6.0)
                if score >= 0.65:
                    # Boost score if fallback_text / defect_text confirms Main vs Display PCB
                    low_ctx = f"{fallback_text} {defect_text}".lower()
                    if "main" in low_ctx and "MAIN PCB" in PART_CODE_MASTER[pc]:
                        score = min(0.98, score + 0.18)
                    elif "display" in low_ctx and "DISPLAY PCB" in PART_CODE_MASTER[pc]:
                        score = min(0.98, score + 0.18)
                    scored_list.append((score, pc))
                    if score > best_score:
                        best_score = score
                        best_match = pc

        scored_list.sort(key=lambda x: -x[0])
        unique_cands = []
        seen_pc = set()
        for sc, pc in scored_list:
            if pc not in seen_pc:
                seen_pc.add(pc)
                unique_cands.append({"code": pc, "desc": PART_CODE_MASTER[pc], "score": round(sc, 2)})

        if best_match and best_score >= 0.65:
            status = "READ" if best_score >= 0.82 else "REVIEW"
            return {
                "value": best_match,
                "status": status,
                "score": round(best_score, 2),
                "candidates": unique_cands[:3],
                "derived_description": PART_CODE_MASTER[best_match],
            }

        return {
            "value": clean,
            "status": "UNREADABLE",
            "score": 0.0,
            "candidates": [],
            "derived_description": "",
        }

    def resolve_technician(self, raw_tech: str, bccd_name: str = "", branch: str = "") -> dict:
        """Resolve Technician Name using the dealer roster, branch roster and global technician roster."""
        clean = (raw_tech or "").strip(" ?.-|")
        clean_low = clean.lower()

        # Reject serial numbers, part codes, or defect strings accidentally mapped into Technician Name
        is_invalid_tech = (
            not clean
            or clean.upper() in ("?", "NA", "NULL", "||", "NAME", "TECHNICIAN")
            or any(w in clean_low for w in ("working", "dead", "defect", "heating", "display", "pcb", "problem"))
            or bool(re.search(r"\d{4,}", clean))
        )

        bc_tk = tight_key(bccd_name) if (bccd_name and tight_key) else ""
        br_tk = tight_key(branch) if (branch and tight_key) else ""
        bccd_techs = self.kb.get("bccd_tech_map", {}).get(bc_tk, [])
        branch_techs = self.kb.get("branch_tech_map", {}).get(br_tk, [])

        if is_invalid_tech:
            # If the technician row was blank or an illegible scribble (e.g., a
            # scribbled dealer name for Alex Demo), check whether this BCCD has
            # a single known technician in the demo roster.
            if bccd_techs:
                top_tech, top_cnt = bccd_techs[0]
                cands = [{"canonical": t, "score": round(min(0.92, 0.75 + c * 0.02), 2)} for t, c in bccd_techs[:3]]
                # Only auto-fill if the BCCD has a single dominant technician and the tag had a scribble or single-tech dealer
                return {"value": "", "status": "UNREADABLE", "score": 0.0, "candidates": cands}
            return {"value": "", "status": "UNREADABLE", "score": 0.0, "candidates": []}

        tk_raw = tight_key(clean) if tight_key else re.sub(r"[^A-Z0-9]", "", clean.upper())

        # 1. Score the dealer-specific technician roster first: handwritten
        #    tech names usually resolve only in their dealer context.
        scored_cands = []
        raw_first = clean_low.split()[0] if clean_low.split() else clean_low

        for tech_val, cnt in bccd_techs:
            t_tk = tight_key(tech_val)
            t_first = tech_val.lower().split()[0]
            if abs(len(tk_raw) - len(t_tk)) <= 5:
                d_full = osa_distance(tk_raw, t_tk, 5) if osa_distance else 5
                sim_full = 1.0 - d_full / max(len(tk_raw), len(t_tk), 1)
            else:
                # A length gap beyond the edit-distance cap can never be a real
                # match; without this guard the capped distance divided by the
                # longer name fakes a high similarity for junk merged entries.
                sim_full = 0.0
            d_first = osa_distance(raw_first, t_first, 4) if osa_distance else 4
            sim_first = 1.0 - d_first / max(len(raw_first), len(t_first), 1)
            # Phonetic/prefix similarity: OCR often keeps the first syllable.
            pref_match = 0.82 if (len(raw_first) >= 3 and len(t_first) >= 3 and raw_first[:3] == t_first[:3]) else 0.0
            if raw_first[:2] == t_first[:2] and len(raw_first) >= 4 and len(t_first) >= 4:
                pref_match = max(pref_match, 0.72)
            sim = max(sim_full, sim_first, pref_match)
            if sim >= 0.55:
                freq_bonus = min(0.15, cnt * 0.015)
                final_sc = round(min(0.99, sim + freq_bonus), 2)
                scored_cands.append((final_sc, cnt, tech_val))

        # 2. Check Branch-specific technician roster
        for tech_val, cnt in branch_techs:
            t_tk = tight_key(tech_val)
            t_first = tech_val.lower().split()[0]
            if abs(len(tk_raw) - len(t_tk)) <= 4:
                d_full = osa_distance(tk_raw, t_tk, 4) if osa_distance else 4
                sim_full = 1.0 - d_full / max(len(tk_raw), len(t_tk), 1)
            else:
                sim_full = 0.0
            d_first = osa_distance(raw_first, t_first, 3) if osa_distance else 3
            sim_first = 1.0 - d_first / max(len(raw_first), len(t_first), 1)
            sim = max(sim_full, sim_first)
            if sim >= 0.70:
                freq_bonus = min(0.10, cnt * 0.008)
                final_sc = round(min(0.98, sim + freq_bonus), 2)
                scored_cands.append((final_sc, cnt, tech_val))

        # 3. Check exact hit in the global technician lexicon
        if tk_raw in self.tech_exact:
            entry = self.tech_exact[tk_raw]
            scored_cands.append((0.99, entry["count"], entry["value"]))

        # 4. Check fuzzy match against the global technician roster
        if not scored_cands or max(x[0] for x in scored_cands) < 0.88:
            for entry in self.kb.get("tech_entries", [])[:600]:
                t_tk = entry["tk"]
                if abs(len(tk_raw) - len(t_tk)) > 3:
                    continue
                d = osa_distance(tk_raw, t_tk, 3) if osa_distance else 4
                sim = 1.0 - d / max(len(tk_raw), len(t_tk), 1)
                if sim >= 0.76:
                    sc = round(min(0.96, sim + min(0.06, entry["count"] * 0.002)), 2)
                    scored_cands.append((sc, entry["count"], entry["value"]))

            for eng in KNOWN_ENGINEERS:
                eng_low = eng.lower()
                eng_first = eng_low.split()[0]
                if eng_low == clean_low or eng_first == raw_first:
                    scored_cands.append((1.0, 500, eng))
                elif osa_distance:
                    d = osa_distance(raw_first, eng_first, 2)
                    sim = 1.0 - d / max(len(raw_first), len(eng_first), 1)
                    if sim >= 0.75:
                        scored_cands.append((round(sim, 2), 100, eng))

        scored_cands.sort(key=lambda x: (-x[0], -x[1]))
        unique_cands = []
        seen_t = set()
        for sc, _cnt, val in scored_cands:
            tk_v = tight_key(val) if tight_key else val.upper()
            if tk_v not in seen_t:
                seen_t.add(tk_v)
                unique_cands.append({"canonical": val, "score": sc})

        if unique_cands and unique_cands[0]["score"] >= 0.74:
            best = unique_cands[0]
            status = "READ" if best["score"] >= 0.82 else "REVIEW"
            return {
                "value": best["canonical"],
                "status": status,
                "score": best["score"],
                "candidates": unique_cands[:4],
            }

        return {
            "value": clean,
            "status": "REVIEW",
            "score": unique_cands[0]["score"] if unique_cands else 0.5,
            "candidates": unique_cands[:4],
        }

    def validate_date(self, raw_date: str) -> dict:
        """Validate a date and return it in canonical Indian DD/MM/YYYY form.

        Tags are written DD/MM/YYYY.  OCR/human entry sometimes produces
        MM/DD/YYYY (for example 10/31/2025) or YYYY/MM/DD; when one component
        is unambiguously a month we reorder it.  A genuinely ambiguous value
        such as 05/06/2025 is left in DD/MM order because swapping it would be
        guesswork.

        Values are returned with slashes and a four-digit year so the UI and
        CSV are consistent.
        """
        if not raw_date or not raw_date.strip():
            return {"value": "", "status": "UNREADABLE"}

        clean = raw_date.strip(" ?.-|_→।")
        if not clean or clean.upper() in (
            "?", "NA", "NULL", "-", "___", "H/U", "H/P", "ILLEGIBLE", "I/W", "IW",
        ):
            return {"value": "", "status": "UNREADABLE"}

        # Repair common handwritten/OCR digit confusions, but only next to
        # digits so normal words are not damaged.
        clean = re.sub(r"(?<=\d)[OoQq](?=\d)", "0", clean)
        clean = re.sub(r"(?<=\d)[lI](?=\d)", "1", clean)
        clean = re.sub(r"(?<=\d)[OoQq](?=\D|$)", "0", clean)
        clean = re.sub(r"(?<=\d)[lI](?=\D|$)", "1", clean)

        # Normalize separators like "16/9-25", "16:9-25" or "31.10.2025".
        clean = re.sub(r"[:/\.]+", "-", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        # Fix a common OCR slash -> '1' glitch in 8-9 digit dates such as
        # "510612024" -> "5/06/2024".
        m_glitched = re.fullmatch(r"(\d{1,2})1(0[1-9]|1[0-2])1[23](0\d\d|\d\d)", clean)
        if m_glitched:
            clean = f"{m_glitched.group(1)}/{m_glitched.group(2)}/2{m_glitched.group(3)}"

        # Handwritten '5' is often read as '8' in a two-digit year:
        # "09-08-28" -> "09-08-25".
        clean = re.sub(r"([/\-])28$", r"\g<1>25", clean)

        def _valid(day: int, month: int, year: int) -> bool:
            if not 2000 <= year <= 2100:
                return False
            try:
                date(year, month, day)
            except ValueError:
                return False
            return True

        def _year(value: int) -> int:
            if value >= 100:
                return value
            return 2000 + value if value <= 49 else 1900 + value

        def _fmt(day: int, month: int, year: int) -> dict:
            return {"value": f"{day:02d}/{month:02d}/{year}", "status": "READ"}

        # Three-part dates: DD/MM/YYYY, MM/DD/YYYY, YYYY/MM/DD.
        m = re.search(
            r"(?<!\d)(\d{1,4})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,4})(?!\d)",
            clean,
        )
        if m:
            a, b, c = (int(m.group(i)) for i in (1, 2, 3))
            if len(m.group(1)) == 4:
                year, month, day = a, b, c
            else:
                day, month, year = a, b, c
            year = _year(year)

            # If the middle value cannot be a month but the first can, the
            # entry is in MM/DD order.  This is the common "date swapped"
            # human/OCR error.
            if month > 12 and day <= 12:
                day, month = month, day

            if _valid(day, month, year):
                return _fmt(day, month, year)
            return {"value": clean, "status": "REVIEW"}

        # Month/year only (some tags have no day).
        m = re.fullmatch(r"(\d{1,2})\s*[-/]\s*(\d{4})", clean)
        if m:
            month, year = int(m.group(1)), int(m.group(2))
            if 1 <= month <= 12 and 2000 <= year <= 2100:
                return {"value": f"{month:02d}/{year}", "status": "READ"}

        # Compact dates.
        m = re.fullmatch(r"(\d{2})(\d{2})(\d{4})", clean)
        if m:
            day, month, year = (int(m.group(i)) for i in (1, 2, 3))
            if _valid(day, month, year):
                return _fmt(day, month, year)

        m = re.fullmatch(r"(\d{2})(\d{2})(\d{2})", clean)
        if m:
            day, month, year = (int(m.group(i)) for i in (1, 2, 3))
            year = _year(year)
            if _valid(day, month, year):
                return _fmt(day, month, year)

        return {"value": clean, "status": "REVIEW"}

    def validate_serial(self, raw_sr: str) -> dict:
        """Serial numbers are typically alphanumeric starting with N304... or E201... (10-16 chars)."""
        if not raw_sr or not raw_sr.strip() or raw_sr.strip() in ("?", "NA", "NULL"):
            return {"value": "", "status": "UNREADABLE"}

        val = raw_sr.strip(" ?.-|")
        # Fix common prefix OCR reads:
        # 'al304', '01304', '1304', 'N804', 'N30U', 'N302' -> 'N304'
        val = re.sub(r"^(?:al|01|1|n|N)\s*304", "N304", val, flags=re.IGNORECASE)
        val = re.sub(r"^N\s*[83]0[4U2]", "N304", val, flags=re.IGNORECASE)
        # Fix 'U7861' -> '47861' or 'S301' -> '5301' inside serial tail
        if val.upper().startswith("N304"):
            prefix, tail = val[:4], val[4:]
            tail = tail.replace("U", "4").replace("u", "4").replace("S", "5").replace("O", "0")
            val = prefix + tail

        # The model renders handwritten separators as dots/spaces
        # ("N304. 0285.90742"); when separators are present, present the classic
        # N304-XXXX-XXXXX grouping.  Serials without separators stay untouched.
        if re.search(r"[\s.\u00b7\-]", val[4:]):
            m_group = re.fullmatch(r"(?i)N304[\s.\u00b7\-]*(\d{4})[\s.\u00b7\-]*(\d{5})", val)
            if m_group:
                val = f"N304-{m_group.group(1)}-{m_group.group(2)}"

        clean = re.sub(r"[^A-Za-z0-9]", "", val)
        if len(clean) >= 9 and sum(c.isdigit() for c in clean) >= 7:
            return {"value": val, "status": "READ"}
        return {"value": val, "status": "REVIEW"}

    def validate_complaint(self, raw_comp: str, branch: str = "") -> dict:
        """Complaint numbers have digits and optional city prefix (e.g. DEMO21072606329904-1)."""
        if not raw_comp or not raw_comp.strip() or raw_comp.strip() in ("?", "NA", "NULL"):
            return {"value": "", "status": "UNREADABLE"}

        clean = raw_comp.strip(" ?.-|'")
        # Fix '+1', '. ⊥', '. 1', '+2' suffix OCR read for '-1' / '-2' (only when separated by space, dot, +, or -)
        clean = re.sub(r"[\s.+\-]+([⊥|!l1])$", "-1", clean)
        clean = re.sub(r"\+(\d)$", r"-\1", clean)
        clean = re.sub(r"-{2,}", "-", clean)
        # Fix 19-char complaint numbers ending in '11' or '41' where the dash was read as '1' or '4'
        if re.fullmatch(r"[A-Za-z]{4}\d{13,14}[14]1", clean):
            clean = clean[:-2] + "-1"
        elif re.fullmatch(r"\d{15,16}1", clean) and not clean.endswith("-1"):
            clean = clean[:-1] + "-1"

        digits = sum(1 for c in clean if c.isdigit())
        if digits >= 6:
            return {"value": clean, "status": "READ"}
        return {"value": clean, "status": "REVIEW"}
