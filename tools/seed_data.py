"""Generate a small demo knowledge base so the app runs offline.

The shipped JSON files are fake on purpose: no real dealer, technician, branch
or complaint data. Replace them with an export from your own records to build
the same files.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

BRANCHES = ["Testnagar", "Demopur", "Samplepur", "Mockabad"]

DEALERS = {
    "Testnagar": ["Demo Electronics", "Sample Traders"],
    "Demopur": ["Example Enterprises", "Placeholder Sales"],
    "Samplepur": ["Mock Electricals", "Testline Services"],
    "Mockabad": ["Faux Traders"],
}

TECHS = {
    "Demo Electronics": ["Alex Demo", "Priya Sample"],
    "Sample Traders": ["Rahul Test"],
    "Example Enterprises": ["Neha Mock", "Alex Demo"],
    "Placeholder Sales": ["Rahul Test"],
    "Mock Electricals": ["Priya Sample"],
    "Testline Services": ["Neha Mock"],
    "Faux Traders": ["Alex Demo"],
}

PART_CODES = {
    "100001": "MAIN PCB ASSY DEMO 100",
    "100002": "DISPLAY PCB DEMO 200",
    "100003": "MOTOR ASSY DEMO 300",
    "100004": "HEATER COIL DEMO 400",
    "100005": "CONTROL PANEL DEMO 500",
}


def tight(value: str) -> str:
    return "".join(ch for ch in value.upper() if ch.isalnum())


def variant_entry(value: str, count: int, variants: list[str]) -> dict:
    return {
        "value": value,
        "count": count,
        "variants": [{"text": v, "count": max(1, count // (i + 2))} for i, v in enumerate(variants)],
    }


def main() -> None:
    branch_entries = []
    for i, branch in enumerate(BRANCHES):
        branch_entries.append(
            variant_entry(
                branch,
                60 - i * 8,
                [branch, branch.upper(), branch.replace("a", "a ").strip()],
            )
        )

    bccd_entries = []
    pairs = []
    for bi, (branch, dealers) in enumerate(DEALERS.items()):
        for di, dealer in enumerate(dealers):
            count = 30 - bi * 4 - di * 3
            variants = [dealer, dealer.upper(), dealer.replace("Electronics", "Electronic")]
            bccd_entries.append(variant_entry(dealer, count, variants))
            pairs.append([branch, dealer, count])

    branch_lex = {
        "field": "Branch",
        "source": "demo",
        "generated_utc": "2026-01-01T00:00:00Z",
        "entries": branch_entries,
        "unique_raw": len(branch_entries),
    }
    bccd_lex = {
        "field": "BCCD Name",
        "source": "demo",
        "generated_utc": "2026-01-01T00:00:00Z",
        "entries": bccd_entries,
        "unique_raw": len(bccd_entries),
    }
    pair_doc = {"source": "demo", "pairs": pairs}

    tech_entries = []
    for name, count in [("Alex Demo", 12), ("Priya Sample", 9), ("Rahul Test", 7), ("Neha Mock", 5)]:
        tech_entries.append({"tk": tight(name), "value": name, "count": count})

    bccd_tech_map = {
        tight(dealer): [[tech, 4 - i] for i, tech in enumerate(techs)]
        for dealer, techs in TECHS.items()
    }
    branch_tech_map = {
        tight(branch): [[tech, 3 - i] for i, tech in enumerate(TECHS[dealers[0]])]
        for branch, dealers in DEALERS.items()
    }
    bccd_branch_map = {
        tight(dealer): [[branch, 4]]
        for branch, dealers in DEALERS.items()
        for dealer in dealers
    }
    comp_prefix_map = {"TST": "Testnagar", "DMO": "Demopur", "SMP": "Samplepur", "MCK": "Mockabad"}

    record = {"br": "Testnagar", "bc": "Demo Electronics", "pc": "100001", "vt": "Alex Demo"}
    kb = {
        "stats": {
            "rows": 0,
            "branches": len(BRANCHES),
            "bccds": len(bccd_entries),
            "technicians": len(tech_entries),
        },
        "tech_entries": tech_entries,
        "bccd_tech_map": bccd_tech_map,
        "branch_tech_map": branch_tech_map,
        "bccd_branch_map": bccd_branch_map,
        "comp_prefix_map": comp_prefix_map,
        "serial_index": {"N304123456789": record},
        "complaint_index": {"TST120576057963441": record},
    }

    for name, payload in [
        ("branch_lexicon.json", branch_lex),
        ("bccd_lexicon.json", bccd_lex),
        ("pairs.json", pair_doc),
        ("master_knowledge.json", kb),
    ]:
        (DATA / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote data/{name}")


if __name__ == "__main__":
    main()
