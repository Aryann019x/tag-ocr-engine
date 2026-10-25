# Method

Notes on the techniques behind the engine, including the bugs that shaped them.
The examples here are generated.

## 1. Let the model transcribe, not decide

The OCR model is asked for plain text lines in reading order. Field assignment is
done by a deterministic mapper. This matters because an LLM asked to emit JSON
will happily invent a plausible serial number when the handwriting is unclear.
A mapper can only copy what the model actually returned, and validators can
reject what does not look like a serial.

The prompt is deliberately small:

```
Text Recognition: Transcribe ALL printed labels and handwritten values
line by line in reading order, including serial and complaint numbers.
Plain lines only, no JSON, no markdown, stop after one pass.
```

Generation settings: `temperature=0`, `num_predict≈300`, `repeat_penalty≈1.25`,
`num_ctx=2048`, plus stop sequences for markdown fences, the printed header and a
repeated `<table>` block. Tags are small; a bigger context only wastes VRAM.

## 2. Orientation scoring

A cheap morphology test decides whether the photo is sideways: threshold with
OTSU, open with a wide horizontal kernel and a tall vertical kernel, and compare
line energy. Sideways photos get candidates `(90, 270)`; upright photos get
`(0, 180)`.

Each candidate is OCR'd and scored:

```
score = number_of_keyword_hits + digit_count // 5
```

Keywords are the printed labels (`branch`, `bccd`, `serial`, `complaint`, …).
The winning rotation is used; if the losing pass found fields the winner missed,
those are recovered as `REVIEW`. This is what makes photos taken at any angle
work without a manual rotate step.

## 3. Label mapping

Labels are matched through a normalized fragment table, so `Bccd Name`,
`BCCDName`, `RCCD Name` and `Product Sr.No.` all resolve. A label may carry its
value on the same line or on the next non-label line.

Two rules prevent the classic failures:

- **Never drop a recognized label.** If a label fragment matches but the
  value-stripping regex does not cover that exact spelling (for example
  `Spare Parts Code`, plural), fall back to splitting at the first separator.
  Without this, the whole line — value included — was discarded.
- **Reject header lines early.** The printed header contains words like
  "Spare Part" and "Return"; without a guard it can be mistaken for a field
  label and bleed into values.

Torn labels are handled by position: if no `Branch` label exists, the first
unlabelled value is treated as the branch and the following one as the dealer.

## 4. Validators

- **Serial** — fixes common prefix misreads (`N804`, `N30U`, `al304` → `N304`),
  maps `U→4`, `S→5`, `O→0` inside the tail, and regroups separated digits:
  `N304. 0285.90742` → `N304-0285-90742`. Serials without separators are left
  untouched.
- **Complaint** — repairs `+1` / `.1` / `⊥` suffixes to `-1`, rejects short
  digit counts, and checks a known complaint-prefix table.
- **Date** — accepts the common handwritten formats and normalizes to
  `DD/MM/YYYY`.
- **Part code** — closed-set match against your catalogue with digit-confusion
  translation; the catalogue's description is then used as the product
  description when the tag left that row blank.

## 5. Entity resolution

Three layers, strongest first:

1. **Exact lexicon** — the canonical spelling from `data/*_lexicon.json`.
2. **Fuzzy** — OSA edit distance, with prefix/phonetic boosts. Guard rails:
   - reject candidates whose length differs by more than the edit-distance cap
     (a 6-character read cannot legitimately match a 19-character merged entry;
     without this guard, capped distance divided by the longer name fakes a high
     similarity and junk entries win);
   - require context agreement where available (dealer→branch pairs, dealer→
     technician rosters).
3. **Domain tables** — OCR alias maps, place aliases, complaint prefixes, and an
   exact record index (serial/complaint → record) that fills fields only
   when the tag itself was unreadable.

Statuses follow confidence: `EXACT`/`HIGH` → **READ**, medium → **REVIEW**,
no credible candidate → **UNREADABLE**. A wrong confident value is worse than a
blank field, because the operator stops looking.

### Handwritten dealer names in other scripts

Handwriting is sometimes returned in Devanagari, Bengali or as a Latin
transliteration. The engine keeps a **script-variant table**:
`HINDI_BCCD_SCRIPT_ALIASES` maps the exact observed script reading (after
stripping non-script characters) to a canonical dealer name. The table is empty
by default. Add the variants you actually see. This is data, not regex code, so
it is easy to audit and extend.

### The circular stamp rule

Many tags carry a round dealer stamp: the **place** is printed in the centre and
the **dealer name** runs around the rim. When the rim is too faint to read but
the centre place is legible, a place→dealer override can fill the dealer field
(`STAMP_PLACE_BCCD_OVERRIDES`, empty by default). Apply it only where a place has
exactly one dealer in your records; otherwise leave the field blank.

## 6. Confidence policy

- Blank is acceptable. Guessing silently is not.
- `REVIEW` means "a human should glance at this" — the UI highlights it.
- Every resolved field carries its candidates and scores, so disagreements are
  visible instead of hidden.

## 7. Extending

| Task | Where |
|---|---|
| New branch / dealer | add to your lexicon export (`data/`) |
| New OCR spelling of a dealer | `BCCD_OCR_ALIASES` |
| New script variant | `HINDI_BCCD_SCRIPT_ALIASES` |
| New part code | `PART_CODE_MASTER` |
| Complaint prefix → branch | `COMPLAINT_PREFIX_TO_BRANCH` |
| Place → sole dealer (stamp) | `STAMP_PLACE_BCCD_OVERRIDES` |
| New label spelling | `LABELS` in `pipeline.py` |

Performance levers: keep the model resident (`keep_alive`), keep `num_ctx` at
2048 for tag-sized inputs, and avoid running batch sweeps while interactive
scans are in progress on a small GPU.
