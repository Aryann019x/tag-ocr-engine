# Benchmark

`python -m bench` runs one or more Ollama vision models over the same images and
labels, then scores the nine fields. It separates model quality from mapping
quality and counts the failure that matters most: a value marked `READ` that is
wrong.

## Quick start

```bash
ollama pull glm-ocr:q8_0
python -m bench --images samples --labels samples/labels.csv \
  --models glm-ocr:q8_0
```

The report goes to `reports/<timestamp>/summary.md`, with per-field details in
`details.csv`.

Compare models in one run:

```bash
python -m bench --images samples --labels samples/labels.csv \
  --models glm-ocr:q8_0,glm-ocr:bf16
```

Models are warmed up before scoring by default so latency is comparable. Add
`--skip-warmup` for a quick accuracy check.

## Labels

The labels file is a CSV with a `file` column and the nine field names from
`pipeline.KEYS`:

```csv
file,Branch,BCCD Name,Product Description,Product Sr. No.,Date of Purchase,Complaint No.,Spare Part Code,Nature Of Defect,Technician Name
sample_01_upright.jpg,Testnagar,Demo Electronics,MAIN PCB ASSY DEMO 100,N304123456789,12/03/2024,TST12057605796344-1,100001,Dead,Alex Demo
```

Values are the expected canonical output, not the raw print on the tag. Dates
are `DD/MM/YYYY`; serial and complaint numbers may be written with separators on
the tag.

The three sample labels are synthetic. For real numbers, label reviewed tags
and keep the CSV and transcripts private. `*.csv` is ignored by default;
`samples/labels.csv` is the only tracked exception.

## Metrics

- **Full row**: every scored field on that image was correct.
- **Critical**: serial, complaint number, part code, and purchase date.
- **False READ**: marked `READ` but wrong. This is the number to keep at zero.
- **READ precision**: of the fields marked `READ`, how many were correct.
- **REVIEW recall**: of the wrong values, how many were flagged `REVIEW` or
  left `UNREADABLE`.
- **OCR miss**: the expected value was not in the transcript at all.
- **Mapper miss**: the value was in the transcript but the mapper missed it.
- **p50 / p95**: warm scan time across the images.

A model with high field accuracy and one false `READ` on a serial number is worse
than a model with slightly lower accuracy and a clean review flag. The report
keeps those separate.

## Limits

- Three demo images are enough to check the wiring, not to rank models. Treat
  percentages from small samples as noisy.
- All models use the same OCR prompt. This is not a prompt-engineering bake-off.
- Latency depends on the machine, model load, and image size. Compare models on
  the same run, not across runs. Set `TAGOCR_GPU_LAYERS=0` to measure CPU-only.
- Do not tune the mapper on the same labels you report. Keep a holdout set.
- FUNSD, SROIE, CORD, and IAM are public OCR datasets, but they contain forms,
  receipts, and handwriting lines. They do not share this return-tag layout, so
  they cannot score the nine-field mapper without a schema adapter. Use them for
  raw OCR comparison only; label your own tags for field accuracy.
- Raw transcripts are only written with `--save-raw`. They can contain dealer or
  customer data, so keep them out of the repository.
