"""Interactive web application for local tag OCR + entity resolution.

Runs on local GLM-OCR (Ollama) with zero cloud cost and an offline master lookup.
Optimized for fast warm scans:
- Vision patch grid tuned to 896x560 + num_ctx=2048 + stop tokens
- Scan & Go Instant Mode: clicking a sample, dropping/uploading a photo, pasting (Ctrl+V),
  or capturing from camera immediately fires the scan and appends to the live session table.
- One-click batch run over the bundled samples + CSV / JSON export.
"""
import csv
import io
import json
import os
import re
import shutil
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, Response, send_from_directory

import pipeline
import config
from lookup import PART_CODE_MASTER

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
MAX_TRAINING_PACK_ROWS = 500
MAX_TRAINING_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TRAINING_PACK_UPLOAD_BYTES = 100 * 1024 * 1024
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

SAMPLE_DIR = BASE_DIR / "samples"
SAMPLE_DIR.mkdir(exist_ok=True)
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tag OCR</title>
<style>
:root {
  color-scheme: dark;
  --primary: #f5f5f5;
  --primary-hover: #ffffff;
  --accent: #d4d4d4;
  --bg: #080808;
  --surface: rgba(255, 255, 255, 0.045);
  --card: rgba(255, 255, 255, 0.055);
  --text: #f5f5f5;
  --text-muted: #a1a1aa;
  --border: rgba(255, 255, 255, 0.10);
  --success: #e4e4e7;
  --warning: #c6923a;
  --danger: #cc6f72;
  --accent-slate: #66758c;
  --accent-teal: #2d827b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.45;
  padding: 16px 20px;
}
.container { max-width: 1420px; margin: 0 auto; }
header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  gap: 10px;
}
h1 { font-size: 21px; font-weight: 650; color: #fafafa; letter-spacing: -.035em; }
.badge-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.badge-tech {
  background: rgba(255,255,255,.055);
  color: #d4d4d8;
  padding: 4px 10px;
  border-radius: 9999px;
  font-size: 11.5px;
  font-weight: 550;
  border: 1px solid rgba(255,255,255,.11);
}
.badge-ok { background: rgba(255,255,255,.10); color: #fafafa; border-color: rgba(255,255,255,.18); }
.badge-warn { background: rgba(255,255,255,.055); color: #d4d4d8; border-color: rgba(255,255,255,.11); }
.badge-bad { background: rgba(255,255,255,.045); color: #a1a1aa; border-color: rgba(255,255,255,.10); }
.grid { display: grid; grid-template-columns: 450px 1fr; gap: 18px; }
@media (max-width: 1080px) { .grid { grid-template-columns: 1fr; } }

.card {
  background: var(--surface);
  border-radius: 12px;
  border: 1px solid var(--border);
  padding: 15px;
  margin-bottom: 16px;
}
.card-title {
  font-size: 14px;
  font-weight: 700;
  color: #e2e8f0;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.samples-bar {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 5px;
  margin-bottom: 10px;
}
.btn-sample {
  background: #1e293b;
  color: #e2e8f0;
  border: 1px solid #334155;
  padding: 6px 8px;
  border-radius: 6px;
  font-size: 11.5px;
  font-weight: 600;
  cursor: pointer;
  text-align: left;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: all 0.12s;
}
.btn-sample:hover, .btn-sample.active {
  background: #f4f4f5;
  border-color: #f4f4f5;
  color: #09090b;
}
.drop-zone {
  border: 2px dashed #475569;
  border-radius: 8px;
  padding: 14px;
  text-align: center;
  background: rgba(15, 23, 42, 0.6);
  cursor: pointer;
  transition: border-color 0.2s;
}
.drop-zone:hover { border-color: rgba(255,255,255,.52); }
#preview-img {
  max-width: 100%;
  max-height: 210px;
  border-radius: 6px;
  margin-top: 8px;
  display: none;
  object-fit: contain;
  background: #000;
}
#camera-box {
  display: none;
  margin-top: 8px;
  text-align: center;
}
#camera-video {
  width: 100%;
  max-height: 220px;
  border-radius: 6px;
  background: #000;
}
.mode-selector {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 10px 0;
  font-size: 12.5px;
  background: rgba(15, 23, 42, 0.5);
  padding: 9px 11px;
  border-radius: 8px;
  border: 1px solid var(--border);
}
.mode-row {
  display: flex;
  gap: 14px;
  align-items: center;
  flex-wrap: wrap;
}
.mode-selector label {
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
}
.action-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
}
.btn-scan {
  background: #f4f4f5;
  color: #09090b;
  border: 1px solid #f4f4f5;
  padding: 11px 16px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  transition: filter 0.15s;
}
.btn-scan:hover { background: #fff; filter: none; transform: translateY(-1px); }
.btn-scan:disabled { opacity: 0.55; cursor: not-allowed; background: #71717a; color: #18181b; border-color: #71717a; }
.btn-batch {
  background: #f4f4f5;
  color: #09090b;
  border: 1px solid #f4f4f5;
  padding: 11px 14px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
}
.btn-batch:hover { background: #fff; filter: none; transform: translateY(-1px); }
.btn-batch:disabled { opacity: 0.55; cursor: not-allowed; background: #71717a; color: #18181b; border-color: #71717a; }

.status-pill {
  font-size: 12.5px;
  padding: 7px 11px;
  border-radius: 6px;
  margin-top: 10px;
  background: #0f172a;
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
}
@media (max-width: 900px) { .form-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 600px) { .form-grid { grid-template-columns: 1fr; } }
.field-box {
  background: rgba(15, 23, 42, 0.55);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 9px 11px;
}
.field-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.field-label {
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text-muted);
}
.status-badge {
  font-size: 10px;
  font-weight: 700;
  padding: 1.5px 6px;
  border-radius: 4px;
  text-transform: uppercase;
}
.badge-READ { background: rgba(255,255,255,.15); color: #fafafa; }
.badge-REVIEW { background: rgba(255,255,255,.085); color: #d4d4d8; }
.badge-UNREADABLE { background: rgba(255,255,255,.05); color: #a1a1aa; }

.field-input {
  width: 100%;
  background: #090d16;
  border: 1px solid #334155;
  border-radius: 6px;
  color: #fff;
  padding: 6px 9px;
  font-size: 13.5px;
  font-weight: 500;
}
.field-input:focus {
  outline: none;
  border-color: rgba(255,255,255,.52);
  box-shadow: 0 0 0 3px rgba(255,255,255,.06);
}
.candidate-chips {
  margin-top: 5px;
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
}
.chip {
  background: #1e293b;
  color: #cbd5e1;
  font-size: 10.5px;
  padding: 2px 7px;
  border-radius: 9999px;
  cursor: pointer;
  border: 1px solid #475569;
}
.chip:hover {
  background: #f4f4f5;
  color: #09090b;
  border-color: #f4f4f5;
}
.raw-box {
  background: #090d16;
  border-radius: 6px;
  padding: 10px;
  font-family: monospace;
  font-size: 11.5px;
  color: #d4d4d8;
  max-height: 155px;
  overflow: auto;
  white-space: pre-wrap;
}
.export-bar {
  display: flex;
  gap: 8px;
  margin-top: 12px;
  flex-wrap: wrap;
}
.btn-sec {
  background: #1e293b;
  color: #f8fafc;
  border: 1px solid var(--border);
  padding: 7px 12px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.btn-sec:hover { background: #334155; }

.history-table-wrap {
  overflow-x: auto;
  max-height: 290px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
}
table.history-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
table.history-table th, table.history-table td {
  padding: 6px 8px;
  border-bottom: 1px solid #1e293b;
  text-align: left;
  white-space: nowrap;
}
table.history-table th {
  background: #0f172a;
  color: #94a3b8;
  position: sticky;
  top: 0;
  font-weight: 700;
}
table.history-table tr:hover td {
  background: rgba(255, 255, 255, 0.045);
}

/* Restrained monochrome glass finish */
body {
  background:
    radial-gradient(ellipse at 15% -15%, rgba(255,255,255,.075), transparent 38rem),
    radial-gradient(ellipse at 100% 20%, rgba(255,255,255,.035), transparent 32rem),
    var(--bg);
  background-attachment: fixed;
}
header { border-bottom-color: rgba(255,255,255,.10); padding: 10px 2px 16px; }
h1 { color: #fafafa; font-size: 21px; font-weight: 650; letter-spacing: -.035em; }
header p { display: flex; align-items: center; flex-wrap: wrap; gap: 7px; }
.brand-line { color: #b1b9c7; }
.brand-separator { color: #586477; }
.badge-tech, .badge-ok, .badge-warn, .badge-bad {
  color: #d4d4d8;
  background: rgba(255,255,255,.055);
  border: 1px solid rgba(255,255,255,.11);
  font-weight: 550;
}
.badge-ok { color: #dce9e6; background: rgba(45,130,123,.18); border-color: rgba(75,167,156,.34); }
.badge-warn { color: #e7d7b9; background: rgba(198,146,58,.13); border-color: rgba(198,146,58,.3); }
.badge-bad { color: #e8c6c7; background: rgba(204,111,114,.13); border-color: rgba(204,111,114,.3); }
.card {
  background: linear-gradient(145deg, rgba(255,255,255,.075), rgba(255,255,255,.025));
  border-color: rgba(255,255,255,.10);
  border-radius: 16px;
  box-shadow: 0 18px 48px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.055);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}
.card-title { color: #f4f4f5; font-weight: 620; letter-spacing: -.01em; }
.btn-sample, .btn-sec, .chip {
  background: rgba(255,255,255,.045);
  color: #e4e4e7;
  border-color: rgba(255,255,255,.12);
  transition: background .16s ease, border-color .16s ease, color .16s ease, transform .16s ease;
}
.btn-sample:hover, .btn-sample.active, .chip:hover {
  color: #09090b;
  background: #f4f4f5;
  border-color: #f4f4f5;
}
.btn-sample:hover { transform: translateY(-1px); }
.btn-sec:hover { background: rgba(255,255,255,.11); border-color: rgba(255,255,255,.24); }
.drop-zone {
  background: rgba(0,0,0,.18);
  border-color: rgba(255,255,255,.20);
}
.drop-zone:hover { border-color: rgba(255,255,255,.52); }
.mode-selector, .field-box {
  background: rgba(0,0,0,.18);
  border-color: rgba(255,255,255,.09);
}
.btn-scan, .btn-batch {
  background: #f4f4f5;
  color: #09090b;
  border: 1px solid #f4f4f5;
  border-radius: 9px;
  font-weight: 650;
  transition: background .16s ease, color .16s ease, transform .16s ease;
}
.btn-scan:hover, .btn-batch:hover { background: #fff; filter: none; transform: translateY(-1px); }
.btn-scan:disabled, .btn-batch:disabled { background: #71717a; color: #18181b; border-color: #71717a; }
.status-pill, .raw-box, .field-input {
  background: rgba(0,0,0,.32);
  border-color: rgba(255,255,255,.10);
}
.field-input { color: #fafafa; }
.field-input:focus { border-color: rgba(255,255,255,.52); box-shadow: 0 0 0 3px rgba(255,255,255,.06); }
.status-badge, .badge-READ, .badge-REVIEW, .badge-UNREADABLE {
  color: #d4d4d8;
  background: rgba(255,255,255,.075);
}
.badge-READ { color: #dce9e6; background: rgba(45,130,123,.18); }
.badge-REVIEW { color: #e7d7b9; background: rgba(198,146,58,.13); }
.raw-box { color: #d4d4d8; }
.history-table-wrap { border-color: rgba(255,255,255,.10); }
table.history-table th { background: rgba(12,12,12,.94); color: #a1a1aa; }
table.history-table td { border-bottom-color: rgba(255,255,255,.055); }
#speed-badge, #timer-text { color: #e4e4e7 !important; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
}

/* Fill the available browser window instead of centering a small fixed canvas. */
html, body { min-height: 100%; }
body { min-height: 100vh; padding: 22px clamp(18px, 2.4vw, 40px); }
.container { width: 100%; max-width: none; min-height: calc(100vh - 44px); margin: 0 auto; }
header { min-height: 72px; margin-bottom: 20px; }
.grid {
  width: 100%;
  min-height: calc(100vh - 136px);
  grid-template-columns: minmax(360px, 0.72fr) minmax(0, 1.55fr);
  gap: 20px;
  align-items: stretch;
}
.grid > div { min-width: 0; display: flex; flex-direction: column; }
.grid > div > .card { width: 100%; }
.grid > div:first-child > .card:first-child { flex: 0 0 auto; }
.grid > div:last-child > .card:last-child { flex: 1 1 auto; min-height: 280px; }
.history-table-wrap { max-height: calc(100vh - 420px); min-height: 220px; }
.form-grid { gap: 12px; }
.field-box { padding: 11px 12px; }
.field-input { min-height: 38px; }
.btn-sample { min-height: 34px; padding: 7px 9px; }
.btn-scan, .btn-batch { min-height: 44px; }
@media (min-width: 1600px) {
  body { padding-inline: 48px; }
  .grid { grid-template-columns: minmax(420px, .75fr) minmax(0, 1.65fr); gap: 24px; }
  .card { padding: 20px; }
  .field-input { min-height: 42px; font-size: 14px; }
}
@media (max-width: 1080px) {
  body { padding: 16px; }
  .container { min-height: auto; }
  .grid { min-height: auto; grid-template-columns: 1fr; }
  .grid > div { display: block; }
  .history-table-wrap { max-height: 55vh; }
}
/* A small slate/teal/amber signal palette keeps statuses clear without visual noise. */
:root {
  --accent-slate: #71839a;
  --accent-teal: #4f9c91;
  --accent-amber: #c99b52;
}
.btn-scan { background: #426f69; border-color: #5a8981; color: #fff; }
.btn-scan:hover { background: #4d8078; color: #fff; }
.btn-batch { background: #394b60; border-color: #536981; color: #f4f6f8; }
.btn-batch:hover { background: #465d76; color: #fff; }  .badge-READ, .badge-ok { color: #c8e2dd; background: rgba(79,156,145,.18); border-color: rgba(79,156,145,.34); }
.badge-REVIEW, .badge-warn { color: #ead9b9; background: rgba(201,155,82,.14); border-color: rgba(201,155,82,.30); }
.badge-bad { color: #e7c5c6; background: rgba(173,94,99,.14); border-color: rgba(173,94,99,.28); }
#engine-badge.badge-ok { color: #c8e2dd; }
#speed-badge, #timer-text { color: #a9c7c2 !important; }
.brand-promise { display:flex; flex-wrap:wrap; gap:7px 16px; margin-top:5px; font-size:12px; color:#d8dde5; }
.brand-promise span + span { position:relative; }
.brand-promise span + span::before { content:""; position:absolute; left:-9px; top:50%; width:3px; height:3px; border-radius:50%; background:#638c86; }
.brand-meta { display:flex; flex-wrap:wrap; gap:5px 10px; margin-top:4px; font-size:11px; color:#95a0ae; }
.brand-meta span + span::before { content:"/"; color:#566273; margin-right:10px; }
.btn-scan { background:#356a64; border-color:#568b83; color:#fff; box-shadow:0 5px 18px rgba(53,106,100,.18); }
.btn-scan:hover { background:#407a73; }
.btn-batch { background:#303b4b; border-color:#536174; color:#f2f4f7; }
.btn-batch:hover { background:#3a485b; }
::selection { background: rgba(79,156,145,.40); color: #fff; }
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>Tag OCR</h1>
      <p class="brand-promise">
        <span>Local tag scanning</span>
        <span>Master-data matching</span>
        <span>Private by design</span>
      </p>
      <p class="brand-meta" aria-label="OCR model, runtime, and master-data details">
        <span id="model-details">GLM-OCR 0.9B vision model · glm-ocr:q8_0</span>
        <span>Ollama runtime · local processing</span>
        <span id="kb-summary">Tag service master knowledge base</span>
      </p>
    </div>
    <div class="badge-row">
      <span class="badge-tech badge-ok">Local OCR</span>
      <span class="badge-tech">Offline</span>
      <span class="badge-tech" id="session-counter">0 Scanned</span>
      <span class="badge-tech badge-warn" id="engine-badge" style="cursor: pointer;"
            onclick="retryWarmup()" title="Click to retry preload">Engine: starting…</span>
    </div>
  </header>

  <div class="grid">
    <!-- Left Column: Input and Controls -->
    <div>
      <div class="card">
        <div class="card-title">
          <span>1. Sample tags</span>
          <span style="font-size: 11px; color: var(--text-muted);">Select to scan</span>
        </div>
        <div class="samples-bar" id="sample-buttons">
          <button class="btn-sample" onclick="loadSample('sample_01_upright.jpg', this)">1. Upright demo</button>
          <button class="btn-sample" onclick="loadSample('sample_02_sideways.jpg', this)">2. Sideways demo</button>
          <button class="btn-sample" onclick="loadSample('sample_03_faded.jpg', this)">3. Faded demo</button>
        </div>

        <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <input type="file" id="file-input" accept="image/*" multiple style="display:none" onchange="onFilesSelected(this.files)">
          <div style="font-size: 13.5px; color: #e2e8f0; font-weight: 600;">Drop photos here, browse files, or paste an image</div>
          <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 3px;">Single image or multi-file batch</div>
        </div>
        <div style="display:flex; gap:8px; margin-top:8px;">
          <button class="btn-sec" style="flex:1" onclick="toggleCamera()">Open camera</button>
          <button class="btn-sec" id="snap-btn" style="display:none; flex:1;" onclick="captureCameraFrame()">Capture &amp; scan</button>
        </div>
        <div id="camera-box">
          <video id="camera-video" autoplay playsinline></video>
        </div>
        <img id="preview-img" alt="Tag preview">

        <div class="mode-selector">
          <div class="mode-row">
            <label><input type="checkbox" id="auto-scan" checked> <strong>Scan on select</strong></label>
            <label><input type="checkbox" id="dual-rot"> Check second orientation</label>
          </div>
          <div class="mode-row">
            <label><input type="radio" name="mode" value="glm-ocr:q8_0" checked> Standard · Q8</label>
            <label><input type="radio" name="mode" value="glm-ocr:bf16"> High precision · BF16</label>
          </div>
        </div>

        <div class="action-row">
          <button class="btn-scan" id="scan-btn" onclick="executeScan()">Scan tag</button>
          <button class="btn-batch" id="batch-btn" onclick="runAllSamplesBatch()">Scan all samples</button>
        </div>
        
        <div class="status-pill" id="status-box">
          <span id="status-text">Starting OCR engine...</span>
          <span id="timer-text" style="font-family: monospace; font-weight: 700; color: #e4e4e7;"></span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">OCR transcription</div>
        <div class="raw-box" id="raw-ocr-view">(Click any sample tag or drop an image to scan)</div>
      </div>
    </div>

    <!-- Right Column: Extracted Master Data Form + Live Session Table -->
    <div>
      <div class="card">
        <div class="card-title">
          <span>2. Tag details</span>
          <span id="speed-badge" style="font-size: 12.5px; color: #e4e4e7; font-weight: 650;"></span>
        </div>
        
        <div class="form-grid" id="form-container"></div>

        <div class="export-bar">
          <button class="btn-sec" onclick="saveCurrentToLog()">Save edited row</button>
          <button class="btn-sec" onclick="copyResultJSON()">Copy JSON</button>
          <button class="btn-sec" onclick="downloadResultJSON()">Download JSON</button>
          <button class="btn-sec" onclick="clearForm()">Reset form</button>
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>3. Session scans (<span id="queue-count">0</span>)</span>
          <div style="display:flex; gap:6px;">
            <button class="btn-sec" style="padding:4px 10px;" onclick="exportQueueCSV()">Export CSV</button>
            <button class="btn-sec" style="padding:4px 10px;" onclick="exportQueueJSON()">Export JSON</button>
            <button class="btn-sec" style="padding:4px 10px;" onclick="exportTrainingPack()" title="Download reviewed labels, OCR text, and source images for local training">Training pack</button>
            <button class="btn-sec" style="padding:4px 10px;" onclick="clearQueue()">Clear</button>
          </div>
        </div>
        <div class="history-table-wrap">
          <table class="history-table">
            <thead>
              <tr>
                <th>#</th>
                <th>File</th>
                <th>Time</th>
                <th>READ</th>
                <th>Branch</th>
                <th>BCCD Name</th>
                <th>Part Code</th>
                <th>Product Description</th>
                <th>Product Sr. No.</th>
                <th>DOP</th>
                <th>Complaint No.</th>
                <th>Defect</th>
                <th>Technician</th>
              </tr>
            </thead>
            <tbody id="queue-tbody">
              <tr><td colspan="13" style="text-align:center; color:var(--text-muted); padding:16px;">No scans yet. Choose a sample or add photos to begin.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const FIELDS = [
  { key: "Branch", label: "Branch (City)" },
  { key: "BCCD Name", label: "BCCD Name (Dealer)" },
  { key: "Spare Part Code", label: "Spare Part Code (6-digit)" },
  { key: "Product Description", label: "Product Description (Derived)" },
  { key: "Product Sr. No.", label: "Product Serial No." },
  { key: "Date of Purchase", label: "Date of Purchase" },
  { key: "Complaint No.", label: "Complaint Number" },
  { key: "Nature Of Defect", label: "Nature of Defect (Verbatim)" },
  { key: "Technician Name", label: "Technician / Engineer" }
];

const ALL_SAMPLES = [
  "sample_01_upright.jpg",
  "sample_02_sideways.jpg",
  "sample_03_faded.jpg"
];

let selectedBlob = null;
let lastResult = null;
let sessionQueue = [];
let cameraStream = null;
const trainingImages = new Map();

function initForm() {
  const container = document.getElementById('form-container');
  container.innerHTML = '';
  FIELDS.forEach(f => {
    const box = document.createElement('div');
    box.className = 'field-box';
    box.id = 'box_' + f.key;
    box.innerHTML = `
      <div class="field-header">
        <span class="field-label">${f.label}</span>
        <span class="status-badge badge-UNREADABLE" id="badge_${f.key}">EMPTY</span>
      </div>
      <input type="text" class="field-input" id="input_${f.key}">
      <div class="candidate-chips" id="chips_${f.key}"></div>
    `;
    container.appendChild(box);
  });
}
initForm();

let warmState = 'starting';
let scanInProgress = false;
let warmPollTimer = null;
let warmPollCount = 0;

function updateScanButton() {
  const busy = scanInProgress || warmState === 'warming';
  document.getElementById('scan-btn').disabled = busy;
  document.getElementById('batch-btn').disabled = busy;
}

function setEngineBadge(state, data) {
  const badge = document.getElementById('engine-badge');
  badge.className = 'badge-tech';
  if (state === 'ready') {
    badge.classList.add('badge-ok');
    badge.textContent = data && data.gpu_percent ? `Engine Ready · ${data.gpu_percent}% GPU` : 'Engine Ready';
  } else if (state === 'warming') {
    badge.classList.add('badge-warn');
    badge.textContent = 'Engine Warming…';
  } else if (state === 'partial') {
    badge.classList.add('badge-warn');
    badge.textContent = `Engine ${data && data.gpu_percent ? data.gpu_percent : '?'}% GPU`;
  } else if (state === 'error') {
    badge.classList.add('badge-bad');
    badge.textContent = 'Engine Error';
  } else {
    badge.classList.add('badge-warn');
    badge.textContent = 'Engine: starting…';
  }
}

function retryWarmup() {
  warmPollCount = 0;
  pollWarmup();
}

async function pollWarmup() {
  clearTimeout(warmPollTimer);
  try {
    const res = await fetch('/api/warmup');
    const d = await res.json();
    warmState = d.state || 'idle';
    setEngineBadge(warmState, d);
    updateScanButton();
    if (warmState === 'warming') {
      warmPollCount += 1;
      if (warmPollCount > 100) {
        warmState = 'timeout';
        setEngineBadge(warmState, d);
        updateScanButton();
        return;
      }
      document.getElementById('status-text').textContent = 'Preloading OCR model...';
      warmPollTimer = setTimeout(pollWarmup, 500);
    } else if (warmState === 'ready' && !scanInProgress) {
      document.getElementById('status-text').textContent =
        'Engine hot. Click any sample tag or drop photos for Scan & Go.';
    }
  } catch (e) {
    warmState = 'error';
    setEngineBadge('error', {});
    updateScanButton();
  }
}
pollWarmup();

async function loadModelDetails() {
  const updateModelName = () => {
    const selected = document.querySelector('input[name="mode"]:checked');
    const model = selected ? selected.value : 'glm-ocr:q8_0';
    document.getElementById('model-details').textContent = `GLM-OCR 0.9B vision model · ${model}`;
  };
  document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener('change', updateModelName));
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error(`Health endpoint returned ${response.status}`);
    const info = await response.json();
    if (info.model_family) {
      const selected = document.querySelector('input[name="mode"]:checked');
      document.getElementById('model-details').textContent = `${info.model_family} · ${selected ? selected.value : info.model}`;
    }
    const stats = info.kb_stats || {};
    const rows = Number(stats.rows_mined || 0).toLocaleString();
    const serials = Number(stats.indexed_serials || 0).toLocaleString();
    const complaints = Number(stats.indexed_complaints || 0).toLocaleString();
    document.getElementById('kb-summary').textContent =
      `${rows} source records · ${serials} serials · ${complaints} complaint IDs`;
  } catch (error) {
    document.getElementById('kb-summary').textContent = 'Master-data matching · local knowledge base';
  }
}
loadModelDetails();

// Drag & drop + Clipboard Paste support
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = 'rgba(255,255,255,.65)'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#475569'; });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.style.borderColor = '#475569';
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    onFilesSelected(e.dataTransfer.files);
  }
});

window.addEventListener('paste', e => {
  const items = (e.clipboardData || e.originalEvent.clipboardData).items;
  for (const item of items) {
    if (item.type.indexOf('image') === 0) {
      const file = item.getAsFile();
      onFilesSelected([file]);
      break;
    }
  }
});

const sampleCache = new Map();
let previewObjectUrl = null;
function setPreviewBlob(blob) {
  const p = document.getElementById('preview-img');
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = URL.createObjectURL(blob);
  p.src = previewObjectUrl;
  p.style.display = 'block';
}

async function onFilesSelected(files) {
  if (!files || files.length === 0) return;
  if (files.length === 1) {
    selectedBlob = files[0];
    setPreviewBlob(selectedBlob);
    document.getElementById('status-text').textContent = 'Loaded: ' + selectedBlob.name;
    if (document.getElementById('auto-scan').checked && warmState !== 'warming') {
      await executeScan();
    }
  } else {
    // Multi-file turbo batch!
    for (let i = 0; i < files.length; i++) {
      selectedBlob = files[i];
      setPreviewBlob(selectedBlob);
      await executeScan();
    }
  }
}

async function toggleCamera() {
  const box = document.getElementById('camera-box');
  const video = document.getElementById('camera-video');
  const snapBtn = document.getElementById('snap-btn');
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
    box.style.display = 'none';
    snapBtn.style.display = 'none';
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = cameraStream;
    box.style.display = 'block';
    snapBtn.style.display = 'inline-block';
  } catch (e) {
    alert('Camera access error: ' + e.message);
  }
}

async function captureCameraFrame() {
  if (!cameraStream) return;
  const video = document.getElementById('camera-video');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 960;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(async blob => {
    selectedBlob = new File([blob], `cam_${Date.now()}.jpg`, { type: 'image/jpeg' });
    setPreviewBlob(blob);
    await executeScan();
  }, 'image/jpeg', 0.90);
}

async function fetchSampleBlob(filename) {
  if (sampleCache.has(filename)) return sampleCache.get(filename);
  const res = await fetch('/samples/' + encodeURIComponent(filename));
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  sampleCache.set(filename, blob);
  return blob;
}

async function loadSample(filename, btnEl) {
  document.querySelectorAll('.btn-sample').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');
  document.getElementById('status-text').textContent = 'Loading ' + filename + '...';
  try {
    const blob = await fetchSampleBlob(filename);
    selectedBlob = new File([blob], filename, { type: 'image/jpeg' });
    setPreviewBlob(blob);
    if (document.getElementById('auto-scan').checked && warmState !== 'warming') {
      await executeScan();
    } else {
      document.getElementById('status-text').textContent = 'Loaded ' + filename + '. Click Scan.';
    }
  } catch(e) {
    document.getElementById('status-text').textContent = 'Error loading sample: ' + e;
  }
}

function formatOcrText(raw) {
  const NL = String.fromCharCode(10);
  const text = (raw || '').trim();
  if (!text) return '(no text returned)';
  if (!/[<>&]/.test(text)) return text;

  const normalized = text
    .replace(/<br[^>]*>/gi, NL)
    .replace(/<\/(p|div|li)>/gi, NL);
  const doc = new DOMParser().parseFromString(normalized, 'text/html');
  const rows = Array.from(doc.querySelectorAll('tr'));

  if (rows.length) {
    const lines = rows.map(row => {
      const cells = Array.from(row.querySelectorAll('th,td'))
        .map(cell => cell.textContent.replace(/\s+/g, ' ').trim())
        .filter(Boolean);
      return cells.join(' : ');
    }).filter(Boolean);
    if (lines.length) return lines.join(NL);
  }

  return (doc.body.textContent || '')
    .split(NL)
    .map(line => line.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .join(NL) || text;
}

let timerInterval = null;
async function executeScan() {
  if (!selectedBlob || scanInProgress) return;
  scanInProgress = true;
  updateScanButton();
  const t0 = Date.now();
  document.getElementById('status-text').textContent = `Scanning ${selectedBlob.name}...`;
  
  timerInterval = setInterval(() => {
    const s = ((Date.now() - t0) / 1000).toFixed(2);
    document.getElementById('timer-text').textContent = s + 's';
  }, 50);

  const fd = new FormData();
  fd.append('image', selectedBlob);
  const mode = document.querySelector('input[name=mode]:checked').value;
  fd.append('model', mode);
  fd.append('dual_rotation', document.getElementById('dual-rot').checked ? '1' : '0');

  try {
    const res = await fetch('/api/scan', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Scan failed');
    data.source_image = selectedBlob;

    lastResult = data;
    clearInterval(timerInterval);
    document.getElementById('timer-text').textContent = data.elapsed + 's';
    const t = data.timings || {};
    const passLabel = (data.passes || 1) === 1 ? '1-pass' : `${data.passes}-pass`;
    document.getElementById('status-text').textContent =
      `Scanned ${selectedBlob.name} in ${data.elapsed}s (${data.angle}°, ${passLabel} · READ ${data.read_count || 0}/${data.filled_count || 0})`;
    document.getElementById('speed-badge').textContent =
      `${data.elapsed}s · READ ${data.read_count || 0}/${data.filled_count || 0} fields`;
    
    document.getElementById('raw-ocr-view').textContent = 
      `Model: ${data.model || mode} | Angle: ${data.angle}° | Passes: ${data.passes || 1} | Scores: ${JSON.stringify(data.scores)} | `
      + `Vision: ${t.prefill_s || 0}s | Decode: ${t.decode_s || 0}s\\n\\n${formatOcrText(data.raw_ocr)}`;

    FIELDS.forEach(f => {
      const fieldData = (data.fields && data.fields[f.key]) || { value: data.result[f.key] || '', status: 'READ' };
      const input = document.getElementById('input_' + f.key);
      const badge = document.getElementById('badge_' + f.key);
      const chips = document.getElementById('chips_' + f.key);
      chips.innerHTML = '';

      input.value = fieldData.value || '';
      const st = fieldData.status || (input.value ? 'READ' : 'UNREADABLE');
      badge.textContent = st;
      badge.className = 'status-badge badge-' + st;

      if (fieldData.candidates && fieldData.candidates.length > 0) {
        fieldData.candidates.forEach(cand => {
          const val = cand.canonical || cand.code || cand;
          if (val === input.value && fieldData.candidates.length === 1) return;
          const chip = document.createElement('span');
          chip.className = 'chip';
          chip.textContent = `${val} (${cand.score || 'alt'})`;
          chip.onclick = () => {
            input.value = val;
            badge.textContent = 'READ';
            badge.className = 'status-badge badge-READ';
          };
          chips.appendChild(chip);
        });
      }
    });

    addQueueRow(selectedBlob.name, data);

  } catch(e) {
    clearInterval(timerInterval);
    document.getElementById('status-text').textContent = 'Error: ' + e.message;
  } finally {
    scanInProgress = false;
    updateScanButton();
  }
}

async function runAllSamplesBatch() {
  if (scanInProgress) return;
  const btns = document.querySelectorAll('.btn-sample');
  for (let i = 0; i < ALL_SAMPLES.length; i++) {
    const fname = ALL_SAMPLES[i];
    btns.forEach(b => b.classList.remove('active'));
    if (btns[i]) btns[i].classList.add('active');
    const blob = await fetchSampleBlob(fname);
    selectedBlob = new File([blob], fname, { type: 'image/jpeg' });
    setPreviewBlob(blob);
    await executeScan();
  }
}

function addQueueRow(filename, data) {
  const fields = getFormJSON();
  const raw = data.raw_mapped || {};
  const rowObj = {
    sample_id: `scan_${Date.now()}_${sessionQueue.length + 1}`,
    file: filename,
    elapsed: data.elapsed,
    model: data.model || document.querySelector('input[name=mode]:checked').value,
    angle: data.angle,
    passes: data.passes,
    read_count: data.read_count || 0,
    filled_count: data.filled_count || 0,
    raw_ocr: data.raw_ocr || '',
    raw_mapped: raw,
    fields,
    field_status: Object.fromEntries(FIELDS.map(f => [f.key, data.fields?.[f.key]?.status || 'REVIEW'])),
    label_source: 'ocr_unreviewed',
    ...fields
  };
  if (selectedBlob) {
    trainingImages.set(rowObj.sample_id, { filename, blob: selectedBlob });
  }
  sessionQueue.unshift(rowObj);
  renderQueue();
}

function saveCurrentToLog() {
  if (!selectedBlob || !lastResult) {
    alert('Scan an image before saving its reviewed labels.');
    return;
  }
  const current = sessionQueue.find(row => row.file === selectedBlob.name && row.label_source === 'ocr_unreviewed');
  if (current) {
    const fields = getFormJSON();
    Object.assign(current, fields, { fields, label_source: 'human_reviewed' });
    current.field_status = Object.fromEntries(FIELDS.map(field => [field.key, fields[field.key] ? 'VERIFIED' : 'EMPTY']));
    if (selectedBlob) trainingImages.set(current.sample_id, { filename: selectedBlob.name, blob: selectedBlob });
    renderQueue();
    document.getElementById('status-text').textContent = 'Reviewed labels saved to the session.';
    return;
  }
  addQueueRow(selectedBlob.name, lastResult);
  const saved = sessionQueue[0];
  const fields = getFormJSON();
  Object.assign(saved, fields, { fields, label_source: 'human_reviewed' });
  saved.field_status = Object.fromEntries(FIELDS.map(field => [field.key, fields[field.key] ? 'VERIFIED' : 'EMPTY']));
  renderQueue();
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

function renderQueue() {
  document.getElementById('queue-count').textContent = sessionQueue.length;
  document.getElementById('session-counter').textContent = `${sessionQueue.length} Scanned`;
  const tbody = document.getElementById('queue-tbody');
  if (sessionQueue.length === 0) {
    tbody.innerHTML = '<tr><td colspan="13" style="text-align:center; color:var(--text-muted); padding:16px;">No scans in session queue yet.</td></tr>';
    return;
  }
  tbody.innerHTML = sessionQueue.map((r, i) => `
    <tr>
      <td>${sessionQueue.length - i}</td>
      <td style="font-weight:600; color:#e4e4e7;">${escapeHtml(r.file)}</td>
      <td>${escapeHtml(r.elapsed)}s</td>
      <td><span class="status-badge badge-READ">${escapeHtml(r.read_count)}/${escapeHtml(r.filled_count)}</span></td>
      <td>${escapeHtml(r["Branch"])}</td>
      <td>${escapeHtml(r["BCCD Name"])}</td>
      <td style="font-family:monospace; font-weight:700;">${escapeHtml(r["Spare Part Code"])}</td>
      <td>${escapeHtml(r["Product Description"])}</td>
      <td style="font-family:monospace;">${escapeHtml(r["Product Sr. No."])}</td>
      <td>${escapeHtml(r["Date of Purchase"])}</td>
      <td style="font-family:monospace;">${escapeHtml(r["Complaint No."])}</td>
      <td>${escapeHtml(r["Nature Of Defect"])}</td>
      <td>${escapeHtml(r["Technician Name"])}</td>
    </tr>
  `).join('');
}

function clearQueue() {
  sessionQueue = [];
  trainingImages.clear();
  renderQueue();
}

function exportQueueCSV() {
  if (sessionQueue.length === 0) return alert('No scans in queue yet.');
  const cols = ["sample_id", "file", "elapsed", "model", "angle", "passes", "read_count", "filled_count", "Branch", "BCCD Name", "Spare Part Code", "Product Description", "Product Sr. No.", "Date of Purchase", "Complaint No.", "Nature Of Defect", "Technician Name"];
  const lines = [cols.join(",")];
  sessionQueue.slice().reverse().forEach(r => {
    const vals = cols.map(c => {
      let value = String(r[c] ?? '');
      if (['=', '+', '@', '-'].some(prefix => value.trimStart().startsWith(prefix))) value = "'" + value;
      return `"${value.replace(/"/g, '""')}"`;
    });
    lines.push(vals.join(","));
  });
  const blob = new Blob([String.fromCharCode(0xFEFF), lines.join(String.fromCharCode(13, 10))], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `tag_tagocr_${Date.now()}.csv`;
  a.click();
}

function exportQueueJSON() {
  if (sessionQueue.length === 0) return alert('No scans in queue yet.');
  const blob = new Blob([JSON.stringify(sessionQueue.slice().reverse(), null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `tagocr_batch_${Date.now()}.json`;
  a.click();
}

async function exportTrainingPack() {
  if (sessionQueue.length === 0) return alert('Scan at least one tag first.');
  const button = [...document.querySelectorAll('button')].find(item => item.textContent.trim() === 'Training pack');
  if (button) { button.disabled = true; button.textContent = 'Preparing…'; }
  try {
    const form = new FormData();
    form.append('rows', JSON.stringify(sessionQueue.slice().reverse()));
    for (const row of sessionQueue) {
      const image = trainingImages.get(row.sample_id);
      if (image?.blob) form.append('image_' + row.sample_id, image.blob, image.filename);
    }
    const response = await fetch('/api/export/training-pack', { method: 'POST', body: form });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      alert(error.error || 'Could not build the training pack.');
      return;
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tag_tagocr_training_${Date.now()}.zip`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    alert(`Could not build the training pack: ${error.message}`);
  } finally {
    if (button) { button.disabled = false; button.textContent = 'Training pack'; }
  }
}

function clearForm() {
  FIELDS.forEach(f => {
    document.getElementById('input_' + f.key).value = '';
    const badge = document.getElementById('badge_' + f.key);
    badge.textContent = 'EMPTY';
    badge.className = 'status-badge badge-UNREADABLE';
    document.getElementById('chips_' + f.key).innerHTML = '';
  });
  document.getElementById('raw-ocr-view').textContent = '(Reset)';
}

function getFormJSON() {
  const out = {};
  FIELDS.forEach(f => {
    out[f.key] = document.getElementById('input_' + f.key).value;
  });
  return out;
}

function copyResultJSON() {
  navigator.clipboard.writeText(JSON.stringify(getFormJSON(), null, 2));
}

function downloadResultJSON() {
  const blob = new Blob([JSON.stringify(getFormJSON(), null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'tag_extracted.json';
  a.click();
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/samples/<filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLE_DIR, filename)

@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "engine": "glm-ocr",
        "model": config.DEFAULT_MODEL,
        "model_family": "GLM-OCR 0.9B vision OCR via Ollama",
        "runtime": "Ollama · local inference",
        "toolkit": "available",
        "warmup": pipeline.warmup_status(config.DEFAULT_MODEL),
        "kb_stats": pipeline.resolver.kb.get("stats", {}),
    })

@app.route("/api/warmup")
def warmup():
    model = request.args.get("model") or config.DEFAULT_MODEL
    return jsonify(pipeline.start_warmup(model))

@app.route("/api/export/training-pack", methods=["POST"])
def export_training_pack():
    """Download reviewed scan labels with source images and OCR for local model training."""
    if request.content_length is not None and request.content_length > MAX_TRAINING_PACK_UPLOAD_BYTES:
        return jsonify({"error": "Training pack upload is limited to 100 MB total"}), 413
    if request.mimetype != "multipart/form-data":
        return jsonify({"error": "Training pack export requires multipart form data with source images"}), 400
    try:
        rows = json.loads(request.form.get("rows", "[]"))
    except (TypeError, ValueError):
        return jsonify({"error": "Training rows must be valid JSON"}), 400
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "No scanned rows provided"}), 400
    if len(rows) > MAX_TRAINING_PACK_ROWS:
        return jsonify({"error": f"Export is limited to {MAX_TRAINING_PACK_ROWS} images at a time"}), 413
    if len(rows) > len(request.files):
        return jsonify({"error": "Attach a source image for every example before exporting the training pack"}), 400

    # The client sends reviewed labels and original image bytes as multipart fields.
    manifest_rows = []
    seen_ids = set()
    kb_stats = pipeline.resolver.kb.get("stats", {})
    runtime = "Ollama · local inference"
    model_family = "GLM-OCR 0.9B vision OCR via Ollama"
    for row in rows:
        if not isinstance(row, dict):
            return jsonify({"error": "Each training example must be a scan-row object"}), 400
        filename = str(row.get("file") or "tag.jpg").replace(chr(92), "/").rsplit("/", 1)[-1] or "tag.jpg"
        sample_id = str(row.get("sample_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", sample_id) or sample_id in seen_ids:
            return jsonify({"error": "Training rows need unique, safe sample IDs"}), 400
        seen_ids.add(sample_id)
        labels = row.get("fields") if isinstance(row.get("fields"), dict) else {
            key: row.get(key, "") for key in pipeline.KEYS
        }
        field_status = row.get("field_status", {})
        raw_mapped = row.get("raw_mapped", {})
        if not isinstance(field_status, dict) or not isinstance(raw_mapped, dict):
            return jsonify({"error": "Field status and raw mapped OCR values must be objects"}), 400
        label_source = str(row.get("label_source") or "ocr_unreviewed")
        if label_source not in {"human_reviewed", "ocr_unreviewed"}:
            return jsonify({"error": "Unknown label source in training row"}), 400
        manifest_rows.append({
            "sample_id": sample_id,
            "file": filename,
            "model": str(row.get("model") or config.DEFAULT_MODEL),
            "runtime": runtime,
            "knowledge_base": {
                "rows_mined": kb_stats.get("rows_mined", 0),
                "indexed_serials": kb_stats.get("indexed_serials", 0),
                "indexed_complaints": kb_stats.get("indexed_complaints", 0),
            },
            "elapsed": row.get("elapsed", ""),
            "read_count": row.get("read_count", 0),
            "filled_count": row.get("filled_count", 0),
            "angle": row.get("angle", ""),
            "passes": row.get("passes", ""),
            "labels": {key: str(labels.get(key) or "") for key in pipeline.KEYS},
            "field_status": field_status,
            "label_source": label_source,
            "raw_ocr": str(row.get("raw_ocr") or ""),
            "raw_mapped": raw_mapped,
        })

    if len(manifest_rows) != len(rows):
        return jsonify({"error": "Every submitted row must be a valid scan example"}), 400

    # Validate and buffer all image bytes before producing any archive. Extensions
    # come from file signatures, not user-controlled filenames.
    image_payloads = {}
    image_types = {
        bytes.fromhex("89504e470d0a1a0a"): ".png",
        b"BM": ".bmp",
    }
    total_image_bytes = 0
    for row in manifest_rows:
        image = request.files.get("image_" + row["sample_id"])
        if image is None or not image.filename:
            return jsonify({"error": f"Missing source image for {row['sample_id']}"}), 400
        image_bytes = image.read(MAX_TRAINING_IMAGE_BYTES + 1)
        if not image_bytes:
            return jsonify({"error": f"Source image for {row['sample_id']} is empty"}), 400
        if len(image_bytes) > MAX_TRAINING_IMAGE_BYTES:
            return jsonify({"error": "Each training image must be 20 MB or smaller"}), 413
        total_image_bytes += len(image_bytes)
        if total_image_bytes > MAX_TRAINING_PACK_UPLOAD_BYTES:
            return jsonify({"error": "Training pack images exceed the 100 MB total limit"}), 413
        extension = next((ext for magic, ext in image_types.items() if image_bytes.startswith(magic)), None)
        if image_bytes.startswith(bytes.fromhex("ffd8ff")):
            extension = ".jpg"
        elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            extension = ".webp"
        if extension is None:
            return jsonify({"error": f"Source image for {row['sample_id']} is not a supported JPEG, PNG, WebP, or BMP image"}), 400
        row["image_file"] = f"images/{row['sample_id']}{extension}"
        image_payloads[row["sample_id"]] = image_bytes

    manifest = io.StringIO(newline="")
    columns = ["sample_id", "source_file", "image_file", "ocr_model", "runtime", "knowledge_base_rows", "indexed_serials", "indexed_complaints", "elapsed_seconds", "read_count", "filled_count", "angle", "passes", "label_source", *pipeline.KEYS, "field_status_json", "raw_ocr", "raw_mapped_json"]
    writer = csv.DictWriter(manifest, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    def excel_safe(value):
        text = str(value if value is not None else "")
        if text.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + text
        return text

    for row in manifest_rows:
        csv_row = {
            "sample_id": row["sample_id"],
            "source_file": row["file"],
            "image_file": row["image_file"],
            "ocr_model": row["model"],
            "runtime": row["runtime"],
            "knowledge_base_rows": row["knowledge_base"]["rows_mined"],
            "indexed_serials": row["knowledge_base"]["indexed_serials"],
            "indexed_complaints": row["knowledge_base"]["indexed_complaints"],
            "elapsed_seconds": row["elapsed"],
            "read_count": row["read_count"],
            "filled_count": row["filled_count"],
            "angle": row["angle"],
            "passes": row["passes"],
            "label_source": row["label_source"],
            **row["labels"],
            "field_status_json": json.dumps(row["field_status"], ensure_ascii=False),
            "raw_ocr": row["raw_ocr"],
            "raw_mapped_json": json.dumps(row["raw_mapped"], ensure_ascii=False),
        }
        writer.writerow({key: excel_safe(value) for key, value in csv_row.items()})

    jsonl_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows)
    timestamp = datetime.now(timezone.utc).isoformat()
    reviewed_count = sum(row["label_source"] == "human_reviewed" for row in manifest_rows)
    metadata = {
        "format": "tag-tagocr-training-pack",
        "schema_version": 1,
        "exported_utc": timestamp,
        "example_count": len(manifest_rows),
        "human_reviewed_count": reviewed_count,
        "ocr_unreviewed_count": len(manifest_rows) - reviewed_count,
        "model_family": model_family,
        "runtime": runtime,
        "knowledge_base": kb_stats,
        "fields": pipeline.KEYS,
        "files": ["README.txt", "metadata.json", "manifest.csv", "examples.jsonl", "images/"],
    }
    readme = (
        "Tag OCR training-data export\n"
        "========================================\n\n"
        "A portable dataset bundle for OCR evaluation, annotation, dataset curation, and compatible training pipelines.\n"
        "This is data, not a ready-to-train model. Check the target model's format and training requirements.\n\n"
        "Review every label against its source image first. human_reviewed means an operator saved edited labels;\n"
        "ocr_unreviewed means machine output has not been confirmed and must not be used as verified ground truth.\n"
        "Field-level status, raw OCR text, mapped OCR, model, runtime, and knowledge-base snapshot are in examples.jsonl.\n\n"
        "manifest.csv is UTF-8 with BOM for Excel and points to the corresponding images/ path. Formula-like label text is\n"
        "apostrophe-prefixed in CSV for spreadsheet safety; examples.jsonl retains original label text. metadata.json documents\n"
        "the schema and provenance. Source image names inside the archive are generated sample IDs, not uploaded filenames.\n\n"
        "Suggested process: verify labels against images; use reviewed rows for ground truth; split by source tag/entity to\n"
        "reduce leakage; retain evaluation examples; and obtain permission before sharing customer or personal information.\n\n"
        f"Exported UTC: {timestamp}\n"
        f"Examples: {len(manifest_rows)} ({reviewed_count} reviewed, {len(manifest_rows) - reviewed_count} OCR-unreviewed)\n"
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", readme)
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
        archive.writestr("manifest.csv", manifest.getvalue().encode("utf-8-sig"))
        archive.writestr("examples.jsonl", jsonl_text.encode("utf-8"))
        for row in manifest_rows:
            archive.writestr(row["image_file"], image_payloads[row["sample_id"]])
    zip_buffer.seek(0)
    return Response(
        zip_buffer.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": 'attachment; filename="tag_tagocr_training_pack.zip"'},
    )

@app.route("/api/scan", methods=["POST"])
def scan():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image uploaded"}), 400
    
    model = request.form.get("model") or config.DEFAULT_MODEL
    dual = request.form.get("dual_rotation", "0") == "1"
    ts = int(time.time() * 1000)
    safe_name = re.sub(r"[^A-Za-z0-9._\- ()]", "_", os.path.basename(f.filename or "tag.jpg")) or "tag.jpg"
    dest = UPLOAD_DIR / f"upload_{ts}_{safe_name}"
    f.save(str(dest))

    try:
        res = pipeline.scan_and_resolve(dest, model=model, dual_rotation=dual)
        t = res.get("timings", {})
        print(
            f"[scan] {f.filename} model={model} dual={int(dual)} "
            f"elapsed={res['elapsed']}s passes={res['passes']} "
            f"READ={res.get('read_count')}/{res.get('filled_count')} "
            f"load={t.get('load_s', 0)}s vision={t.get('prefill_s', 0)}s "
            f"decode={t.get('decode_s', 0)}s",
            flush=True,
        )
        return jsonify(res)
    except Exception as e:
        print(f"[scan] {f.filename} FAILED: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

def _preload_worker() -> None:
    try:
        state = pipeline.start_warmup(config.DEFAULT_MODEL, blocking=True)
        print(
            f"[warmup] {state['state']}: {state['detail']} ({state['seconds']}s)",
            flush=True,
        )
    except Exception as exc:
        print(f"[warmup] failed: {exc}", flush=True)


if __name__ == "__main__":
    print(f"Model   : {config.DEFAULT_MODEL}", flush=True)
    print(f"Starting Tag OCR at http://127.0.0.1:{config.PORT} ...", flush=True)
    print("Preloading OCR model in the background...", flush=True)
    threading.Thread(target=_preload_worker, daemon=True).start()
    app.run(host="127.0.0.1", port=config.PORT, threaded=True)
