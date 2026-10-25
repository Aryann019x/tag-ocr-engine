"""Central configuration for the Tag OCR app.

Everything deployment-specific lives in `ocr_config.json` next to this file, so
model settings can change per machine without editing code.  Environment
variables override the file.

Resolution order (highest priority first):
    1. environment variables: TAGOCR_PORT, TAGOCR_MODEL, TAGOCR_KEEP_ALIVE,
       TAGOCR_NUM_CTX, TAGOCR_GPU_LAYERS
    2. ocr_config.json
    3. built-in defaults below
"""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "ocr_config.json"

DEFAULTS = {
    # Ollama model used when the request does not specify one.
    "default_model": "glm-ocr:q8_0",
    # Port for the local web UI.
    "port": 5005,
    # Keep the model resident between scans ("24h", "30m", or -1 for forever).
    "keep_alive": "24h",
    # Context window. 2048 is optimal for one 896x560 form + ~140 output tokens
    # and halves KV-cache VRAM overhead on 4GB GPUs.
    "num_ctx": 2048,
    # Layers offloaded to the GPU. 99 forces full-GPU loading, 0 disables the
    # GPU (CPU-only), and -1 lets Ollama decide automatically.
    "gpu_layers": 99,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)

    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Could not read %s: %s" % (CONFIG_PATH, exc)) from exc
        if not isinstance(data, dict):
            raise RuntimeError("%s must contain a JSON object" % CONFIG_PATH)
        for key in DEFAULTS:
            if key in data and data[key] not in (None, ""):
                cfg[key] = data[key]

    env_port = os.environ.get("TAGOCR_PORT", "").strip()
    if env_port.isdigit():
        cfg["port"] = int(env_port)

    env_model = os.environ.get("TAGOCR_MODEL", "").strip()
    if env_model:
        cfg["default_model"] = env_model

    env_keep_alive = os.environ.get("TAGOCR_KEEP_ALIVE", "").strip()
    if env_keep_alive:
        cfg["keep_alive"] = env_keep_alive

    env_num_ctx = os.environ.get("TAGOCR_NUM_CTX", "").strip()
    if env_num_ctx.isdigit() and int(env_num_ctx) >= 512:
        cfg["num_ctx"] = int(env_num_ctx)

    env_gpu_layers = os.environ.get("TAGOCR_GPU_LAYERS", "").strip()
    try:
        if env_gpu_layers:
            cfg["gpu_layers"] = int(env_gpu_layers)
    except ValueError:
        pass

    return cfg


CONFIG = load_config()
PORT = int(CONFIG["port"])
DEFAULT_MODEL = str(CONFIG["default_model"])
KEEP_ALIVE = str(CONFIG["keep_alive"])

try:
    NUM_CTX = int(CONFIG["num_ctx"])
except (TypeError, ValueError):
    NUM_CTX = DEFAULTS["num_ctx"]

try:
    GPU_LAYERS = int(CONFIG["gpu_layers"])
except (TypeError, ValueError):
    GPU_LAYERS = DEFAULTS["gpu_layers"]
