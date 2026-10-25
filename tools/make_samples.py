"""Draw sample tag photos for the demo.

Produces three JPEGs in ``samples/``:
  sample_01_upright.jpg   - clean, upright
  sample_02_sideways.jpg  - rotated 90 degrees (tests orientation scoring)
  sample_03_faded.jpg     - low-contrast handwriting (tests fallback behaviour)

Requires Pillow.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)

W, H = 1400, 880
LABEL_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
SCRIPT_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoesc.ttf",
    r"C:\Windows\Fonts\seguisbi.ttf",
    r"C:\Windows\Fonts\ariali.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]

FIELDS = [
    ("Branch", "Testnagar"),
    ("BCCD Name", "Demo Electronics"),
    ("Product Description", ""),
    ("Product Sr.No.", "N304123456789"),
    ("Date Of Purchase", "12/03/24"),
    ("Complaint no.", "TST12057605796344-1"),
    ("Spare Parts Code", "100001"),
    ("Nature Of Defect", "Dead"),
    ("technician Name", "Alex Demo"),
]


def first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).is_file():
            return p
    return None


def draw_tag(handwriting_color=(25, 25, 60), jitter: int = 3, seed: int = 7) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("RGB", (W, H), (250, 248, 242))
    d = ImageDraw.Draw(img)

    # card border + header
    d.rectangle([18, 18, W - 18, H - 18], outline=(40, 40, 40), width=3)
    d.rectangle([18, 18, W - 18, 90], fill=(235, 232, 224), outline=(40, 40, 40), width=2)

    label_font_path = first_existing(LABEL_FONT_CANDIDATES)
    script_font_path = first_existing(SCRIPT_FONT_CANDIDATES)
    label_font = ImageFont.truetype(label_font_path, 25) if label_font_path else ImageFont.load_default()
    title_font = ImageFont.truetype(label_font_path, 30) if label_font_path else label_font
    script_font = ImageFont.truetype(script_font_path, 34) if script_font_path else label_font

    d.text((44, 38), "DEFECTIVE SPARE PART RETURN TAG", font=title_font, fill=(30, 30, 30))

    top = 120
    row_h = (H - top - 40) // len(FIELDS)
    label_w = 430
    d.line([label_w, top, label_w, top + row_h * len(FIELDS)], fill=(90, 90, 90), width=2)

    for i, (label, value) in enumerate(FIELDS):
        y = top + i * row_h
        if i:
            d.line([24, y, W - 24, y], fill=(120, 120, 120), width=1)
        d.text((40, y + row_h // 2 - 14), label, font=label_font, fill=(50, 50, 50))
        if not value:
            continue
        # slight per-line rotation/offset so it looks handwritten rather than typed
        text_img = Image.new("RGBA", (900, 80), (0, 0, 0, 0))
        td = ImageDraw.Draw(text_img)
        td.text((10, 0), value, font=script_font, fill=handwriting_color + (255,))
        if jitter:
            text_img = text_img.rotate(rnd.uniform(-jitter, jitter), expand=True, resample=Image.BICUBIC)
        img.paste(text_img, (label_w + 35 + rnd.randint(-8, 8), y + row_h // 2 - 42), text_img)

    return img


def main() -> None:
    clean = draw_tag()
    clean.save(SAMPLES / "sample_01_upright.jpg", quality=92)
    print("wrote samples/sample_01_upright.jpg")

    sideways = clean.rotate(-90, expand=True)
    sideways.save(SAMPLES / "sample_02_sideways.jpg", quality=92)
    print("wrote samples/sample_02_sideways.jpg")

    faded = draw_tag(handwriting_color=(150, 150, 160), jitter=5, seed=11)
    faded = faded.filter(ImageFilter.GaussianBlur(radius=1.2))
    faded.save(SAMPLES / "sample_03_faded.jpg", quality=85)
    print("wrote samples/sample_03_faded.jpg")


if __name__ == "__main__":
    main()
