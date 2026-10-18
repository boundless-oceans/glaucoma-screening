"""Preview training/validation preprocessing without importing PyTorch or changing data."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from . import CONFIG_DIR, DEFAULT_DATA_ROOT, PROJECT_ROOT
from .data_utils import json_save, read_rows, subset
from .model_config import load_config
from .preprocessing import prepare_image
from .run_utils import protect_output, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "train.json")
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 32 or not 128 <= args.tile_size <= 512:
        parser.error("count must be 1..32; tile-size must be 128..512")
    config = load_config(args.config)
    validate_config(config)
    rows = read_rows(args.data_root, args.split)
    rows = rows[:1] if args.count == 1 else subset(rows, args.count, config["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = protect_output(args.data_root, args.output_dir or PROJECT_ROOT / "outputs" / f"preprocess_preview_{stamp}", directory=True)
    output.mkdir(parents=True, exist_ok=False)
    tile = args.tile_size
    pair_width, pair_height = tile * 2 + 24, tile + 52
    columns = min(2, len(rows))
    canvas = Image.new("RGB", (columns * pair_width, math.ceil(len(rows) / columns) * pair_height), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    records = []
    for index, row in enumerate(rows):
        with Image.open(row["path"]) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
            processed, info = prepare_image(source, config["image_size"], config)
        before = Image.new("RGB", (tile, tile), (0, 0, 0))
        contained = ImageOps.contain(original, (tile, tile), Image.Resampling.BILINEAR)
        offset_x, offset_y = (tile - contained.width) // 2, (tile - contained.height) // 2
        before.paste(contained, (offset_x, offset_y))
        if info["cropped"]:
            left, top, right, bottom = info["crop_box"]
            scale_x, scale_y = contained.width / original.width, contained.height / original.height
            ImageDraw.Draw(before).rectangle((offset_x + left * scale_x, offset_y + top * scale_y,
                                             offset_x + right * scale_x - 1, offset_y + bottom * scale_y - 1),
                                            outline=(0, 255, 80), width=2)
        after = processed.resize((tile, tile), Image.Resampling.BILINEAR)
        x, y = (index % columns) * pair_width, (index // columns) * pair_height
        draw.text((x + 4, y + 2), f"{row['Image_name']} label={row['True_Label']}", fill=(0, 0, 0))
        draw.text((x + 4, y + 17), "before / green=crop box", fill=(0, 0, 0))
        draw.text((x + tile + 16, y + 17), f"after: {info['reason']}", fill=(0, 0, 0))
        canvas.paste(before, (x + 4, y + 36))
        canvas.paste(after, (x + tile + 16, y + 36))
        records.append(dict(image_name=row["Image_name"], label=int(row["True_Label"]), **info))
        print(f"{index + 1}/{len(rows)} {row['Image_name']}: {info['reason']}, crop_box={info['crop_box']}", flush=True)
    canvas.save(output / "preview.jpg", quality=92)
    json_save(output / "report.json", dict(split=args.split, sampled=True, config=config,
              counts=dict(Counter(record["reason"] for record in records)), images=records))
    print(f"Preview: {output / 'preview.jpg'}\nDetails: {output / 'report.json'}", flush=True)
    print("No source files changed; this preview does not establish a performance improvement.", flush=True)


if __name__ == "__main__":
    main()
