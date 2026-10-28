"""SERVER diagnostic: frozen-model Grad-CAM candidate ROI previews, train/valid only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

from . import CONFIG_DIR, DEFAULT_DATA_ROOT, PROJECT_ROOT
from .data_utils import age_value, data_signature, json_save, read_rows
from .model_config import canonical_config, checkpoint_model_name
from .preprocessing import prepare_image
from .roi_geometry import roi_options, preview_rows, select_roi, preview_strip
from .run_utils import protect_output, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--roi-config", type=Path, default=CONFIG_DIR / "local_roi.json")
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--include-name", action="append", default=[])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--save-crops", action="store_true", help="Save accepted 384px crops for inspection only")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 2 <= args.count <= 64 or args.seed < 0 or not 128 <= args.tile_size <= 384:
        parser.error("count must be 2..64; seed nonnegative; tile-size 128..384")
    policy = roi_options(json.loads(args.roi_config.read_text(encoding="utf-8-sig")))
    # Import model framework only after argument validation; local --help stays lightweight.
    import torch
    from torchvision.transforms.functional import normalize, to_tensor
    from .common import FusionModel
    from .gradcam import GradCam
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Run on a GPU server, or explicitly use --device cpu on the server.")
    raw = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(raw).hexdigest()
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    del raw
    config = canonical_config(state["config"])
    validate_config(config)
    if config["input_view"] != "global" or not config["use_age"]:
        parser.error("This Grad-CAM preview requires the global image+age reference checkpoint")
    if any(config.get(k, 0) for k in ("train_limit", "valid_limit", "test_limit")):
        parser.error("Use a full-data reference checkpoint, not a debug model")
    signature = data_signature(args.data_root)
    if signature != state["data_signature"]:
        parser.error("CSV files differ from source checkpoint")
    rows = preview_rows(read_rows(args.data_root, args.split), args.count, args.seed,
                        args.sampling == "balanced", args.include_name)
    age_stats = state["age_stats"]
    if not all(math.isfinite(float(age_stats[k])) for k in ("median", "mean", "std")) or age_stats["std"] <= 0:
        raise ValueError("Invalid saved age statistics")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model_name = checkpoint_model_name(state)
    model = FusionModel(model_name=model_name).to(device)
    model.load_state_dict(state["model"], strict=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = protect_output(args.data_root, args.output_dir or PROJECT_ROOT / "outputs" / ("local_roi_" + stamp), directory=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "heatmaps").mkdir()
    if args.save_crops:
        (output / "crops").mkdir()
    report = dict(status="running", start_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                  source_epoch=state["epoch"], source_validation=state.get("best_metrics"),
                  model_name=model_name, model_config=config, roi_policy=policy, split=args.split,
                  sampling=args.sampling, seed=args.seed, data_signature=signature,
                  selection_sha256=hashlib.sha256(json.dumps([r["Image_name"] for r in rows]).encode()).hexdigest(),
                  target="positive_class_logit_for_all_images", arithmetic="float32_no_autocast",
                  device=str(device), torch=str(torch.__version__), pillow=pillow_version,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  warning="Candidate attention regions, not verified optic-disc annotations. No training, selection or test prediction.",
                  images=[], pages=[])
    json_save(output / "report.json", report)
    page = None
    try:
        with GradCam(model) as cam:
            for index, row in enumerate(rows):
                if device.type == "cuda": torch.cuda.synchronize()
                started = time.perf_counter()
                with Image.open(row["path"]) as source:
                    processed, info = prepare_image(source, config["image_size"], config)
                tensor = normalize(to_tensor(processed), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(device)
                age = age_value(row["Age"])
                missing = age is None
                age = age_stats["median"] if missing else age
                age_features = [(age - age_stats["mean"]) / age_stats["std"], float(missing)]
                heat, probability = cam(tensor, torch.tensor([age_features], dtype=torch.float32, device=device))
                if not math.isfinite(probability):
                    raise ValueError("Nonfinite diagnostic probability")
                if device.type == "cuda": torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                selection = select_roi(heat, processed, info, policy)
                # Reopen full original for display/cropping; never crop the 320px input as a 'high resolution' ROI.
                with Image.open(row["path"]) as source:
                    original = ImageOps.exif_transpose(source).convert("RGB")
                if list(original.size) != info["original_size"]:
                    raise ValueError("Original geometry differs from preprocessing metadata")
                name = row["Image_name"]
                heat_path = "heatmaps/" + name + ".npz"
                np.savez_compressed(output / heat_path, positive_logit_gradcam=heat)
                record = dict(image_name=name, label=int(row["True_Label"]), age_missing=missing,
                              diagnostic_probability_fp32=probability, preprocess_and_cam_seconds=seconds,
                              preprocessing=info, selection=selection, heatmap_file=heat_path)
                if args.save_crops and not selection["fallback"]:
                    crop_path = "crops/" + name
                    ImageOps.pad(original.crop(selection["roi_box"]), (384, 384), method=Image.Resampling.BILINEAR,
                                 color="black").save(output / crop_path, quality=95)
                    record["inspection_crop_file"] = crop_path
                report["images"].append(record)
                if index % 4 == 0:
                    count = min(4, len(rows) - index)
                    page = Image.new("RGB", (args.tile_size * 3, (args.tile_size + 42) * count), "white")
                title = f"{name} label={row['True_Label']} FP32 P={probability:.4f} {selection['reason']}"
                strip = preview_strip(original, processed, heat, selection, title, args.tile_size)
                page.paste(strip, (0, (index % 4) * strip.height))
                page_name = f"preview_{index // 4 + 1:02d}.jpg"
                page.save(output / page_name, quality=95)
                if page_name not in report["pages"]: report["pages"].append(page_name)
                report["reasons"] = dict(Counter(r["selection"]["reason"] for r in report["images"]))
                json_save(output / "report.json", report)
                print(f"{index + 1}/{len(rows)} {name}: {selection['reason']}; ROI={selection['roi_box']}", flush=True)
        report.update(status="completed", end_utc=datetime.now(timezone.utc).isoformat(),
                      candidates=sum(not r["selection"]["fallback"] for r in report["images"]))
        json_save(output / "report.json", report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        json_save(output / "report.json", report)
        raise
    print(f"Report and previews: {output}")
    print("No optimizer, model update, source-image modification or submission generation. Not T4 timing acceptance.")


if __name__ == "__main__":
    main()
