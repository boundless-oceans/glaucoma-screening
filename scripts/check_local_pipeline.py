"""SERVER ONLY: synthetic global/local training, cache, resume and fixed fusion check.

Uses no official data or pretrained downloads. Does train tiny models explicitly.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    import torch
    from PIL import Image, ImageDraw
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from glaucoma_training.common import FusionModel
    from glaucoma_training.roi_cache import load_roi_cache
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA unavailable")
    torch.set_num_threads(4)
    model = FusionModel(model_name="efficientnet_b0", use_age=False).to(args.device).eval()
    image = torch.rand(2, 3, 64, 64, device=args.device)
    with torch.inference_mode():
        expected = model(image, torch.zeros(2, 2, device=args.device))
        torch.testing.assert_close(model(image, torch.full((2, 2), float("nan"), device=args.device)), expected, rtol=0, atol=0)
    del model
    output = root / "outputs" / ("check_local_pipeline_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    data = output / "data"; data.mkdir(parents=True)
    for split,count in (("train", 8), ("valid", 4), ("test", 4)):
        (data / split).mkdir()
        with (data / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["Image_name", "Age"] + ([] if split == "test" else ["True_Label"]))
            writer.writeheader()
            for i in range(count):
                name = f"{split}_{i}.jpg"
                image = Image.new("RGB", (160, 128), "black")
                draw = ImageDraw.Draw(image); draw.ellipse((5, 5, 155, 123), fill=(100+i*5, 50, 20))
                draw.ellipse((35, 45, 60, 70), fill=(220, 180, 90))
                image.save(data / split / name)
                row = dict(Image_name=name, Age=50+i)
                if split != "test": row["True_Label"] = i % 2
                writer.writerow(row)
    base = dict(model_name="resnet18", image_size=64, batch_size=2, num_workers=0, epochs=1, pretrained=False, amp=False)
    global_config = output / "global.json"; global_config.write_text(json.dumps(base))
    local_config = output / "local.json"
    local_config.write_text(json.dumps({**base, "model_name":"efficientnet_b0", "use_age":False, "input_view":"local", "jpeg_draft_size":128}))
    roi_config = output / "roi.json"; roi_config.write_text(json.dumps(dict(classic_roi_version=2, probe_size=128)))
    env = {**os.environ, "PYTHONPATH":str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def run(module, options, expected_code=0):
        result = subprocess.run([sys.executable, "-u", "-m", "glaucoma_training."+module, *map(str,options)], cwd=root, env=env)
        if result.returncode != expected_code: raise RuntimeError(f"{module}: expected {expected_code}, got {result.returncode}")

    run("build_roi_cache", ["--data-root", data, "--output-dir", output/"cache", "--config", local_config,
                            "--roi-config", roi_config, "--workers", "2"])
    load_roi_cache(data, output/"cache")
    train = ["--data-root", data, "--device", args.device, "--skip-predict"]
    run("train", train + ["--config", global_config, "--run-dir", output/"global"])
    global_hash = hashlib.sha256((output/"global/best.pt").read_bytes()).hexdigest()
    local = train + ["--config", local_config, "--run-dir", output/"local", "--roi-cache", output/"cache"]
    run("train", local + ["--max-hours", "0.000000001"], 3)
    # Resume from the run-side coordinate snapshot, without the external cache argument.
    run("train", train + ["--config", local_config, "--run-dir", output/"local"])
    progress = json.loads((output/"local/progress.json").read_bytes())
    if progress["status"] != "completed": raise RuntimeError("Local training not completed")
    run("evaluate_fusion", ["--data-root", data, "--global-checkpoint", output/"global/best.pt",
                            "--local-checkpoint", output/"local/best.pt", "--roi-cache", output/"cache",
                            "--device", args.device, "--batch-size", "2", "--workers", "0", "--output-dir", output/"fusion"])
    manifest = json.loads((output/"fusion/manifest.json").read_bytes())
    assert manifest["status"] == "completed" and set(manifest["validation"]) == {"global", "local", "fusion"}
    assert hashlib.sha256((output/"global/best.pt").read_bytes()).hexdigest() == global_hash
    with (output/"fusion/submission.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream); rows = list(reader)
        assert reader.fieldnames == ["Image_name", "Probability", "Pred_Label"]
    assert [r["Image_name"] for r in rows] == [f"test_{i}.jpg" for i in range(4)]
    assert manifest["files"]["submission.csv"]["rows"] == 4
    print(f"Local pipeline integration passed: {output}; global checkpoint unchanged.")


if __name__ == "__main__": main()
