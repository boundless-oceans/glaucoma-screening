"""Server-only model integration check with synthetic data; no pretrained downloads."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    from PIL import Image
    root = Path(__file__).resolve().parents[1]
    folder = root / "outputs" / ("check_finetune_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    data = folder / "data"
    data.mkdir(parents=True)
    for split, count in [("train", 8), ("valid", 4), ("test", 4)]:
        (data / split).mkdir()
        with (data / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["Image_name", "Age"] + ([] if split == "test" else ["True_Label"])
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in range(count):
                name = f"{split}_{i}.jpg"
                Image.new("RGB", (80, 64), (30 + 20 * i, 60, 40)).save(data / split / name)
                row = dict(Image_name=name, Age=50 + i)
                if split != "test": row["True_Label"] = i % 2
                writer.writerow(row)
    base = dict(model_name="resnet18", image_size=64, batch_size=2, num_workers=0,
                epochs=1, pretrained=False, amp=False)
    source_config, fine_config = folder / "source.json", folder / "fine.json"
    source_config.write_text(json.dumps(base))
    fine_config.write_text(json.dumps({**base, "finetune": True, "learning_rate": 1e-5}))
    env = {**os.environ, "PYTHONPATH": str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def run(module, options, expected=0):
        result = subprocess.run([sys.executable, "-u", "-m", "glaucoma_training." + module, *map(str, options)], env=env, cwd=root)
        if result.returncode != expected: raise RuntimeError(f"{module}: expected {expected}, got {result.returncode}")

    options = ["--data-root", data, "--device", args.device, "--skip-predict"]
    run("train", options + ["--config", source_config, "--run-dir", folder / "source"])
    fine = options + ["--config", fine_config, "--run-dir", folder / "fine"]
    run("train", fine + ["--init-checkpoint", folder / "source/best.pt", "--max-hours", "0.000000001"], 3)
    # Resume without depending on the external source checkpoint path.
    run("train", fine)
    initial = json.loads((folder / "fine/initial_validation.json").read_text())["metrics"]
    best = json.loads((folder / "fine/best_metrics.json").read_text())
    if (best["auroc"], best["auprc_trapezoid"]) < (initial["auroc"], initial["auprc_trapezoid"]):
        raise RuntimeError("Fine-tuning replaced the baseline with a worse model")
    progress = json.loads((folder / "fine/progress.json").read_text())
    if progress["status"] != "completed": raise RuntimeError("Fine-tuning did not finish")
    run("evaluate", ["--data-root", data, "--checkpoint", folder / "fine/best.pt", "--output-dir", folder / "evaluation",
                     "--device", args.device, "--workers", "0", "--batch-size", "2"])
    with (folder / "evaluation/submission.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames != ["Image_name", "Probability", "Pred_Label"] or [r["Image_name"] for r in rows] != [f"test_{i}.jpg" for i in range(4)]:
            raise RuntimeError("Wrong submission schema/count/order")
    print(f"Fine-tuning integration passed: {folder}")


if __name__ == "__main__":
    main()
