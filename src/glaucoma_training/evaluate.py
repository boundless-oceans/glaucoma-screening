"""Model evaluation CLI (not a unit test): validate, then predict unlabeled test data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from .common import FusionModel, FundusDataset, evaluate, make_loader
from .data_utils import data_signature, json_save, metrics, read_rows, subset, write_predictions
from .run_utils import protect_output
from .model_config import checkpoint_model_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-sample", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.batch_size <= 0 or args.workers < 0:
        parser.error("Invalid threshold, batch-size or workers")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable. Run on a GPU instance.")
    if not args.checkpoint.is_file():
        parser.error("best.pt not found. Finish at least one validation epoch, or supply --checkpoint.")
    output = protect_output(args.data_root, args.output_dir, directory=True)
    # Read once so validation and test refer to exactly the same checkpoint,
    # even if training replaces best.pt concurrently.
    with args.checkpoint.open("rb") as stream:
        content = stream.read()
    import io
    checkpoint_hash = hashlib.sha256(content).hexdigest()
    state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    del content
    config = state["config"]
    model_name = checkpoint_model_name(state)
    sampled = any(config.get(k, 0) for k in ("train_limit", "valid_limit", "test_limit"))
    if sampled and not args.allow_sample:
        parser.error("Debug checkpoint refused for a full evaluation. Use bash scripts/test.sh --smoke for debug.")
    if data_signature(args.data_root) != state["data_signature"]:
        raise ValueError("CSV files do not match the checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device("cuda")
    model = FusionModel(model_name=model_name).to(device)
    model.load_state_dict(state["model"])
    report = dict(start_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=checkpoint_hash, checkpoint_epoch=state["epoch"], threshold=args.threshold,
                  model_name=model_name,
                  sampled=sampled, age_stats=state["age_stats"], config=config,
                  torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    json_save(output / "checkpoint_info.json", report)
    for split in ("valid", "test"):
        rows = read_rows(args.data_root, split)
        if args.allow_sample:
            rows = subset(rows, config.get(f"{split}_limit", 0), config["seed"])
        dataset = FundusDataset(rows, state["age_stats"], config["image_size"])
        loader = make_loader(dataset, args.batch_size, args.workers)
        print(f"Evaluating {split}: {len(rows)} images", flush=True)
        names, probabilities, labels = evaluate(model, loader, device, config["amp"])
        if names != [row["Image_name"] for row in rows]:
            raise ValueError(f"{split} prediction order mismatch")
        if split == "valid":
            result = metrics(labels, probabilities, args.threshold)
            result.update(rows=len(rows), sampled=sampled, checkpoint_sha256=checkpoint_hash)
            json_save(output / "validation_metrics.json", result)
            write_predictions(output / "validation_predictions.csv", names, probabilities, args.threshold, labels)
            print("Validation metrics: " + json.dumps(result), flush=True)
        else:
            filename = "DEBUG_predictions.csv" if sampled else "submission.csv"
            write_predictions(output / filename, names, probabilities, args.threshold)
            print(f"Saved {len(rows)} test predictions: {output / filename}", flush=True)
            print("Test labels are unavailable; no test AUROC/accuracy can be computed.", flush=True)
    report["end_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "completed"
    json_save(output / "checkpoint_info.json", report)


if __name__ == "__main__":
    main()
