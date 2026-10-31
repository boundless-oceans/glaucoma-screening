"""Explicit SERVER diagnostic: compare short training loops without saving weights.

Runs forward/backward/AdamW on official train only. Does not continue any run,
write checkpoints, select models, or predict test. Do not run on local computers.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from . import DEFAULT_DATA_ROOT, PROJECT_ROOT
from .common import FusionModel, FundusDataset, make_loader
from .data_utils import read_rows, data_signature, json_save
from .model_config import checkpoint_model_name, uses_age, input_view
from .preprocess_config import preprocessing_options
from .run_utils import protect_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--draft-size", type=int, default=640)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--order", nargs=2, choices=["full", "draft"], default=["full", "draft"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0 or args.workers < 0 or len(set(args.order)) != 2:
        parser.error("Invalid steps/warmup/workers/order")
    if not torch.cuda.is_available():
        parser.error("Run this explicit training diagnostic on a GPU server")
    content = args.checkpoint.read_bytes()
    checkpoint_sha = hashlib.sha256(content).hexdigest()
    import io
    state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    del content
    config = state["config"]
    preprocessing_options({**config, "jpeg_draft_size": args.draft_size})
    if args.draft_size < config["image_size"]:
        parser.error("draft-size must be at least the model input size")
    if state["data_signature"] != data_signature(args.data_root):
        parser.error("CSV files differ from source checkpoint")
    if any(config.get(k, 0) for k in ("train_limit", "valid_limit", "test_limit")):
        parser.error("Use a full-data reference checkpoint")
    rows = read_rows(args.data_root, "train")
    batch = config["batch_size"]
    count = (args.warmup + args.steps) * batch
    if count > len(rows):
        parser.error("Requested warmup+steps exceeds one full training pass")
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(config["seed"])).tolist()[:count]
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    results = {}
    for mode in args.order:
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        options = {**config, "jpeg_draft_size": 0 if mode == "full" else args.draft_size}
        if input_view(config) == "local":
            raise ValueError("This decoder A/B script only benchmarks global input, not cached local crops")
        model = FusionModel(model_name=checkpoint_model_name(state), use_age=uses_age(config)).to(device)
        model.load_state_dict(state["model"])
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        amp = config["amp"]
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"], device=device))
        dataset = FundusDataset(rows, state["age_stats"], config["image_size"], training=True,
                                seed=config["seed"], epoch=0, preprocessing=options)
        loader = make_loader(dataset, batch, args.workers, order, config["seed"], device=device)
        iterator = iter(loader)

        def step():
            image, ages, labels, _ = next(iterator)
            image, ages, labels = (x.to(device, non_blocking=True) for x in (image, ages, labels))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = criterion(model(image, ages), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite diagnostic loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            return scaler.get_scale() >= old_scale

        for _ in range(args.warmup): step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        successful = 0
        for index in range(args.steps):
            successful += int(step())
            if (index + 1) % 25 == 0:
                print(f"{mode}: {index + 1}/{args.steps} timed batches", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results[mode] = dict(elapsed_seconds=elapsed, seconds_per_batch=elapsed / args.steps,
                             images_per_second=args.steps * batch / elapsed, successful_optimizer_updates=successful,
                             attempted_steps=args.steps, peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del iterator, loader, dataset, model, optimizer, scaler
        torch.cuda.empty_cache()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = protect_output(args.data_root, args.output_dir or PROJECT_ROOT / "outputs" / ("throughput_" + stamp), directory=True)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                  config=config, draft_size=args.draft_size, gpu=torch.cuda.get_device_name(0),
                  torch=str(torch.__version__), cuda=torch.version.cuda, workers=args.workers,
                  warmup_batches=args.warmup, order=args.order, results=results,
                  measured_training_ratio=results["full"]["elapsed_seconds"] / results["draft"]["elapsed_seconds"],
                  scope="DataLoader delivery + H2D + forward/backward + AdamW; excludes startup, checkpoints and validation",
                  caveat="OS caches uncontrolled; repeat with reversed --order. This is not full-epoch or T4 inference acceptance.")
    json_save(output / "report.json", report)
    print(json.dumps(report, indent=2))
    print(f"Report: {output}. No weights saved or existing runs modified.")


if __name__ == "__main__":
    main()
