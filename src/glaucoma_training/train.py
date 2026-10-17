"""ResNet18 + age training. Run --help for portable CLI options."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import platform
import signal
import sys
import time
from pathlib import Path

import torch
import torchvision
from torch import nn
from . import CONFIG_DIR, DEFAULT_DATA_ROOT, PROJECT_ROOT

from .common import (FusionModel, FundusDataset, atomic_save, data_signature, evaluate,
                    fit_age, json_save, make_loader, metrics, predict_checkpoint,
                    read_rows, subset, write_predictions)
from .run_utils import RunLock, protect_output, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "baseline")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "train.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-hours", type=float, default=0, help="Per invocation training budget; 0 means no limit")
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.max_hours) or args.max_hours < 0:
        parser.error("--max-hours must be finite and nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    defaults = json.loads((CONFIG_DIR / "train.json").read_text(encoding="utf-8-sig"))
    unknown = set(config) - set(defaults)
    if unknown:
        parser.error(f"Unknown configuration keys: {unknown}")
    config = {**defaults, **config}
    try:
        validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use a GPU instance, or --device cpu for a small debug run.")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
    root, run = args.data_root.resolve(), args.run_dir.resolve()
    try:
        protect_output(root, run, directory=True)
    except ValueError as error:
        parser.error(str(error))
    run.mkdir(parents=True, exist_ok=True)
    run_lock = RunLock(run)
    last = run / "last.pt"
    existing = {p.name for p in run.iterdir()} - {".run.lock"}
    if args.resume == "never" and existing:
        parser.error("Use an empty --run-dir for --resume never")
    if not last.exists() and existing - {"last.pt.tmp"}:
        parser.error("Run directory has files but no last.pt. Choose a new --run-dir.")
    signature = data_signature(root)
    full_train = read_rows(root, "train")
    train_rows = subset(full_train, config["train_limit"], config["seed"])
    valid_rows = subset(read_rows(root, "valid"), config["valid_limit"], config["seed"])
    # Catch incomplete uploads before downloading weights or training for hours.
    test_rows = read_rows(root, "test")
    print(f"Preflight paths OK: train={len(full_train)}, valid={len(valid_rows)}, test={len(test_rows)}", flush=True)
    del test_rows
    for split, rows in (("train", train_rows), ("valid", valid_rows)):
        if {r["True_Label"] for r in rows} != {"0", "1"}:
            raise ValueError(f"{split} must include both classes")
    state = torch.load(last, map_location="cpu", weights_only=True) if last.exists() else None
    if state:
        if state["config"] != config:
            raise ValueError("Configuration differs from checkpoint. Restore the old config or choose a new run directory.")
        if state["data_signature"] != signature:
            raise ValueError("Dataset CSV files differ from checkpoint")
        if state["best_metrics"] is not None and not (run / "best.pt").is_file():
            raise FileNotFoundError("best.pt is missing. Restore it together with last.pt before resuming.")
    age = state["age_stats"] if state else fit_age(train_rows)
    model = FusionModel(pretrained=config["pretrained"] and state is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"], device=device))
    epoch, next_batch, global_step, loss_sum, seen = 0, 0, 0, 0.0, 0
    best_score, best_metrics = (-1.0, -1.0), None
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state["scaler"] and amp:
            scaler.load_state_dict(state["scaler"])
        epoch, next_batch, global_step = state["epoch"], state["next_batch"], state["global_step"]
        loss_sum, seen = state["loss_sum"], state["seen"]
        best_score, best_metrics = tuple(state["best_score"]), state["best_metrics"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming: epoch={epoch + 1}, next_batch={next_batch}, optimizer_steps={global_step}", flush=True)
    env = dict(python=sys.version, torch=str(torch.__version__), torchvision=str(torchvision.__version__),
               cuda=torch.version.cuda, platform=platform.platform(), device=str(device),
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    print(json.dumps(env, indent=2), flush=True)
    print(f"Samples: train={len(train_rows)}, valid={len(valid_rows)}; age stats: {age}", flush=True)
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True
        print(f"Signal {signum}: will save after the current training batch/validation phase.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()

    def expired():
        return stop["requested"] or (args.max_hours > 0 and time.monotonic() - started >= args.max_hours * 3600)

    def save(status="training"):
        atomic_save(dict(format_version=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), config=config, age_stats=age, data_signature=signature,
                         epoch=epoch, next_batch=next_batch, global_step=global_step,
                         loss_sum=loss_sum, seen=seen, best_score=list(best_score), best_metrics=best_metrics,
                         torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []), last)
        json_save(run / "progress.json", dict(status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
                  epoch_index=epoch, next_batch=next_batch, global_step=global_step,
                  total_epochs=config["epochs"], best_metrics=best_metrics))
        print(f"Checkpoint saved: {last}; status={status}, epoch_index={epoch}, "
              f"next_batch={next_batch}, global_step={global_step}", flush=True)

    # Create a recoverable state before writing metadata or starting the first batch.
    if state is None:
        save("initialized")
    json_save(run / "environment.json", env)
    json_save(run / "config.json", config)
    json_save(run / "age_stats.json", age)
    valid_dataset = FundusDataset(valid_rows, age, config["image_size"])
    valid_loader = make_loader(valid_dataset, config["batch_size"], config["num_workers"])
    while epoch < config["epochs"]:
        dataset = FundusDataset(train_rows, age, config["image_size"], True, config["seed"], epoch)
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config["seed"] + epoch)).tolist()
        batches = math.ceil(len(dataset) / config["batch_size"])
        offset = next_batch * config["batch_size"]
        loader = make_loader(dataset, config["batch_size"], config["num_workers"], order[offset:], config["seed"] + epoch)
        lr = config["learning_rate"] * (0.1 + 0.9 * (1 + math.cos(math.pi * epoch / config["epochs"])) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        if expired():
            save("paused")
            print("Paused and saved last.pt. Run the same command to resume.", flush=True)
            return
        for image, ages, labels, _ in loader:
            image, ages, labels = (x.to(device, non_blocking=True) for x in (image, ages, labels))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(image, ages)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; stop and inspect data/config")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            next_batch += 1
            global_step += 1
            loss_sum += float(loss.item()) * len(labels)
            seen += len(labels)
            if next_batch % 25 == 0 or next_batch == batches:
                print(f"epoch {epoch + 1}/{config['epochs']} batch {next_batch}/{batches} "
                      f"loss={loss_sum / seen:.5f} lr={lr:.7f}", flush=True)
            if global_step % config["save_every_steps"] == 0:
                save()
            if expired():
                save("paused")
                print("Paused and saved last.pt. Resume with the same command; no automatic inference on pause.", flush=True)
                return
        # Keep completed training safe even if validation is interrupted externally.
        save("validating")
        print(f"Validating epoch {epoch + 1}...", flush=True)
        names, probabilities, labels = evaluate(model, valid_loader, device, amp)
        result = metrics(labels, probabilities)
        result.update(epoch=epoch + 1, train_loss=loss_sum / seen, learning_rate=lr, global_step=global_step)
        score = (result["auroc"], result["auprc_trapezoid"])
        if score > best_score:
            best_score, best_metrics = score, result
            atomic_save(dict(format_version=1, model=model.state_dict(), config=config, age_stats=age,
                             data_signature=signature, epoch=epoch + 1, best_metrics=result), run / "best.pt")
            write_predictions(run / "valid_predictions.csv", names, probabilities, labels=labels)
            json_save(run / "best_metrics.json", result)
            threshold_rows = [metrics(labels, probabilities, i / 100) for i in range(1, 100)]
            with (run / "valid_thresholds.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(threshold_rows[0]))
                writer.writeheader()
                writer.writerows(threshold_rows)
            print("New best checkpoint saved.", flush=True)
        with (run / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        print(json.dumps(result), flush=True)
        epoch += 1
        next_batch, loss_sum, seen = 0, 0.0, 0
        save("completed" if epoch >= config["epochs"] else "paused" if expired() else "training")
        if expired():
            print("Paused after validation. best.pt is available; rerun to continue.", flush=True)
            return
    # Also update progress when resuming an already completed run.
    if epoch >= config["epochs"]:
        save("completed")
    # Release optimizer memory before inference.
    del optimizer, model, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sampled = any(config[k] for k in ("train_limit", "valid_limit", "test_limit"))
    if not args.skip_predict:
        filename = "DEBUG_predictions.csv" if sampled else "submission.csv"
        predict_checkpoint(run / "best.pt", root, run / filename, args.device,
                           config["batch_size"], config["num_workers"], allow_sample=sampled)
    print(f"Training complete. Best validation metrics: {best_metrics}", flush=True)


if __name__ == "__main__":
    main()
