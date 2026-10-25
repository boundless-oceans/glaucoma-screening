"""Logged server entry point. This module itself never imports PyTorch."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback
from . import CONFIG_DIR, DEFAULT_DATA_ROOT, PROJECT_ROOT
from .model_config import default_run_name, load_config
from .run_utils import validate_config

CODE = PROJECT_ROOT


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(path, state):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class LoggedSession:
    def __init__(self, log_dir):
        self.directory = Path(log_dir)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / "console.log").open("a", encoding="utf-8", buffering=1)
        self.child = None
        self.interrupted = False

    def log(self, message):
        line = f"[{utc_now()}] {message}"
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def forward_stop(self, signum, frame):
        self.interrupted = True
        self.log(f"Received signal {signum}; asking child to stop. Wait for checkpoint confirmation.")
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)

    def run(self, name, command, *, optional=False, timeout=None, accepted_codes=(0,)):
        if self.interrupted:
            raise InterruptedError("Stop requested before the next stage")
        self.log(f"START {name}: {json.dumps([str(x) for x in command])}")
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1",
                       "PYTHONPATH": str(CODE / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        try:
            if timeout is not None:
                result = subprocess.run(command, cwd=CODE, env=environment, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=timeout)
                for line in result.stdout.splitlines():
                    self.log(line)
                code = result.returncode
            else:
                self.child = subprocess.Popen(command, cwd=CODE, env=environment, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", bufsize=1)
                try:
                    for line in self.child.stdout:
                        self.log(line.rstrip("\r\n"))
                    code = self.child.wait()
                finally:
                    self.child.stdout.close()
                    if self.child.poll() is None:
                        self.child.terminate()
                        self.child.wait()
                    self.child = None
            self.log(f"END {name}: exit_code={code}")
            if code not in accepted_codes and not optional:
                raise subprocess.CalledProcessError(code, command)
            return code
        except (OSError, subprocess.TimeoutExpired) as error:
            self.log(f"{name}: {type(error).__name__}: {error}")
            if not optional:
                raise
            return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-hours", type=float, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Test only; defaults to run-dir/best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Evaluation device")
    parser.add_argument("--splits", nargs="+", choices=["valid", "test"], default=["valid", "test"])
    args = parser.parse_args()
    if not math.isfinite(args.max_hours) or args.max_hours < 0:
        parser.error("--max-hours must be finite and nonnegative")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be within [0, 1]")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("Invalid batch-size/workers")
    if args.smoke and args.config:
        parser.error("Choose --smoke OR --config")
    if args.init_checkpoint and args.mode != "train":
        parser.error("--init-checkpoint is for training only")
    root = args.data_root.resolve()
    run = args.run_dir.resolve() if args.run_dir else None
    config = (args.config or CONFIG_DIR / ("smoke.json" if args.smoke else "train.json")).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}"
    session = LoggedSession(CODE / "logs" / f"{args.mode}_{stamp}")
    status_path = session.directory / "status.json"
    state = dict(status="running", mode=args.mode, stage="startup", start_utc=utc_now(),
                 pid=os.getpid(), data_root=str(root), run_dir=str(run) if run else None, argv=sys.argv)
    write_status(status_path, state)
    (CODE / "logs" / f"latest_{args.mode}.txt").write_text(str(session.directory), encoding="utf-8")
    signal.signal(signal.SIGINT, session.forward_stop)
    signal.signal(signal.SIGTERM, session.forward_stop)
    code = 0
    try:
        session.log(f"Logs: {session.directory}")
        resolved_config = load_config(config)
        validate_config(resolved_config)
        sampled = any(resolved_config[k] for k in ("train_limit", "valid_limit", "test_limit"))
        profile_model = resolved_config["model_name"]
        run = (run or CODE / "runs" / default_run_name(profile_model, sampled, resolved_config["remove_black_border"], resolved_config["finetune"])).resolve()
        state.update(run_dir=str(run), model_profile=profile_model, config_path=str(config))
        write_status(status_path, state)
        write_status(session.directory / "effective_config.json", resolved_config)
        session.log(f"Model profile: {profile_model}; run directory: {run}")
        session.log(f"Black-border removal: {resolved_config['remove_black_border']}; "
                    f"threshold={resolved_config['black_border_threshold']}, margin={resolved_config['black_border_margin']}")
        versions = {}
        for package in ("torch", "torchvision", "numpy", "Pillow"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "NOT INSTALLED"
        snapshot = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                        packages=versions, conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
                        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        max_hours=args.max_hours, threshold=args.threshold)
        snapshot["model_profile"] = profile_model
        write_status(session.directory / "environment.json", snapshot)
        session.log(json.dumps(snapshot, ensure_ascii=False))
        session.run("git revision", ["git", "rev-parse", "HEAD"], optional=True, timeout=10)
        session.run("GPU snapshot", ["nvidia-smi"], optional=True, timeout=15)
        session.run("disk space", ["df", "-h", str(CODE), str(root)], optional=True, timeout=10)
        session.run("memory", ["free", "-h"], optional=True, timeout=10)
        if args.mode == "train":
            state["stage"] = "preflight"
            write_status(status_path, state)
            (session.directory / "requested_config.json").write_bytes(config.read_bytes())
            session.run("preflight", [sys.executable, "-u", "-m", "glaucoma_training.preflight", "--data-root", str(root), "--config", str(config)])
            state["stage"] = "training"
            write_status(status_path, state)
            command = [sys.executable, "-u", "-X", "faulthandler", "-m", "glaucoma_training.train",
                        "--data-root", str(root), "--run-dir", str(run), "--config", str(config),
                        "--max-hours", str(args.max_hours), "--skip-predict"]
            if args.init_checkpoint:
                command.extend(["--init-checkpoint", str(args.init_checkpoint.resolve())])
            child_code = session.run("training", command, accepted_codes=(0, 3, 130))
            progress_path = run / "progress.json"
            if progress_path.is_file():
                state["training_progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
                state["status"] = state["training_progress"]["status"]
            else:
                raise RuntimeError("Training ended without progress.json")
            if state["status"] not in ("completed", "paused"):
                raise RuntimeError(f"Child exited without a terminal training status: {state['status']}")
            if child_code in (3, 130) and state["status"] != "paused":
                raise RuntimeError("Child pause/interruption code disagrees with progress.json")
            if state["status"] == "paused":
                progress = state["training_progress"]
                code = 130 if progress.get("stop_reason") == "user_signal" else 3
                session.log(f"PAUSED: completed epochs={progress['epoch_index']}/{progress['total_epochs']}; "
                            f"next_batch={progress['next_batch']}. Rerun the same command to resume.")
            session.log("Training invocation ended. Run bash scripts/test.sh once best.pt exists to evaluate and predict.")
        else:
            state["stage"] = "evaluation_and_prediction"
            output = run / "evaluations" / stamp
            state["output_dir"] = str(output)
            checkpoint = (args.checkpoint or run / "best.pt").resolve()
            state["checkpoint"] = str(checkpoint)
            write_status(status_path, state)
            command = [sys.executable, "-u", "-X", "faulthandler", "-m", "glaucoma_training.evaluate", "--data-root", str(root),
                       "--checkpoint", str(checkpoint), "--output-dir", str(output), "--threshold", str(args.threshold),
                       "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                       "--device", args.device, "--splits", *args.splits]
            if args.smoke:
                command.append("--allow-sample")
            session.run("evaluation and prediction", command)
            state["status"] = "completed"
            session.log(f"Results: {output}")
        if session.interrupted:
            code = 130
    except Exception as error:
        code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        if code < 0:
            code = 128 - code
        if session.interrupted:
            code = 130
        state.update(status="interrupted" if session.interrupted else "failed", error=str(error))
        session.log(traceback.format_exc())
    finally:
        state.update(exit_code=code, end_utc=utc_now())
        write_status(status_path, state)
        session.log(f"Session status={state['status']} exit_code={code}")
        session.stream.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
