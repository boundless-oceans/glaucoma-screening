"""Shared data, model, metrics and portable inference; no external training data."""
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.functional import to_tensor, normalize


def read_rows(root, split):
    root = Path(root)
    with (root / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Image_name", "Age"} | ({"True_Label"} if split != "test" else set())
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"{split}.csv is missing columns: {required}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{split}.csv is empty")
    by_name = {}
    for path in (root / split).rglob("*"):
        if path.is_file():
            key = path.name.casefold()
            if key in by_name:
                raise ValueError(f"Ambiguous image name: {path.name}")
            by_name[key] = path
    seen = set()
    for row in rows:
        name = row["Image_name"]
        if not name or name != name.strip() or "/" in name or "\\" in name:
            raise ValueError(f"Invalid image name: {name!r}")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate CSV name: {name}")
        seen.add(name.casefold())
        if name.casefold() not in by_name:
            raise FileNotFoundError(f"{split}: {name}")
        row["path"] = str(by_name[name.casefold()])
        if split != "test" and row.get("True_Label") not in ("0", "1"):
            raise ValueError(f"Invalid label: {split}/{name}")
    return rows


def age_value(text):
    try:
        value = float(text)
        return value if math.isfinite(value) and 0 < value <= 110 else None
    except (ValueError, TypeError):
        return None


def fit_age(rows):
    values = [age_value(r["Age"]) for r in rows]
    valid = [x for x in values if x is not None]
    if not valid:
        raise ValueError("Training data has no usable ages")
    median = float(np.median(valid))
    filled = np.array([median if x is None else x for x in values])
    return {"median": median, "mean": float(filled.mean()),
            "std": max(float(filled.std()), 1.0), "valid_range": "0 < age <= 110"}


def subset(rows, limit, seed):
    if not limit or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    if "True_Label" not in rows[0]:
        return [rows[i] for i in sorted(rng.sample(range(len(rows)), limit))]
    if limit < 2:
        raise ValueError("Labeled sample limit must be >= 2")
    pos = [i for i, r in enumerate(rows) if r["True_Label"] == "1"]
    neg = [i for i, r in enumerate(rows) if r["True_Label"] == "0"]
    if not pos or not neg:
        raise ValueError("Both classes are required")
    npos = max(1, min(len(pos), limit - 1, round(limit * len(pos) / len(rows))))
    nneg = min(len(neg), limit - npos)
    npos = min(len(pos), limit - nneg)
    chosen = rng.sample(pos, npos) + rng.sample(neg, nneg)
    return [rows[i] for i in sorted(chosen)]


class FundusDataset(Dataset):
    def __init__(self, rows, age, size, training=False, seed=42, epoch=0):
        self.rows, self.age, self.size = rows, age, size
        self.training, self.seed, self.epoch = training, seed, epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            with Image.open(row["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            # Contain + letterbox preserves the entire field; no random crop.
            image = ImageOps.pad(image, (self.size, self.size),
                                 method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            if self.training:
                # Per-image/epoch seed makes worker prefetch and mid-epoch resume safe.
                rng = random.Random(self.seed + self.epoch * 1000003 + index * 9176)
                if rng.random() < 0.5:
                    image = ImageOps.mirror(image)
                image = image.rotate(rng.uniform(-10, 10), resample=Image.Resampling.BILINEAR)
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
            tensor = normalize(to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        except Exception as error:
            raise RuntimeError(f"Cannot read {row['path']}: {error}") from error
        value = age_value(row["Age"])
        missing = value is None
        value = self.age["median"] if missing else value
        age = torch.tensor([(value - self.age["mean"]) / self.age["std"], float(missing)], dtype=torch.float32)
        label = float(row.get("True_Label", -1))
        return tensor, age, torch.tensor(label, dtype=torch.float32), row["Image_name"]


class FusionModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(features + 2, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, image, age):
        return self.head(torch.cat([self.backbone(image), age], dim=1)).squeeze(1)


def make_loader(dataset, batch_size, workers, order=None, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=order,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      generator=generator, persistent_workers=False)


def metrics(labels, probabilities, threshold=0.5):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(y) == 0 or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid metric inputs")
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if not positives or not negatives:
        raise ValueError("Validation needs both classes")
    order = np.argsort(-p, kind="stable")
    sy, sp = y[order], p[order]
    end = np.r_[np.flatnonzero(np.diff(sp)), len(sp) - 1]
    tp = np.cumsum(sy)[end].astype(float)
    fp = (end + 1 - tp).astype(float)
    recall, precision = tp / positives, tp / (tp + fp)
    tpr, fpr = np.r_[0, recall], np.r_[0, fp / negatives]
    # Explicit tie grouping, trapezoidal ROC and PR; AP is reported separately.
    auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    pr_precision, pr_recall = np.r_[1, precision], np.r_[0, recall]
    auprc = float(np.sum(np.diff(pr_recall) * (pr_precision[1:] + pr_precision[:-1]) / 2))
    ap = float(np.sum(np.diff(pr_recall) * precision))
    pred = p >= threshold
    true_pos = int(((y == 1) & pred).sum())
    true_neg = int(((y == 0) & ~pred).sum())
    false_pos, false_neg = negatives - true_neg, positives - true_pos
    return dict(auroc=auroc, auprc_trapezoid=auprc, average_precision=ap,
                threshold=float(threshold), recall=true_pos / positives,
                specificity=true_neg / negatives, precision=true_pos / max(true_pos + false_pos, 1),
                tp=true_pos, tn=true_neg, fp=false_pos, fn=false_neg,
                hard_label_auroc=(true_pos / positives + true_neg / negatives) / 2)


def write_predictions(path, names, probabilities, threshold=0.5, labels=None):
    if len(names) != len(probabilities) or len(set(names)) != len(names):
        raise ValueError("Prediction length mismatch or duplicate image names")
    fields = ["Image_name", "Probability", "Pred_Label"] + (["True_Label"] if labels is not None else [])
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (name, probability) in enumerate(zip(names, probabilities)):
            probability = float(probability)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid probability for {name}")
            row = dict(Image_name=name, Probability=f"{probability:.10f}", Pred_Label=int(probability >= threshold))
            if labels is not None:
                row["True_Label"] = int(labels[index])
            writer.writerow(row)
    os.replace(temporary, path)


def atomic_save(state, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def data_signature(root):
    return {split: hashlib.sha256((Path(root) / f"{split}.csv").read_bytes()).hexdigest()
            for split in ("train", "valid", "test")}


def json_save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@torch.inference_mode()
def evaluate(model, loader, device, amp=False):
    model.eval()
    names, probabilities, labels = [], [], []
    for index, (image, age, label, name) in enumerate(loader):
        image, age = image.to(device, non_blocking=True), age.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(image, age)
        probabilities.extend(logits.float().sigmoid().cpu().tolist())
        labels.extend(label.tolist())
        names.extend(name)
        if (index + 1) % 50 == 0:
            print(f"inference: {len(names)}/{len(loader.dataset)}", flush=True)
    return names, probabilities, labels


def predict_checkpoint(checkpoint, root, output, device="cuda", batch_size=32, workers=4,
                       threshold=0.5, allow_sample=False):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be within [0, 1]")
    device = torch.device(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    if any(config.get(k, 0) for k in ("train_limit", "valid_limit", "test_limit")) and not allow_sample:
        raise ValueError("This is a sample/debug checkpoint. Train a full run before creating a submission.")
    rows = read_rows(root, "test")
    if allow_sample:
        rows = subset(rows, config.get("test_limit", 0), config["seed"])
    if data_signature(root) != state["data_signature"]:
        raise ValueError("CSV data differs from the checkpoint; use the original dataset")
    model = FusionModel().to(device)
    model.load_state_dict(state["model"])
    dataset = FundusDataset(rows, state["age_stats"], config["image_size"])
    loader = make_loader(dataset, batch_size, workers)
    names, probabilities, _ = evaluate(model, loader, device, config["amp"])
    assert names == [r["Image_name"] for r in rows]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output, names, probabilities, threshold)
    json_save(str(output) + ".json", dict(checkpoint=str(Path(checkpoint).resolve()),
              checkpoint_epoch=state["epoch"], threshold=threshold, rows=len(names),
              sampled=allow_sample and any(config.get(k, 0) for k in ("train_limit", "valid_limit", "test_limit")),
              validation=state.get("best_metrics")))
    print(f"Predictions written: {output} ({len(names)} rows)", flush=True)
