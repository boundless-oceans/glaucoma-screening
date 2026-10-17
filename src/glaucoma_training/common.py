"""Shared data, model, metrics and portable inference; no external training data."""
import os
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.functional import to_tensor, normalize
from .run_utils import protect_output
from .data_utils import (read_rows, age_value, fit_age, subset, metrics,
                        write_predictions, data_signature, json_save)


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


def atomic_save(state, path):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers must be nonnegative")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use a GPU instance or explicitly select --device cpu")
    output = protect_output(root, output)
    if output == Path(checkpoint).resolve():
        raise ValueError("Prediction output must not overwrite the checkpoint")
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
    if names != [r["Image_name"] for r in rows]:
        raise ValueError("Predictions differ from test.csv row order")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output, names, probabilities, threshold)
    json_save(str(output) + ".json", dict(checkpoint=str(Path(checkpoint).resolve()),
              checkpoint_epoch=state["epoch"], threshold=threshold, rows=len(names),
              sampled=allow_sample and any(config.get(k, 0) for k in ("train_limit", "valid_limit", "test_limit")),
              validation=state.get("best_metrics")))
    print(f"Predictions written: {output} ({len(names)} rows)", flush=True)
