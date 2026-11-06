"""Shared data, model, metrics and portable inference; no external training data."""
import os
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (resnet18, ResNet18_Weights, efficientnet_b2, EfficientNet_B2_Weights,
                                efficientnet_b0, EfficientNet_B0_Weights)
from torchvision.transforms.functional import to_tensor, normalize
from .run_utils import protect_output
from .model_config import checkpoint_model_name, model_name_from_config, uses_age, input_view
from .roi_cache import load_roi_cache, cache_rows, prepare_cached_image
from .preprocess_config import preprocessing_options
from .preprocessing import prepare_image
from .data_utils import (read_rows, age_value, fit_age, subset, metrics,
                        write_predictions, data_signature, json_save)


class FundusDataset(Dataset):
    def __init__(self, rows, age, size, training=False, seed=42, epoch=0, preprocessing=None, roi_records=None):
        self.rows, self.age, self.size = rows, age, size
        self.training, self.seed, self.epoch = training, seed, epoch
        self.preprocessing = preprocessing_options(preprocessing or {})
        self.roi_records = roi_records

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            if self.roi_records is not None:
                image = prepare_cached_image(Path(row["path"]).read_bytes(),
                    self.roi_records[row["Image_name"].casefold()], self.size, self.preprocessing["jpeg_draft_size"])
            else:
                with Image.open(row["path"]) as source:
                    image, _ = prepare_image(source, self.size, self.preprocessing)
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
    def __init__(self, pretrained=False, model_name="resnet18", use_age=True):
        super().__init__()
        self.use_age = uses_age({"use_age": use_age})
        self.model_name = model_name_from_config({"model_name": model_name})
        if self.model_name == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            factory, weights = (efficientnet_b0, EfficientNet_B0_Weights) if self.model_name == "efficientnet_b0" else (efficientnet_b2, EfficientNet_B2_Weights)
            self.backbone = factory(weights=weights.DEFAULT if pretrained else None)
            features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Linear(features + (2 if self.use_age else 0), 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, image, age):
        features = self.backbone(image)
        if self.use_age:
            features = torch.cat([features, age], dim=1)
        return self.head(features).squeeze(1)


def make_loader(dataset, batch_size, workers, order=None, seed=42, device=None):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=order,
                      num_workers=workers, pin_memory=(torch.device(device).type == "cuda" if device is not None else torch.cuda.is_available()),
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
                       threshold=0.5, allow_sample=False, roi_cache=None):
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
    model_name = checkpoint_model_name(state)
    if any(config.get(k, 0) for k in ("train_limit", "valid_limit", "test_limit")) and not allow_sample:
        raise ValueError("This is a sample/debug checkpoint. Train a full run before creating a submission.")
    rows = read_rows(root, "test")
    if allow_sample:
        rows = subset(rows, config.get("test_limit", 0), config["seed"])
    if data_signature(root) != state["data_signature"]:
        raise ValueError("CSV data differs from the checkpoint; use the original dataset")
    cache = None
    if input_view(config) == "local":
        if not state.get("roi_cache_signature"): raise ValueError("Local checkpoint lacks ROI cache signature")
        cache = load_roi_cache(root, roi_cache or Path(checkpoint).parent / "roi_cache", state["roi_cache_signature"])
    elif roi_cache is not None:
        raise ValueError("Global prediction does not use --roi-cache")
    model = FusionModel(model_name=model_name, use_age=uses_age(config)).to(device)
    model.load_state_dict(state["model"])
    dataset = FundusDataset(rows, state["age_stats"], config["image_size"], preprocessing=config,
                            roi_records=cache_rows(cache, "test", rows) if cache else None)
    loader = make_loader(dataset, batch_size, workers, device=device)
    names, probabilities, _ = evaluate(model, loader, device, config["amp"])
    if names != [r["Image_name"] for r in rows]:
        raise ValueError("Predictions differ from test.csv row order")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output, names, probabilities, threshold)
    json_save(str(output) + ".json", dict(checkpoint=str(Path(checkpoint).resolve()),
              checkpoint_epoch=state["epoch"], threshold=threshold, rows=len(names),
              model_name=model_name,
              use_age=uses_age(config), input_view=input_view(config), roi_cache_signature=state.get("roi_cache_signature"),
              preprocessing=preprocessing_options(config),
              sampled=allow_sample and any(config.get(k, 0) for k in ("train_limit", "valid_limit", "test_limit")),
              validation=state.get("best_metrics")))
    print(f"Predictions written: {output} ({len(names)} rows)", flush=True)
