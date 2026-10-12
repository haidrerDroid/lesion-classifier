"""Train a ResNet18 + EfficientNet-B0 probability ensemble."""

from __future__ import annotations

import argparse
import json
import ssl
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from train_model import (
    LesionDataset,
    build_transforms,
    load_records,
    make_splits,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ISIC-images"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def load_weights() -> None:
    try:
        import certifi

        ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(
            cafile=certifi.where()
        )
    except ImportError:
        warnings.warn("certifi is unavailable; pretrained downloads may fail")


def build_models(num_classes: int, pretrained: bool) -> list[nn.Module]:
    load_weights()
    models: list[nn.Module] = []
    try:
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        efficientnet = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
    except Exception as error:
        warnings.warn(f"Could not load pretrained weights ({error}); using scratch models")
        resnet = resnet18(weights=None)
        efficientnet = efficientnet_b0(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    efficientnet.classifier[1] = nn.Linear(
        efficientnet.classifier[1].in_features, num_classes
    )
    models.extend([resnet, efficientnet])
    return models


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, batch_targets in loader:
        images, batch_targets = images.to(device), batch_targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, batch_targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(images)
        targets.extend(batch_targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), f1_score(
        targets, predictions, average="macro", zero_division=0
    )


def predict_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_targets in loader:
            probabilities.append(model(images.to(device)).softmax(dim=1).cpu().numpy())
            targets.extend(batch_targets.tolist())
    return np.asarray(targets), np.concatenate(probabilities)


def extract_model_features(model, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "fc"):
        features = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features = model.layer4(features)
        return torch.flatten(model.avgpool(features), 1)
    features = model.features(images)
    return torch.flatten(model.avgpool(features), 1)


def collect_model_features(model, loader, device) -> np.ndarray:
    model.eval()
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in loader:
            features.append(extract_model_features(model, images.to(device)).cpu().numpy())
    return np.concatenate(features)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    records = load_records(args.data_dir)
    splits = make_splits(records, args.seed)
    for name, split in splits.items():
        split.to_csv(split_dir / f"{name}.csv", index=False)
    classes = sorted(records["dx"].unique())
    class_to_index = {label: index for index, label in enumerate(classes)}
    train_transform, eval_transform = build_transforms(args.image_size)
    datasets = {
        "train": LesionDataset(splits["train"], class_to_index, train_transform),
        "validation": LesionDataset(splits["validation"], class_to_index, eval_transform),
        "test": LesionDataset(splits["test"], class_to_index, eval_transform),
    }
    loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train")
        for name, dataset in datasets.items()
    }
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = [model.to(device) for model in build_models(len(classes), not args.from_scratch)]
    counts = splits["train"]["dx"].value_counts().reindex(classes).to_numpy()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            len(splits["train"]) / (len(classes) * counts),
            dtype=torch.float32,
            device=device,
        )
    )
    checkpoints = []
    for model_index, model in enumerate(models):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        best_f1 = -1.0
        best_state = None
        for epoch in range(1, args.epochs + 1):
            train_loss, train_f1 = run_epoch(
                model, loaders["train"], criterion, optimizer, device, True
            )
            validation_loss, validation_f1 = run_epoch(
                model, loaders["validation"], criterion, optimizer, device, False
            )
            print(
                f"model={model_index + 1}/2 epoch={epoch:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
                f"val_loss={validation_loss:.4f} val_f1={validation_f1:.4f}"
            )
            if validation_f1 > best_f1:
                best_f1 = validation_f1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        model.load_state_dict(best_state)
        checkpoints.append(model)

    validation_targets, validation_probabilities = predict_probabilities(
        checkpoints[0], loaders["validation"], device
    )
    test_targets, test_probabilities = predict_probabilities(
        checkpoints[0], loaders["test"], device
    )
    validation_predictions = [
        predict_probabilities(model, loaders["validation"], device)[1]
        for model in checkpoints
    ]
    test_predictions = [
        predict_probabilities(model, loaders["test"], device)[1]
        for model in checkpoints
    ]
    validation_probabilities = np.mean(validation_predictions, axis=0)
    test_probabilities = np.mean(test_predictions, axis=0)
    malignant_indices = [
        index for index, label in enumerate(classes) if label in {"mel", "bcc", "akiec"}
    ]
    validation_risk = validation_probabilities[:, malignant_indices].sum(axis=1)
    validation_malignant = np.asarray(
        [classes[target] in {"mel", "bcc", "akiec"} for target in validation_targets],
        dtype=int,
    )
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold = float(
        thresholds[
            np.argmax(
                [
                    fbeta_score(
                        validation_malignant,
                        validation_risk >= value,
                        beta=2,
                        zero_division=0,
                    )
                    for value in thresholds
                ]
            )
        ]
    )
    train_features = [
        collect_model_features(model, loaders["train"], device) for model in checkpoints
    ]
    validation_features = [
        collect_model_features(model, loaders["validation"], device)
        for model in checkpoints
    ]
    normalized_train = [features / np.linalg.norm(features, axis=1, keepdims=True) for features in train_features]
    normalized_validation = [features / np.linalg.norm(features, axis=1, keepdims=True) for features in validation_features]
    similarities = [
        (validation @ train.T).max(axis=1)
        for validation, train in zip(normalized_validation, normalized_train)
    ]
    ood_threshold = float(np.percentile(np.mean(similarities, axis=0), 5))
    risk = test_probabilities[:, malignant_indices].sum(axis=1)
    test_malignant = np.asarray(
        [classes[target] in {"mel", "bcc", "akiec"} for target in test_targets], dtype=int
    )
    binary_predictions = (risk >= threshold).astype(int)
    metrics = {
        "model_version": "resnet18-efficientnetb0-ensemble-v1",
        "balanced_accuracy": balanced_accuracy_score(
            test_targets, test_probabilities.argmax(axis=1)
        ),
        "macro_f1": f1_score(
            test_targets, test_probabilities.argmax(axis=1), average="macro", zero_division=0
        ),
        "malignant_risk_threshold": threshold,
        "malignant_risk_recall": recall_score(test_malignant, binary_predictions, zero_division=0),
        "malignant_risk_precision": precision_score(test_malignant, binary_predictions, zero_division=0),
        "malignant_risk_specificity": recall_score(1 - test_malignant, 1 - binary_predictions, zero_division=0),
        "malignant_risk_roc_auc": roc_auc_score(test_malignant, risk),
        "classification_report": classification_report(
            test_targets, test_probabilities.argmax(axis=1), target_names=classes, output_dict=True, zero_division=0
        ),
    }
    checkpoint = {
        "model_version": metrics["model_version"],
        "classes": classes,
        "image_size": args.image_size,
        "risk_threshold": threshold,
        "uncertainty_margin": 0.03,
        "ood_similarity_threshold": ood_threshold,
        "model_states": [model.state_dict() for model in checkpoints],
        "reference_features": [features.astype(np.float32) for features in normalized_train],
    }
    torch.save(checkpoint, args.output_dir / "ensemble_model.pt")
    (args.output_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved ensemble to {args.output_dir / 'ensemble_model.pt'}")


if __name__ == "__main__":
    main()