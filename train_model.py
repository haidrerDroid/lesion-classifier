"""Train and evaluate a skin-lesion classifier from the ISIC image folder."""

from __future__ import annotations

import argparse
import json
import random
import ssl
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ISIC-images"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Do not download pretrained ResNet18 weights.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / "metadata.csv"
    image_dir = data_dir / "Images"
    metadata = pd.read_csv(metadata_path)

    duplicate_labels = metadata.groupby("image_id")["dx"].nunique()
    conflicting_ids = duplicate_labels[duplicate_labels > 1].index.tolist()
    if conflicting_ids:
        raise ValueError(
            "Some image IDs have conflicting labels: "
            + ", ".join(conflicting_ids[:10])
        )

    metadata = metadata.drop_duplicates(subset="image_id", keep="first").copy()
    metadata = metadata.dropna(subset=["dx"])
    image_paths = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    metadata["path"] = metadata["image_id"].map(image_paths)
    missing = metadata["path"].isna()
    if missing.any():
        missing_ids = metadata.loc[missing, "image_id"].tolist()
        raise FileNotFoundError(
            f"Missing {len(missing_ids)} image files, including {missing_ids[:5]}"
        )

    return metadata.reset_index(drop=True)


def make_splits(records: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    outer_splitter = StratifiedGroupKFold(
        n_splits=7, shuffle=True, random_state=seed
    )
    train_validation_indices, test_indices = next(
        outer_splitter.split(records, records["dx"], groups=records["lesion_id"])
    )
    train_validation = records.iloc[train_validation_indices].reset_index(drop=True)
    test_records = records.iloc[test_indices].reset_index(drop=True)

    inner_splitter = StratifiedGroupKFold(
        n_splits=6, shuffle=True, random_state=seed
    )
    train_indices, validation_indices = next(
        inner_splitter.split(
            train_validation,
            train_validation["dx"],
            groups=train_validation["lesion_id"],
        )
    )
    train_records = train_validation.iloc[train_indices]
    validation_records = train_validation.iloc[validation_indices]
    group_sets = {
        name: set(split["lesion_id"])
        for name, split in {
            "train": train_records,
            "validation": validation_records,
            "test": test_records,
        }.items()
    }
    if group_sets["train"] & group_sets["validation"] or group_sets["train"] & group_sets["test"] or group_sets["validation"] & group_sets["test"]:
        raise RuntimeError("A lesion appears in more than one data split")
    return {
        "train": train_records.reset_index(drop=True),
        "validation": validation_records.reset_index(drop=True),
        "test": test_records.reset_index(drop=True),
    }


class LesionDataset(Dataset):
    def __init__(self, records: pd.DataFrame, class_to_index: dict[str, int], transform):
        self.records = records.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records.iloc[index]
        image = Image.open(record["path"]).convert("RGB")
        return self.transform(image), self.class_to_index[record["dx"]]


def build_transforms(image_size: int) -> tuple[v2.Compose, v2.Compose]:
    normalize = v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    evaluation_transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    return train_transform, evaluation_transform


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        if pretrained:
            import certifi

            ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(
                cafile=certifi.where()
            )
        model = resnet18(weights=weights)
    except Exception as error:
        warnings.warn(
            f"Could not load pretrained weights ({error}); using random initialization."
        )
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), macro_f1, all_targets, all_predictions


def save_confusion_matrix(targets, predictions, classes, output_path: Path) -> None:
    matrix = confusion_matrix(targets, predictions, labels=range(len(classes)))
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="Test confusion matrix",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def collect_probabilities(model, loader, device):
    model.eval()
    all_targets: list[int] = []
    all_probabilities: list[list[float]] = []
    with torch.inference_mode():
        for images, targets in loader:
            probabilities = model(images.to(device)).softmax(dim=1).cpu()
            all_targets.extend(targets.tolist())
            all_probabilities.extend(probabilities.tolist())
    return np.asarray(all_targets), np.asarray(all_probabilities)


def extract_features(model, images: torch.Tensor) -> torch.Tensor:
    features = model.conv1(images)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    features = model.avgpool(features)
    return torch.flatten(features, 1)


def collect_features(model, loader, device) -> np.ndarray:
    model.eval()
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in loader:
            batch_features = extract_features(model, images.to(device))
            features.append(batch_features.cpu().numpy())
    return np.concatenate(features)


def choose_risk_threshold(
    targets: np.ndarray, probabilities: np.ndarray, classes: list[str]
) -> tuple[float, float]:
    malignant_indices = [index for index, label in enumerate(classes) if label in {"mel", "bcc", "akiec"}]
    malignant_probability = probabilities[:, malignant_indices].sum(axis=1)
    malignant_targets = np.asarray(
        [classes[target] in {"mel", "bcc", "akiec"} for target in targets], dtype=int
    )
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        fbeta_score(
            malignant_targets,
            malignant_probability >= threshold,
            beta=2,
            zero_division=0,
        )
        for threshold in thresholds
    ]
    threshold = float(thresholds[int(np.argmax(scores))])
    return threshold, float(malignant_probability.mean())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)

    records = load_records(args.data_dir)
    splits = make_splits(records, args.seed)
    for name, split in splits.items():
        split.to_csv(split_dir / f"{name}.csv", index=False)
    print(f"Using {len(records)} unique labeled images")
    print(records["dx"].value_counts().sort_index().to_string())
    print({name: len(split) for name, split in splits.items()})

    classes = sorted(records["dx"].unique())
    class_to_index = {name: index for index, name in enumerate(classes)}
    train_transform, evaluation_transform = build_transforms(args.image_size)
    datasets = {
        "train": LesionDataset(splits["train"], class_to_index, train_transform),
        "validation": LesionDataset(
            splits["validation"], class_to_index, evaluation_transform
        ),
        "test": LesionDataset(splits["test"], class_to_index, evaluation_transform),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, dataset in datasets.items()
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(len(classes), pretrained=not args.from_scratch).to(device)
    counts = splits["train"]["dx"].value_counts().reindex(classes).to_numpy()
    class_weights = len(splits["train"]) / (len(classes) * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_f1 = -1.0
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1, _, _ = run_epoch(
            model, loaders["train"], criterion, optimizer, device, training=True
        )
        validation_loss, validation_f1, _, _ = run_epoch(
            model, loaders["validation"], criterion, optimizer, device, training=False
        )
        scheduler.step(validation_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_macro_f1={train_f1:.4f} "
            f"val_loss={validation_loss:.4f} val_macro_f1={validation_f1:.4f} "
            f"lr={current_lr:.6f}"
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": args.image_size,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_targets, validation_probabilities = collect_probabilities(
        model, loaders["validation"], device
    )
    reference_features = collect_features(model, loaders["train"], device)
    validation_features = collect_features(model, loaders["validation"], device)
    reference_features /= np.linalg.norm(reference_features, axis=1, keepdims=True)
    validation_features /= np.linalg.norm(validation_features, axis=1, keepdims=True)
    validation_similarity = validation_features @ reference_features.T
    ood_similarity_threshold = float(np.percentile(validation_similarity.max(axis=1), 5))
    risk_threshold, _ = choose_risk_threshold(
        validation_targets, validation_probabilities, classes
    )
    checkpoint["risk_threshold"] = risk_threshold
    checkpoint["uncertainty_margin"] = 0.03
    checkpoint["ood_similarity_threshold"] = ood_similarity_threshold
    checkpoint["reference_features"] = reference_features.astype(np.float32)
    checkpoint["model_version"] = "resnet18-isic-baseline-v2"
    torch.save(checkpoint, checkpoint_path)

    targets, probabilities = collect_probabilities(model, loaders["test"], device)
    predictions = probabilities.argmax(axis=1).tolist()
    report = classification_report(
        targets, predictions, target_names=classes, output_dict=True, zero_division=0
    )
    malignant_indices = [
        index for index, label in enumerate(classes) if label in {"mel", "bcc", "akiec"}
    ]
    malignant_probability = probabilities[:, malignant_indices].sum(axis=1)
    malignant_targets = np.asarray(
        [classes[target] in {"mel", "bcc", "akiec"} for target in targets], dtype=int
    )
    malignant_predictions = (malignant_probability >= risk_threshold).astype(int)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "model_version": checkpoint["model_version"],
        "risk_threshold": risk_threshold,
        "risk_uncertainty_margin": checkpoint["uncertainty_margin"],
        "malignant_risk_f2": fbeta_score(
            malignant_targets, malignant_predictions, beta=2, zero_division=0
        ),
        "malignant_risk_recall": recall_score(
            malignant_targets, malignant_predictions, zero_division=0
        ),
        "malignant_risk_precision": precision_score(
            malignant_targets, malignant_predictions, zero_division=0
        ),
        "malignant_risk_specificity": recall_score(
            1 - malignant_targets, 1 - malignant_predictions, zero_division=0
        ),
        "malignant_risk_roc_auc": roc_auc_score(
            malignant_targets, malignant_probability
        ),
        "classification_report": report,
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_matrix(
        targets, predictions, classes, output_dir / "confusion_matrix.png"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {checkpoint_path}")


if __name__ == "__main__":
    main()