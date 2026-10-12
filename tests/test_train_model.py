from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from train_model import (
    choose_risk_threshold,
    load_records,
    make_splits,
    save_confusion_matrix,
)


def write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color="red").save(path)


def test_load_records_deduplicates_drops_unlabeled_and_matches_image_extensions(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "Images"
    image_dir.mkdir()
    write_image(image_dir / "image-1.JPG")
    write_image(image_dir / "image-3.png")
    pd.DataFrame(
        [
            {"image_id": "image-1", "dx": "mel", "lesion_id": "lesion-1"},
            {"image_id": "image-1", "dx": "mel", "lesion_id": "lesion-1"},
            {"image_id": "image-2", "dx": None, "lesion_id": "lesion-2"},
            {"image_id": "image-3", "dx": "bcc", "lesion_id": "lesion-3"},
        ]
    ).to_csv(tmp_path / "metadata.csv", index=False)

    records = load_records(tmp_path)

    assert records["image_id"].tolist() == ["image-1", "image-3"]
    assert records["path"].tolist() == [
        image_dir / "image-1.JPG",
        image_dir / "image-3.png",
    ]


def test_load_records_rejects_conflicting_labels(tmp_path: Path) -> None:
    (tmp_path / "Images").mkdir()
    pd.DataFrame(
        [
            {"image_id": "duplicate", "dx": "mel", "lesion_id": "lesion-1"},
            {"image_id": "duplicate", "dx": "bcc", "lesion_id": "lesion-1"},
        ]
    ).to_csv(tmp_path / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="conflicting labels"):
        load_records(tmp_path)


def test_load_records_reports_missing_images(tmp_path: Path) -> None:
    (tmp_path / "Images").mkdir()
    pd.DataFrame(
        [{"image_id": "missing", "dx": "mel", "lesion_id": "lesion-1"}]
    ).to_csv(tmp_path / "metadata.csv", index=False)

    with pytest.raises(FileNotFoundError, match="Missing 1 image files"):
        load_records(tmp_path)


def test_make_splits_are_reproducible_and_keep_lesions_together() -> None:
    records = pd.DataFrame(
        [
            {
                "image_id": f"{label}-{lesion_number}",
                "dx": label,
                "lesion_id": f"{label}-lesion-{lesion_number}",
            }
            for label in ("akiec", "bcc", "bkl", "df", "mel", "vasc")
            for lesion_number in range(12)
        ]
    )

    splits = make_splits(records, seed=17)
    repeated_splits = make_splits(records, seed=17)

    assert set(splits) == {"train", "validation", "test"}
    assert all(not split.empty for split in splits.values())
    assert {
        name: split["image_id"].tolist() for name, split in splits.items()
    } == {
        name: split["image_id"].tolist() for name, split in repeated_splits.items()
    }
    assert set().union(*(set(split["image_id"]) for split in splits.values())) == set(
        records["image_id"]
    )
    group_sets = [set(split["lesion_id"]) for split in splits.values()]
    assert not (group_sets[0] & group_sets[1])
    assert not (group_sets[0] & group_sets[2])
    assert not (group_sets[1] & group_sets[2])


def test_choose_risk_threshold_uses_validation_probabilities() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

    threshold, mean_risk = choose_risk_threshold(
        targets, probabilities, classes=["mel", "bkl"]
    )

    assert threshold == pytest.approx(0.21)
    assert mean_risk == pytest.approx(0.5)


def test_save_confusion_matrix_creates_image(tmp_path: Path) -> None:
    output_path = tmp_path / "confusion_matrix.png"

    save_confusion_matrix(
        targets=[0, 1, 1],
        predictions=[0, 0, 1],
        classes=["mel", "bcc"],
        output_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
