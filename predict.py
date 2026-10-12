"""Predict the top disease classes for one skin-lesion image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet18
from torchvision.transforms import v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    transform = v2.Compose(
        [
            v2.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(image).softmax(dim=1)[0]
    values, indices = probabilities.topk(min(args.top_k, len(classes)))
    for probability, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        print(f"{classes[index]}: {probability:.2%}")
    print("This output is for research/education and is not a medical diagnosis.")


if __name__ == "__main__":
    main()