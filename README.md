# Skin Disease Detection Baseline

This project trains a six-class image classifier from an ISIC-style skin-lesion dataset. Dataset files and trained model artifacts are not included in this repository.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset

Obtain the dataset and confirm its terms of use before using or redistributing it. Place `metadata.csv` and the corresponding image files in this local layout:

```text
ISIC-images/
	metadata.csv
	Images/
		<image_id>.jpg
```

The training scripts read from `ISIC-images` by default. Use `--data-dir` to specify a different location.

## Train and evaluate

From the project directory, run:

```bash
python train_model.py --epochs 10
```

The script validates duplicate labels, removes repeated image IDs, keeps images from the same lesion together, creates stratified train/validation/test CSV files, trains a pretrained ResNet18, and uses class-weighted loss for imbalanced labels. It calibrates the malignant-risk threshold on the validation split and evaluates that risk grouping on the untouched test split.

Generated files are placed in `artifacts/`:

- `best_model.pt`: best validation macro-F1 checkpoint
- `test_metrics.json`: balanced accuracy, macro-F1, and per-class metrics
- `confusion_matrix.png`: test-set confusion matrix
- `splits/*.csv`: reproducible data splits

The first run downloads pretrained ResNet18 weights. On an Apple Silicon Mac, the script uses the MPS accelerator when available.

## Predict one image

After training:

```bash
python predict.py ISIC-images/Images/ISIC_0000031.jpg
```

The model predicts `akiec`, `bcc`, `bkl`, `df`, `mel`, or `vasc`. These predictions are for research or education and must not be used as a medical diagnosis.

## Train the ensemble

The project includes a probability ensemble using ResNet18 and EfficientNet-B0. Both models use the same lesion-level splits; their six-class probabilities are averaged before risk and unsupported-image checks.

```bash
python train_ensemble.py --epochs 10 --batch-size 64
```

This creates `artifacts/ensemble_model.pt` and `artifacts/ensemble_metrics.json`. The app automatically prefers this ensemble when the file exists and otherwise falls back to `artifacts/best_model.pt`.

## Run the image interface

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal. The interface accepts an uploaded image or a photo captured with the device camera; camera access requires browser permission. It combines the probabilities of `mel`, `bcc`, and `akiec` into the displayed malignant-risk estimate, uses an uncertainty band around the calibrated threshold, and warns about very small or low-contrast images. This is a research prototype, not a medical diagnostic tool.