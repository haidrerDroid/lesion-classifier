"""Local Streamlit interface for the trained skin-lesion classifier."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, ImageStat
from torchvision.models import efficientnet_b0, resnet18
from torchvision.transforms import v2


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "ensemble_model.pt"
FALLBACK_MODEL_PATH = ARTIFACT_DIR / "best_model.pt"
METRICS_PATH = ARTIFACT_DIR / "ensemble_metrics.json"
FALLBACK_METRICS_PATH = ARTIFACT_DIR / "test_metrics.json"
MALIGNANT_LABELS = {"mel", "bcc", "akiec"}
MIN_CLASS_CONFIDENCE = 0.70
DISPLAY_NAMES = {
    "akiec": "Actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "vasc": "Vascular lesion",
}


@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    models = []
    if "model_states" in checkpoint:
        for model_index, state in enumerate(checkpoint["model_states"]):
            if model_index == 0:
                model = resnet18(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
            else:
                model = efficientnet_b0(weights=None)
                model.classifier[1] = torch.nn.Linear(
                    model.classifier[1].in_features, len(checkpoint["classes"])
                )
            model.load_state_dict(state)
            models.append(model.to(device).eval())
    else:
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
        model.load_state_dict(checkpoint["model_state"])
        models.append(model.to(device).eval())
    transform = v2.Compose(
        [
            v2.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return (
        models,
        transform,
        checkpoint["classes"],
        device,
        checkpoint.get("risk_threshold", 0.5),
        checkpoint.get("uncertainty_margin", 0.10),
        checkpoint.get("reference_features"),
        checkpoint.get("ood_similarity_threshold", 0.0),
        model_path.name,
    )


def predict(image: Image.Image):
    (
        models,
        transform,
        classes,
        device,
        risk_threshold,
        uncertainty_margin,
        reference_features,
        ood_similarity_threshold,
        model_name,
    ) = load_model()
    original = image.convert("RGB")
    views = [original, ImageOps.mirror(original), ImageOps.flip(original)]
    tensor = torch.stack([transform(view) for view in views]).to(device)
    with torch.inference_mode():
        model_probabilities = [model(tensor).softmax(dim=1).cpu() for model in models]
        view_probabilities = torch.stack(model_probabilities).mean(dim=0)
        view_features = []
        for model in models:
            if hasattr(model, "fc"):
                features = model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
                features = model.layer1(features)
                features = model.layer2(features)
                features = model.layer3(features)
                features = model.layer4(features)
                features = model.avgpool(features)
            else:
                features = model.avgpool(model.features(tensor))
            view_features.append(torch.flatten(features, 1).cpu())
    probabilities = view_probabilities.mean(dim=0)
    view_predictions = view_probabilities.argmax(dim=1).tolist()
    consistency = sum(
        prediction == view_predictions[0] for prediction in view_predictions
    ) / len(view_predictions)
    if reference_features is not None:
        similarities = []
        for features, references in zip(view_features, reference_features if isinstance(reference_features, list) else [reference_features]):
            normalized_features = features / features.norm(dim=1, keepdim=True)
            reference_tensor = torch.as_tensor(references)
            similarities.append((normalized_features @ reference_tensor.T).max(dim=1).values)
        similarity = torch.stack(similarities).mean().item()
        is_supported = similarity >= ood_similarity_threshold
    else:
        similarity = 1.0
        is_supported = True
    scores = {label: float(probabilities[index]) for index, label in enumerate(classes)}
    return (
        scores,
        risk_threshold,
        uncertainty_margin,
        consistency,
        similarity,
        is_supported,
        model_name,
    )


def image_quality_warning(image: Image.Image) -> str | None:
    width, height = image.size
    if min(width, height) < 128:
        return "This image is very small. Use a clearer, higher-resolution image."
    grayscale = image.convert("L")
    if ImageStat.Stat(grayscale).stddev[0] < 12:
        return "This image has very low contrast. Try a better-lit image."
    return None


st.set_page_config(
    page_title="Lesion Lens",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    :root {
        --ink: #1d2a30;
        --muted: #5d6d76;
        --paper: #f5f1ea;
        --panel: rgba(255,255,255,0.76);
        --line: rgba(30, 58, 74, 0.09);
        --mint: #d9efe7;
        --mint-strong: #5aa67b;
        --coral: #f07863;
        --coral-soft: #fff0ec;
        --amber: #f8d39b;
        --shadow: 0 12px 28px rgba(21, 46, 52, 0.08);
    }
    .stApp { background: radial-gradient(circle at top left, #fef7f4 0%, var(--paper) 34%, #f1f6f4 100%); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1200px; padding-top: 2.8rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.03em; }
    p, label, div, button, span { font-family: 'DM Sans', sans-serif; }
    .eyebrow {
        color: var(--coral);
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .5rem;
    }
    .hero-panel {
        background: linear-gradient(135deg, rgba(255,255,255,0.68), rgba(217,239,231,0.55));
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.6rem 1.7rem 1.2rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.4rem;
    }
    .hero-title { font-size: clamp(2.1rem, 5vw, 3.1rem); margin: 0; line-height: 1.08; }
    .intro {
        color: var(--muted);
        font-size: 1.06rem;
        max-width: 760px;
        line-height: 1.6;
        margin: 0.7rem 0 0;
    }
    .panel {
        background: rgba(255,255,255,0.74);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.25rem;
        box-shadow: var(--shadow);
    }
    .panel h3 { margin-top: 0; margin-bottom: 0.6rem; }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(217,239,231,0.45));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.8rem 0.9rem;
        height: 100%;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        margin-top: .25rem;
    }
    .result {
        padding: 1.25rem 1.3rem;
        border-radius: 18px;
        border: 1px solid rgba(240,120,99,0.18);
        background: linear-gradient(135deg, var(--coral-soft), rgba(255,255,255,0.9));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
    }
    .result.safe {
        border-color: rgba(90,166,123,0.22);
        background: linear-gradient(135deg, rgba(217,239,231,0.8), rgba(255,255,255,0.9));
    }
    .result-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0;
    }
    .result-copy { color: var(--muted); margin-top: .35rem; line-height: 1.5; }
    .status-pill {
        display: inline-block;
        padding: .35rem .75rem;
        border-radius: 999px;
        background: rgba(240,120,99,0.1);
        color: #9d4c3b;
        font-weight: 700;
        font-size: .76rem;
        letter-spacing: .06em;
        text-transform: uppercase;
    }
    .status-pill.safe {
        background: rgba(90,166,123,0.12);
        color: #2e6d4d;
    }
    .note { color: var(--muted); font-size: .88rem; line-height: 1.6; }
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(17, 75, 95, 0.28);
        border-radius: 18px;
        background: rgba(217,239,231,0.22);
        padding: .75rem;
    }
    [data-testid="stImage"] > img { border-radius: 18px; }
    .stButton > button {
        border-radius: 999px;
        background: linear-gradient(90deg, var(--coral), #ea8d62);
        color: white;
        border: none;
        font-weight: 700;
        padding: .7rem 1.2rem;
    }
    .stButton > button:hover { filter: brightness(0.97); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">ISIC lesion classifier</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-panel">
        <h1 class="hero-title">Lesion Lens</h1>
        <p class="intro">Upload or capture a skin-lesion image to see the model’s predicted category and its estimated malignant-risk pattern.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

if not MODEL_PATH.exists() and not FALLBACK_MODEL_PATH.exists():
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.stop()

left, right = st.columns([1.05, 1.15], gap="large")
with left:
    with st.container():
        st.markdown('<div class="panel"><h3>1. Add an image</h3></div>', unsafe_allow_html=True)
        input_method = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
        if input_method == "Upload":
            image_file = st.file_uploader(
                "Choose a JPG, JPEG, or PNG image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key=f"lesion_uploader_{st.session_state.uploader_version}",
            )
        else:
            image_file = st.camera_input(
                "Take a photo",
                key=f"lesion_camera_{st.session_state.uploader_version}",
            )

        if image_file:
            image = Image.open(image_file).convert("RGB")
            caption = image_file.name if input_method == "Upload" else "Camera capture"
            st.image(image, caption=caption, use_container_width=True)
            quality_warning = image_quality_warning(image)
            if quality_warning:
                st.warning(quality_warning)

with right:
    st.markdown('<div class="panel"><h3>2. Review prediction</h3></div>', unsafe_allow_html=True)
    if not image_file:
        st.info("Your result will appear here after you add an image.")
    else:
        with st.spinner("Analyzing image..."):
            (
                scores,
                risk_threshold,
                uncertainty_margin,
                consistency,
                similarity,
                is_supported,
                model_name,
            ) = predict(image)
        top_label = max(scores, key=scores.get)
        top_confidence = scores[top_label]
        malignant_score = sum(
            score for label, score in scores.items() if label in MALIGNANT_LABELS
        )
        distance_from_threshold = abs(malignant_score - risk_threshold)
        is_uncertain = (
            distance_from_threshold < uncertainty_margin
            or top_confidence < MIN_CLASS_CONFIDENCE
            or consistency < 1.0
            or not is_supported
        )
        is_malignant = malignant_score >= risk_threshold
        risk_status = (
            "Uncertain"
            if is_uncertain
            else "Higher risk"
            if is_malignant
            else "Lower risk"
        )
        result_class = " safe" if not is_malignant and not is_uncertain else ""
        pill_class = " safe" if not is_malignant and not is_uncertain else ""
        if not is_supported:
            headline = "Not one of the six supported cases"
            result_copy = (
                "This image is too different from the lesion images used for training."
            )
        elif is_uncertain:
            headline = "Unfamiliar or uncertain image"
            result_copy = (
                "The model is not confident enough to assign a reliable category."
            )
        elif is_malignant:
            headline = "Higher-risk pattern detected"
            result_copy = "The model estimates a higher malignant-risk pattern."
        else:
            headline = "Lower-risk pattern detected"
            result_copy = "The model estimates a lower malignant-risk pattern."
        st.markdown(
            f'<div class="result{result_class}"><div style="display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:.5rem;">'
            f'<div class="result-title">{headline}</div>'
            f'<span class="status-pill{pill_class}">{risk_status}</span>'
            f'</div>'
            f'<div class="result-copy">{result_copy} '
            f'Best available category: <strong>{DISPLAY_NAMES.get(top_label, top_label)}</strong> '
            f'({top_confidence:.1%})</div></div>',
            unsafe_allow_html=True,
        )

        metric_col_1, metric_col_2 = st.columns(2)
        with metric_col_1:
            st.markdown(
                f'<div class="metric-card"><div style="color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.09em;">Malignant risk</div><div class="metric-value">{malignant_score:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        with metric_col_2:
            st.markdown(
                f'<div class="metric-card"><div style="color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.09em;">Top category</div><div class="metric-value">{DISPLAY_NAMES.get(top_label, top_label)}</div></div>',
                unsafe_allow_html=True,
            )

        st.caption(
            "Lower risk is a model estimate, not confirmation that the image is safe or cancer-free."
        )
        st.caption(
            f"Calibrated threshold: {risk_threshold:.1%} · View consistency: {consistency:.0%} · Lesion similarity: {similarity:.1%}."
        )
        st.caption(f"Model used: {model_name}")
        st.subheader("Class probabilities")
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            st.write(f"**{DISPLAY_NAMES.get(label, label)}** · {score:.1%}")
            st.progress(score)

        if st.button("Upload another image", type="secondary"):
            st.session_state.uploader_version += 1
            st.rerun()

st.divider()
metrics_path = METRICS_PATH if METRICS_PATH.exists() else FALLBACK_METRICS_PATH
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    st.caption(
        f"Model: {metrics.get('model_version', 'baseline')} · Test macro-F1: {metrics.get('macro_f1', 0):.1%} · Test balanced accuracy: {metrics.get('balanced_accuracy', 0):.1%}"
    )
st.markdown(
    '<p class="note"><strong>Important:</strong> This research prototype cannot confirm or rule out cancer and is not a medical diagnosis. Model predictions can be wrong, especially for images that differ from the training data. Consult a qualified clinician for any concerning lesion or change.</p>',
    unsafe_allow_html=True,
)