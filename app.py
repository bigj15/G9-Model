"""Streamlit demo: 7-class skin-lesion classifier (HAM10000 / ISIC 2018).

Loads a trained checkpoint from a private Hugging Face model repo and classifies
an uploaded dermoscopic image using the CNN head (ResNet-50 encoder + linear head).

RESEARCH DEMO ONLY. Not a medical device. Not for diagnosis.
Training code: https://github.com/bigj15/GNAL-Skin-Lesion-Diagnosis-G9
"""

from __future__ import annotations

import gc
import os

import altair as alt
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO_ID = "bigj15/gnal-training-checkpoints"
HF_REPO_TYPE = "model"

# Index order 0-6 must match training.
CLASSES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
CLASS_NAMES = {
    "AKIEC": "Actinic keratosis / intraepithelial carcinoma",
    "BCC": "Basal cell carcinoma",
    "BKL": "Benign keratosis-like lesion",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "VASC": "Vascular lesion",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LOW_CONFIDENCE = 0.5

MODELS = {
    "CNN-only ResNet-50 (Experiment 2) — default": {
        "filename": "pretrained-v1-cnnonly-fold0-20pct/checkpoints/epoch_019.pt",
        "note": (
            "CNN-only ResNet-50. Held-out test: **78.25% accuracy, 55.02% macro F1**, "
            "measured with the same CNN-head prediction path this app uses."
        ),
    },
    "Full GNAL pretrained_v1 (Experiment 1) — CNN head only": {
        "filename": "pretrained-v1-fold0-20pct/checkpoints/epoch_014.pt",
        "note": (
            "Full GNAL model. Its reported 77.16% accuracy / 56.46% macro F1 came from "
            "**graph** inference. **CNN head only; may differ from reported scores.**"
        ),
    },
}


# ---------------------------------------------------------------------------
# Model (attribute names mirror the training module so state_dict keys match)
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        m = torchvision.models.resnet50(weights=None)  # weights come from the checkpoint
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNClassifier(nn.Module):
    """GNAL without the graph layers (g1/g2): encoder -> linear head."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def get_hf_token() -> str | None:
    """Token from st.secrets, falling back to the HF_TOKEN env var. Never logged."""
    try:
        token = st.secrets["HF_TOKEN"]
        if token:
            return str(token)
    except Exception:  # no secrets.toml, or key missing
        pass
    return os.environ.get("HF_TOKEN") or None


@st.cache_resource(show_spinner=False)
def load_model(filename: str) -> dict:
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type=HF_REPO_TYPE,
        token=get_hf_token(),
    )

    # weights_only=False unpickles arbitrary objects. This is acceptable ONLY because
    # these are the author's own private training checkpoints (they include config,
    # optimizer/scheduler/scaler/RNG state). Never do this with untrusted files.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    config = dict(ckpt.get("config") or {})
    epoch = ckpt.get("epoch")
    ckpt_classes = ckpt.get("classes")
    full_state = ckpt["model"]

    if ckpt_classes is not None:
        normalized = [str(c).upper() for c in ckpt_classes]
        if normalized != CLASSES:
            raise ValueError(
                f"Checkpoint class order {list(ckpt_classes)} does not match expected {CLASSES}."
            )

    state = {
        k: v for k, v in full_state.items() if k.startswith("encoder.") or k.startswith("head.")
    }
    # Free the rest of the checkpoint (optimizer, scheduler, graph layers, ...).
    del ckpt, full_state
    gc.collect()

    model = CNNClassifier(num_classes=len(CLASSES))
    model.load_state_dict(state, strict=True)  # any key mismatch fails loudly
    model.eval()
    del state
    gc.collect()

    image_size = int(config["image_size"])
    steps = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if config.get("imagenet_normalize"):
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return {
        "model": model,
        "transform": transforms.Compose(steps),
        "epoch": epoch,
        "ablation": config.get("ablation", "n/a"),
        "image_size": image_size,
        "imagenet_normalize": bool(config.get("imagenet_normalize")),
    }


def predict(bundle: dict, image: Image.Image) -> pd.DataFrame:
    model = bundle["model"]
    model.eval()
    x = bundle["transform"](image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    df = pd.DataFrame(
        {
            "Code": CLASSES,
            "Class": [CLASS_NAMES[c] for c in CLASSES],
            "Probability": probs,
        }
    )
    return df.sort_values("Probability", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Skin Lesion Classifier (Research Demo)", layout="wide")

st.title("Skin Lesion Classifier — Research Demo")
st.error(
    "**Research demo only. This is not a medical device and must not be used for "
    "diagnosis or treatment decisions.** If you are concerned about a skin lesion, "
    "see a qualified clinician."
)

with st.sidebar:
    st.header("Model")
    choice = st.selectbox("Checkpoint", list(MODELS.keys()), index=0)
    spec = MODELS[choice]
    st.markdown(spec["note"])
    st.caption(f"`{HF_REPO_ID}` / `{spec['filename']}`")

    st.divider()
    st.subheader("Known limits")
    st.markdown(
        "- Weak on **dermatofibroma (DF)**.\n"
        "- Often confuses **melanoma (MEL)** with **nevi (NV)**.\n"
        "- Probabilities are **not well calibrated**; treat them as relative scores."
    )

with st.spinner("Loading model (first load downloads ~285 MB)…"):
    try:
        bundle = load_model(spec["filename"])
    except Exception as exc:  # show a clear error; never echo the token
        st.error(
            "Could not load the model. Check that `HF_TOKEN` is set (Streamlit secrets "
            "or environment) and has read access to the private repo "
            f"`{HF_REPO_ID}`.\n\n**Error:** {type(exc).__name__}: {exc}"
        )
        st.stop()

with st.sidebar:
    epoch = bundle["epoch"]
    st.divider()
    st.markdown(f"**Loaded epoch:** {epoch + 1 if isinstance(epoch, int) else 'unknown'}")
    st.markdown(f"**Ablation:** `{bundle['ablation']}`")
    st.caption(
        f"Input {bundle['image_size']}×{bundle['image_size']}, "
        f"ImageNet normalization: {'on' if bundle['imagenet_normalize'] else 'off'}"
    )

uploaded = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a JPG or PNG dermoscopic image to get a prediction.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Could not read the image: {exc}")
    st.stop()

results = predict(bundle, image)
top = results.iloc[0]

col_img, col_res = st.columns([1, 1.4])
with col_img:
    st.image(image, caption=uploaded.name, use_container_width=True)

with col_res:
    st.subheader("Prediction")
    st.metric(
        label=f"{top['Code']} — {top['Class']}",
        value=f"{top['Probability']:.1%}",
    )
    if top["Probability"] < LOW_CONFIDENCE:
        st.warning(
            f"Low confidence: the top probability is below {LOW_CONFIDENCE:.0%}. "
            "The model is uncertain about this image."
        )

    chart = (
        alt.Chart(results)
        .mark_bar()
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Code:N", sort="-x", title=None),
            tooltip=["Code", "Class", alt.Tooltip("Probability:Q", format=".2%")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

    table = results.copy()
    table["Probability"] = table["Probability"].map(lambda p: f"{p:.2%}")
    st.dataframe(table, hide_index=True, use_container_width=True)

st.caption("Research demo only — not a medical device, not for diagnosis.")
