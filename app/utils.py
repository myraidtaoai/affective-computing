"""
Shared utilities for the Affective Computing Streamlit app.

Data loading, embedding extraction, and split logic — mirrors the
classification_extra_data.ipynb notebook exactly.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import streamlit as st

# ── Paths (relative to repo root) ────────────────────────────────────────────
REPO_ROOT     = Path(__file__).parent.parent
DATA_ROOT     = REPO_ROOT / "data"
FER_ROOT      = DATA_ROOT / "FER-2013"
TRAINING_SET  = DATA_ROOT / "training_set"
ANN_PATH      = TRAINING_SET / "annotations.csv"
CACHE_DIR     = REPO_ROOT / "app" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLIP_CACHE    = CACHE_DIR / "clip_embeddings.npz"
SIG_CACHE     = CACHE_DIR / "siglip2_embeddings.npz"

SEED = 42
LABEL_MAP = {"neutral": 0, "happy": 1}
IDX_TO_LABEL = {0: "neutral", 1: "happy"}


# ── Data loading ─────────────────────────────────────────────────────────────

def normalize_label(raw):
    raw = str(raw).strip().lower()
    if raw in ("happy", "happiness", "1"):
        return "happy"
    if raw in ("neutral", "0"):
        return "neutral"
    return None


def collect_fer2013_df():
    """Collect all FER-2013 image paths with labels (neutral + happy only)."""
    rows = []
    for split in ("train", "test"):
        for label_dir in (FER_ROOT / split).iterdir():
            norm = normalize_label(label_dir.name)
            if norm is None:
                continue
            for img_path in label_dir.glob("*.jpg"):
                rows.append({"filepath": str(img_path), "label": norm, "source": "FER-2013"})
            for img_path in label_dir.glob("*.png"):
                rows.append({"filepath": str(img_path), "label": norm, "source": "FER-2013"})
    return pd.DataFrame(rows)


def collect_training_set_df():
    """Collect training_set images via annotations.csv."""
    df = pd.read_csv(ANN_PATH, header=None, names=["filename", "label"])
    actual = {f.lower(): f for f in os.listdir(TRAINING_SET) if f.lower().endswith(".jpg")}
    df["resolved"] = df["filename"].apply(lambda x: actual.get(x.lower()))
    df = df[df["resolved"].notna()].copy()
    df["filepath"] = df["resolved"].apply(lambda f: str(TRAINING_SET / f))
    df["source"] = "training_set"
    return df[["filepath", "label", "source"]]


@st.cache_data(show_spinner="Loading datasets…")
def load_combined_df(fer_sample_frac=0.20):
    """Combine FER-2013 (downsampled) + training_set, return combined + splits."""
    from sklearn.model_selection import train_test_split

    fer_df = collect_fer2013_df()
    fer_small = fer_df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(frac=fer_sample_frac, random_state=SEED)
    ).reset_index(drop=True)

    ts_df = collect_training_set_df()
    combined = pd.concat([fer_small, ts_df], ignore_index=True)
    combined["label_idx"] = combined["label"].map(LABEL_MAP)

    train_df, temp_df = train_test_split(combined, test_size=0.30, stratify=combined["label_idx"], random_state=SEED)
    val_df,  test_df  = train_test_split(temp_df,  test_size=0.50, stratify=temp_df["label_idx"],  random_state=SEED)

    return (
        combined.reset_index(drop=True),
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ── Embedding models ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading CLIP model…")
def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return proc, model


@st.cache_resource(show_spinner="Loading SigLIP2 model…")
def load_siglip():
    from transformers import AutoModel, AutoProcessor
    proc  = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
    model.eval()
    return proc, model


def extract_embeddings(df, processor, model, model_type, device=None, batch_size=32):
    """
    Extract L2-normalized embeddings for all images in *df*.

    Parameters
    ----------
    model_type : "clip" | "siglip"

    Returns
    -------
    embs   : np.ndarray (n, dim)
    labels : np.ndarray (n,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_embs, all_labels = [], []

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            images = []
            for _, row in batch.iterrows():
                try:
                    img = Image.open(row["filepath"]).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224))
                images.append(img)

            inputs = processor(images=images, return_tensors="pt").to(device)
            if model_type == "clip":
                embs = model.get_image_features(**inputs)
            else:
                embs = model.vision_model(**inputs).pooler_output

            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().float().numpy())
            all_labels.extend(batch["label_idx"].tolist())

    return np.vstack(all_embs), np.array(all_labels)


def get_or_compute_embeddings(train_df, val_df, test_df):
    """
    Load embeddings from .npz cache files, or compute and save them.
    Returns dict with keys clip/siglip2, each containing X_train/val/test + y_train/y_test.
    """
    result = {}

    for name, cache_path, model_type, loader_fn in [
        ("clip",    CLIP_CACHE, "clip",   load_clip),
        ("siglip2", SIG_CACHE,  "siglip", load_siglip),
    ]:
        if cache_path.exists():
            data = np.load(cache_path)
            result[name] = {
                "X_train": data["X_train"],
                "X_val":   data["X_val"],
                "X_test":  data["X_test"],
                "y_train": data["y_train"],
                "y_test":  data["y_test"],
            }
        else:
            proc, model = loader_fn()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_train, y_train = extract_embeddings(train_df, proc, model, model_type, device)
            X_val,   _       = extract_embeddings(val_df,   proc, model, model_type, device)
            X_test,  y_test  = extract_embeddings(test_df,  proc, model, model_type, device)
            np.savez_compressed(
                cache_path,
                X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_test=y_test,
            )
            result[name] = {
                "X_train": X_train, "X_val": X_val, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
            }

    return result


# ── Classifiers definition ────────────────────────────────────────────────────

def get_classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    return {
        "LogReg (linear probe)": LogisticRegression(C=0.1, max_iter=1000, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED, verbosity=0,
        ),
        "Deep Probe (MLP)": MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), activation="relu",
            solver="adam", batch_size=32, max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            random_state=SEED,
        ),
    }


EMBEDDING_NAMES  = ["CLIP", "SigLIP2"]
CLASSIFIER_NAMES = ["LogReg (linear probe)", "XGBoost", "Deep Probe (MLP)"]
COMBO_KEYS       = [f"{e} + {c}" for e in EMBEDDING_NAMES for c in CLASSIFIER_NAMES]

EMB_KEY_MAP = {"CLIP": "clip", "SigLIP2": "siglip2"}  # display → cache key
