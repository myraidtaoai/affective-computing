"""
Page 4 — Model Inference
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.manifold import TSNE
import umap as umap_lib

from app.utils import (
    load_combined_df, get_or_compute_embeddings,
    load_clip, load_siglip, get_classifiers,
    EMBEDDING_NAMES, CLASSIFIER_NAMES, EMB_KEY_MAP,
    IDX_TO_LABEL, SEED,
)

st.set_page_config(page_title="Model Inference", page_icon="🔮", layout="wide")
st.title("🔮 Model Inference")

st.markdown(
    """
Upload any face image to get a live **neutral / happy** prediction.
The app extracts an embedding using the selected foundation model,
then passes it through the trained classifier.
"""
)

# ── Load data & embeddings ────────────────────────────────────────────────────
if "combined_df" not in st.session_state:
    combined_df, train_df, val_df, test_df = load_combined_df()
    st.session_state.update({
        "combined_df": combined_df,
        "train_df": train_df,
        "val_df":   val_df,
        "test_df":  test_df,
    })

train_df = st.session_state["train_df"]
val_df   = st.session_state["val_df"]
test_df  = st.session_state["test_df"]

if "emb_data" not in st.session_state:
    if st.button("Load / Compute Background Embeddings"):
        with st.spinner("Loading embeddings…"):
            emb_data = get_or_compute_embeddings(train_df, val_df, test_df)
            st.session_state["emb_data"] = emb_data
        st.success("Background embeddings ready.")
    else:
        st.info("Load background embeddings first to enable the 2D projection view.")

emb_data = st.session_state.get("emb_data")

# ── Section 1: Upload & settings ─────────────────────────────────────────────
st.header("1. Upload Image")

col_up, col_cfg = st.columns([1, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "Upload a face image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

with col_cfg:
    emb_choice = st.radio(
        "Embedding model",
        ["CLIP (512-d)", "SigLIP2 (768-d)"],
        help="Select which foundation model extracts the image embedding.",
    )
    emb_name   = "CLIP"    if "CLIP" in emb_choice else "SigLIP2"
    emb_key    = EMB_KEY_MAP[emb_name]

    clf_choice = st.selectbox(
        "Classifier",
        CLASSIFIER_NAMES,
        index=2,  # default: Deep Probe (MLP)
    )

    run_button = st.button("Run Inference", type="primary", disabled=(uploaded_file is None))

# ── Section 2: Inference ──────────────────────────────────────────────────────
st.header("2. Prediction")

if run_button and uploaded_file is not None:
    # ── Extract embedding ─────────────────────────────────────────────────────
    with st.spinner(f"Extracting {emb_name} embedding…"):
        if emb_name == "CLIP":
            proc, model = load_clip()
        else:
            proc, model = load_siglip()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)

        with torch.no_grad():
            inputs = proc(images=img, return_tensors="pt").to(device)
            if emb_name == "CLIP":
                emb = model.get_image_features(**inputs)
            else:
                emb = model.vision_model(**inputs).pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_np = emb.cpu().float().numpy()  # (1, dim)

    # ── Load or train the chosen classifier ───────────────────────────────────
    if "trained_clfs" in st.session_state:
        key = f"{emb_name} + {clf_choice}"
        if key in st.session_state["trained_clfs"]:
            clf_fit, _, _ = st.session_state["trained_clfs"][key]
        else:
            clf_fit = None
    else:
        clf_fit = None

    if clf_fit is None:
        st.info(f"Classifier not found in session. Training {emb_name} + {clf_choice} now…")
        if emb_data is not None:
            with st.spinner("Training classifier…"):
                classifiers = get_classifiers()
                clf_fit = clone(classifiers[clf_choice])
                X_tr = emb_data[emb_key]["X_train"]
                y_tr = emb_data[emb_key]["y_train"]
                clf_fit.fit(X_tr, y_tr)
        else:
            st.error("Background embeddings not loaded. Cannot train classifier. Load embeddings first.")
            st.stop()

    # ── Predict ───────────────────────────────────────────────────────────────
    pred_idx  = int(clf_fit.predict(emb_np)[0])
    pred_label = IDX_TO_LABEL[pred_idx]

    if hasattr(clf_fit, "predict_proba"):
        proba = clf_fit.predict_proba(emb_np)[0]
    elif hasattr(clf_fit, "decision_function"):
        raw = clf_fit.decision_function(emb_np)[0]
        p1  = 1 / (1 + np.exp(-raw))
        proba = [1 - p1, p1]
    else:
        proba = [1.0, 0.0] if pred_idx == 0 else [0.0, 1.0]

    # ── Display result ────────────────────────────────────────────────────────
    col_res, col_conf = st.columns(2)

    emoji = "😐" if pred_label == "neutral" else "😊"
    col_res.markdown(f"### {emoji} Prediction: **{pred_label.upper()}**")
    col_res.markdown(f"Using: `{emb_name}` + `{clf_choice}`")

    col_conf.markdown("#### Confidence")
    for label, prob in zip(["neutral", "happy"], proba):
        col_conf.markdown(f"**{label}**: {prob:.1%}")
        col_conf.progress(float(prob))

    # Store for projection
    st.session_state["infer_emb_np"]  = emb_np
    st.session_state["infer_emb_key"] = emb_key
    st.session_state["infer_label"]   = pred_label

# ── Section 3: Embedding projection ──────────────────────────────────────────
st.header("3. Embedding in 2D Space")

if emb_data is None:
    st.info("Load background embeddings (button at top of page) to see the 2D projection.")
elif "infer_emb_np" not in st.session_state:
    st.info("Run inference above to see where the uploaded image falls in embedding space.")
else:
    infer_emb  = st.session_state["infer_emb_np"]
    infer_key  = st.session_state["infer_emb_key"]
    infer_lbl  = st.session_state["infer_label"]

    proj_method = st.radio("Projection method", ["t-SNE", "UMAP"], horizontal=True, key="proj_infer")

    X_all = np.vstack([
        emb_data[infer_key]["X_train"],
        emb_data[infer_key]["X_val"],
        emb_data[infer_key]["X_test"],
        infer_emb,
    ])
    # Derive labels for all splits
    val_y_vals = val_df["label_idx"].values
    all_labels = np.concatenate([
        emb_data[infer_key]["y_train"],
        val_y_vals,
        emb_data[infer_key]["y_test"],
        [-1],  # uploaded image — marked separately
    ])
    all_label_names = [IDX_TO_LABEL.get(int(l), "uploaded") for l in all_labels]

    @st.cache_data(show_spinner="Computing projection…")
    def project(X, method, _key):
        if method == "t-SNE":
            return TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED).fit_transform(X)
        return umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(X)

    # Use a cache key that changes when infer_emb changes
    cache_key = hash(infer_emb.tobytes())
    proj = project(X_all, proj_method, cache_key)

    plot_df = pd.DataFrame({
        "x": proj[:, 0], "y": proj[:, 1],
        "Emotion": all_label_names,
    })
    # Mark uploaded image point
    plot_df.loc[plot_df["Emotion"] == "uploaded", "Emotion"] = f"uploaded ({infer_lbl})"

    color_map = {
        "neutral":                "#4C72B0",
        "happy":                  "#DD8452",
        f"uploaded ({infer_lbl})": "#FF0000",
    }
    size_map = [6 if e == f"uploaded ({infer_lbl})" else 4 for e in plot_df["Emotion"]]

    fig_proj = px.scatter(
        plot_df, x="x", y="y", color="Emotion",
        color_discrete_map=color_map,
        opacity=0.7,
        title=f"{proj_method} Projection — {infer_key.upper()} Embeddings",
        labels={"x": "dim 1", "y": "dim 2"},
    )
    # Make uploaded point larger
    fig_proj.update_traces(marker_size=4)
    uploaded_mask = plot_df["Emotion"] == f"uploaded ({infer_lbl})"
    fig_proj.add_scatter(
        x=proj[uploaded_mask, 0],
        y=proj[uploaded_mask, 1],
        mode="markers",
        marker=dict(size=16, color="red", symbol="star", line=dict(color="black", width=1)),
        name=f"Uploaded ({infer_lbl})",
    )
    st.plotly_chart(fig_proj, use_container_width=True)
    st.caption("Red star = uploaded image. Blue = neutral, orange = happy (background dataset).")
