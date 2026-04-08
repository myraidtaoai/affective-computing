"""
Page 1 — Data Exploration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from sklearn.manifold import TSNE
import umap

from app.utils import (
    load_combined_df, get_or_compute_embeddings,
    LABEL_MAP, IDX_TO_LABEL, SEED,
)

st.set_page_config(page_title="Data Exploration", page_icon="🔍", layout="wide")
st.title("🔍 Data Exploration")

# ── Load data ─────────────────────────────────────────────────────────────────
if "combined_df" not in st.session_state:
    combined_df, train_df, val_df, test_df = load_combined_df()
    st.session_state["combined_df"] = combined_df
    st.session_state["train_df"]    = train_df
    st.session_state["val_df"]      = val_df
    st.session_state["test_df"]     = test_df
else:
    combined_df = st.session_state["combined_df"]
    train_df    = st.session_state["train_df"]
    val_df      = st.session_state["val_df"]
    test_df     = st.session_state["test_df"]

# ── Section 1: Dataset overview ───────────────────────────────────────────────
st.header("1. Dataset Overview")

col1, col2 = st.columns(2)

# Summary table
summary_rows = []
for source in combined_df["source"].unique():
    sub = combined_df[combined_df["source"] == source]
    summary_rows.append({
        "Source": source,
        "Total": len(sub),
        "Neutral": int((sub["label"] == "neutral").sum()),
        "Happy": int((sub["label"] == "happy").sum()),
    })
summary_rows.append({
    "Source": "**Combined**",
    "Total": len(combined_df),
    "Neutral": int((combined_df["label"] == "neutral").sum()),
    "Happy": int((combined_df["label"] == "happy").sum()),
})
col1.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# Split sizes
split_df = pd.DataFrame({
    "Split": ["Train", "Val", "Test"],
    "Size":  [len(train_df), len(val_df), len(test_df)],
})
fig_split = px.bar(
    split_df, x="Split", y="Size", text_auto=True,
    color="Split",
    title="Train / Val / Test Split Sizes",
)
fig_split.update_layout(showlegend=False, height=300)
col2.plotly_chart(fig_split, use_container_width=True)

# ── Section 2: Sample images ──────────────────────────────────────────────────
st.header("2. Sample Images")

source_choice = st.radio("Dataset source", combined_df["source"].unique(), horizontal=True)
sub = combined_df[combined_df["source"] == source_choice]

random.seed(SEED)
n_show = 5

for emotion in ["neutral", "happy"]:
    st.subheader(f"{emotion.capitalize()}")
    rows = sub[sub["label"] == emotion]
    samples = rows.sample(min(n_show, len(rows)), random_state=SEED)
    cols = st.columns(n_show)
    for col, (_, row) in zip(cols, samples.iterrows()):
        try:
            img = Image.open(row["filepath"]).convert("RGB")
            col.image(img, use_container_width=True, caption=Path(row["filepath"]).name)
        except Exception as e:
            col.warning(f"Cannot load image:\n{e}")

# ── Section 3: Embedding visualizations ───────────────────────────────────────
st.header("3. Embedding Visualizations")

st.info(
    "Embeddings are computed once and cached in `app/.cache/`. "
    "First run may take several minutes."
)

if st.button("Load / Compute Embeddings"):
    with st.spinner("Loading embeddings (this may take a while on first run)…"):
        emb_data = get_or_compute_embeddings(train_df, val_df, test_df)
        st.session_state["emb_data"] = emb_data
    st.success("Embeddings ready.")

if "emb_data" not in st.session_state:
    st.warning("Click the button above to load embeddings before viewing projections.")
    st.stop()

emb_data = st.session_state["emb_data"]

# Combine train+val+test for full-dataset projection
def get_all_embeddings(key):
    d = emb_data[key]
    X = np.vstack([d["X_train"], d["X_val"], d["X_test"]])
    y = np.concatenate([d["y_train"],
                        np.full(len(d["X_val"]),  -1),  # val labels not stored → use train/test
                        d["y_test"]])
    return X, y

# Re-derive all labels from combined_df (same order as extraction)
all_labels = combined_df["label_idx"].values

X_clip_all  = np.vstack([emb_data["clip"]["X_train"],    emb_data["clip"]["X_val"],    emb_data["clip"]["X_test"]])
X_sig_all   = np.vstack([emb_data["siglip2"]["X_train"], emb_data["siglip2"]["X_val"], emb_data["siglip2"]["X_test"]])
# Use only train+test labels (val not saved); re-derive from split dfs
all_y = np.concatenate([
    emb_data["clip"]["y_train"],
    np.full(len(emb_data["clip"]["X_val"]), -1),
    emb_data["clip"]["y_test"],
])
# Replace -1 with val labels derived from val_df
val_y = val_df["label_idx"].values
n_train = len(emb_data["clip"]["y_train"])
n_val   = len(val_y)
all_y[n_train:n_train + n_val] = val_y

label_names = [IDX_TO_LABEL[i] for i in all_y]
COLOR_MAP = {"neutral": "#4C72B0", "happy": "#DD8452"}

viz_type = st.radio("Projection method", ["t-SNE", "UMAP"], horizontal=True)

@st.cache_data(show_spinner="Computing projections…")
def compute_projections(X_clip, X_sig, method):
    if method == "t-SNE":
        proj = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
        return proj.fit_transform(X_clip), proj.fit_transform(X_sig)
    else:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)
        return reducer.fit_transform(X_clip), reducer.fit_transform(X_sig)

proj_clip, proj_sig = compute_projections(X_clip_all, X_sig_all, viz_type)

plot_df_clip = pd.DataFrame({"x": proj_clip[:, 0], "y": proj_clip[:, 1], "Emotion": label_names})
plot_df_sig  = pd.DataFrame({"x": proj_sig[:, 0],  "y": proj_sig[:, 1],  "Emotion": label_names})

col_l, col_r = st.columns(2)
fig_clip = px.scatter(
    plot_df_clip, x="x", y="y", color="Emotion",
    color_discrete_map=COLOR_MAP, opacity=0.7,
    title=f"{viz_type} — CLIP ViT-B/32",
    labels={"x": "dim 1", "y": "dim 2"},
)
fig_clip.update_traces(marker_size=4)
col_l.plotly_chart(fig_clip, use_container_width=True)

fig_sig = px.scatter(
    plot_df_sig, x="x", y="y", color="Emotion",
    color_discrete_map=COLOR_MAP, opacity=0.7,
    title=f"{viz_type} — SigLIP2 ViT-B/16",
    labels={"x": "dim 1", "y": "dim 2"},
)
fig_sig.update_traces(marker_size=4)
col_r.plotly_chart(fig_sig, use_container_width=True)
