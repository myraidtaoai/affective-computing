"""
Affective Computing Dashboard — Home Page
==========================================
Run from the repo root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure repo root is on the Python path so `app.utils` resolves correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd

from app.utils import load_combined_df, LABEL_MAP

st.set_page_config(
    page_title="Affective Computing",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("😊 Affective Computing — Emotion Classification Dashboard")
st.markdown(
    """
This dashboard illustrates a complete facial emotion recognition pipeline:
**neutral** vs **happy** classification using frozen vision-language embeddings
(CLIP & SigLIP2) and three classifiers (Logistic Regression, XGBoost, Deep Probe MLP).

Use the **sidebar** to navigate between pages.
"""
)

st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
combined_df, train_df, val_df, test_df = load_combined_df()

# Store splits in session_state so all pages can reuse them
st.session_state["combined_df"] = combined_df
st.session_state["train_df"]    = train_df
st.session_state["val_df"]      = val_df
st.session_state["test_df"]     = test_df

# ── Dataset summary cards ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total images", f"{len(combined_df):,}")
c2.metric("Training images", f"{len(train_df):,}")
c3.metric("Val images", f"{len(val_df):,}")
c4.metric("Test images", f"{len(test_df):,}")

st.subheader("Class Distribution")

col_a, col_b = st.columns(2)

# Combined class bar
counts = combined_df["label"].value_counts().reset_index()
counts.columns = ["Emotion", "Count"]
fig_combined = px.bar(
    counts, x="Emotion", y="Count",
    color="Emotion",
    color_discrete_map={"neutral": "#4C72B0", "happy": "#DD8452"},
    title="Combined Dataset",
    text_auto=True,
)
fig_combined.update_layout(showlegend=False, height=300)
col_a.plotly_chart(fig_combined, use_container_width=True)

# Per-source class bar
src_counts = combined_df.groupby(["source", "label"]).size().reset_index(name="Count")
fig_src = px.bar(
    src_counts, x="source", y="Count", color="label",
    barmode="group",
    color_discrete_map={"neutral": "#4C72B0", "happy": "#DD8452"},
    title="Per Source",
    labels={"source": "Dataset", "label": "Emotion"},
)
fig_src.update_layout(height=300)
col_b.plotly_chart(fig_src, use_container_width=True)

# ── Pipeline overview ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Pipeline Overview")
st.markdown(
    """
| Step | Details |
|------|---------|
| **Data** | FER-2013 (20% sample, ~3 037 images) + custom training_set (500 images) |
| **Embedding models** | `openai/clip-vit-base-patch32` (512-d) · `google/siglip2-base-patch16-224` (768-d) |
| **Classifiers** | Logistic Regression · XGBoost · Deep Probe MLP (512→256→128) |
| **Evaluation** | 5-fold CV + held-out test set · accuracy, F1, precision, recall |
| **Best model** | **SigLIP2 + Deep Probe — Test F1 = 0.949** |
| **Experiment tracking** | MLflow (local `mlruns/` directory) |

Navigate to the pages in the sidebar to explore each step interactively.
"""
)

st.sidebar.success("Select a page above.")
