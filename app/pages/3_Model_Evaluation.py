"""
Page 3 — Model Evaluation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.base import clone

from app.utils import (
    load_combined_df, get_or_compute_embeddings, get_classifiers,
    EMBEDDING_NAMES, CLASSIFIER_NAMES, EMB_KEY_MAP, SEED,
)

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
st.title("📊 Model Evaluation")

# ── Load data ─────────────────────────────────────────────────────────────────
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

# ── Ensure embeddings ─────────────────────────────────────────────────────────
if "emb_data" not in st.session_state:
    if st.button("Load / Compute Embeddings"):
        with st.spinner("Loading embeddings…"):
            emb_data = get_or_compute_embeddings(train_df, val_df, test_df)
            st.session_state["emb_data"] = emb_data
        st.success("Embeddings ready.")
    else:
        st.info("Embeddings must be loaded before evaluation.")
        st.stop()

emb_data = st.session_state["emb_data"]

# ── Ensure trained classifiers ────────────────────────────────────────────────
if "trained_clfs" not in st.session_state:
    st.warning(
        "No trained classifiers found in session. "
        "Go to **Model Training** page and train first, or use the quick-train button below."
    )
    if st.button("Quick-Train All 6 Combinations"):
        classifiers = get_classifiers()
        trained_clfs = {}
        results = {}
        pb = st.progress(0)
        total = len(EMBEDDING_NAMES) * len(CLASSIFIER_NAMES)
        step = 0
        for emb_name in EMBEDDING_NAMES:
            emb_key = EMB_KEY_MAP[emb_name]
            X_tr = emb_data[emb_key]["X_train"]
            X_te = emb_data[emb_key]["X_test"]
            y_tr = emb_data[emb_key]["y_train"]
            y_te = emb_data[emb_key]["y_test"]
            for clf_name, clf_tmpl in classifiers.items():
                key = f"{emb_name} + {clf_name}"
                clf = clone(clf_tmpl)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                results[key] = {
                    "accuracy":  round(accuracy_score(y_te, y_pred), 3),
                    "f1":        round(f1_score(y_te, y_pred), 3),
                    "precision": round(precision_score(y_te, y_pred), 3),
                    "recall":    round(recall_score(y_te, y_pred), 3),
                }
                trained_clfs[key] = (clf, X_te, y_te)
                step += 1
                pb.progress(step / total)
        st.session_state["trained_clfs"] = trained_clfs
        st.session_state["train_results"] = results
        st.rerun()
    st.stop()

trained_clfs = st.session_state["trained_clfs"]
results      = st.session_state.get("train_results", {})

# Rebuild results if missing
if not results:
    for key, (clf, X_te, y_te) in trained_clfs.items():
        y_pred = clf.predict(X_te)
        results[key] = {
            "accuracy":  round(accuracy_score(y_te, y_pred), 3),
            "f1":        round(f1_score(y_te, y_pred), 3),
            "precision": round(precision_score(y_te, y_pred), 3),
            "recall":    round(recall_score(y_te, y_pred), 3),
        }

best_key = max(results, key=lambda k: results[k]["f1"])

# ── Section 1: Results table ──────────────────────────────────────────────────
st.header("1. Test-Set Results")

results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Combination"})
results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)

def highlight_best(row):
    is_best = row["Combination"] == best_key
    return ["background-color: #c6efce" if is_best else "" for _ in row]

st.dataframe(
    results_df.style.apply(highlight_best, axis=1),
    use_container_width=True,
    hide_index=True,
)
st.success(f"Best combination: **{best_key}** — Test F1 = {results[best_key]['f1']:.3f}")

# ── Section 2: F1 heatmap ─────────────────────────────────────────────────────
st.header("2. F1 Score Heatmap")

f1_matrix = pd.DataFrame(
    index=CLASSIFIER_NAMES,
    columns=EMBEDDING_NAMES,
    dtype=float,
)
for emb_name in EMBEDDING_NAMES:
    for clf_name in CLASSIFIER_NAMES:
        key = f"{emb_name} + {clf_name}"
        if key in results:
            f1_matrix.loc[clf_name, emb_name] = results[key]["f1"]

fig_heat = px.imshow(
    f1_matrix.astype(float),
    text_auto=".3f",
    color_continuous_scale="YlGnBu",
    zmin=0.5, zmax=1.0,
    title="F1 Score — Classifier × Embedding Model",
    labels={"x": "Embedding", "y": "Classifier", "color": "F1"},
)
fig_heat.update_layout(height=320)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Section 3: Multi-metric bar chart ────────────────────────────────────────
st.header("3. All Metrics Comparison")

metrics = ["accuracy", "f1", "precision", "recall"]
bar_df = results_df.melt(id_vars="Combination", value_vars=metrics,
                          var_name="Metric", value_name="Score")
fig_bar = px.bar(
    bar_df, x="Combination", y="Score", color="Metric",
    barmode="group", title="All Metrics — Test Set",
    range_y=[0.5, 1.0],
)
fig_bar.update_layout(height=400, xaxis_tickangle=-20)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Section 4: Confusion matrix ───────────────────────────────────────────────
st.header("4. Confusion Matrix")

combo_choice = st.selectbox("Select combination", list(results.keys()),
                             index=list(results.keys()).index(best_key)
                             if best_key in results else 0)

clf_fit, X_te, y_te = trained_clfs[combo_choice]
y_pred = clf_fit.predict(X_te)
cm = confusion_matrix(y_te, y_pred)

fig_cm = ff.create_annotated_heatmap(
    cm,
    x=["Predicted neutral", "Predicted happy"],
    y=["True neutral",      "True happy"],
    colorscale="Blues",
    showscale=True,
)
fig_cm.update_layout(title=f"Confusion Matrix — {combo_choice}", height=350)
st.plotly_chart(fig_cm, use_container_width=True)

# ── Section 5: Classification report ─────────────────────────────────────────
st.header("5. Detailed Classification Report")

report_choice = st.selectbox(
    "Select combination for classification report",
    list(results.keys()),
    index=list(results.keys()).index(best_key) if best_key in results else 0,
    key="report_select",
)
clf_fit2, X_te2, y_te2 = trained_clfs[report_choice]
y_pred2 = clf_fit2.predict(X_te2)
report_dict = classification_report(y_te2, y_pred2,
                                     target_names=["neutral", "happy"],
                                     output_dict=True)
report_df = pd.DataFrame(report_dict).T.round(3)
st.dataframe(report_df, use_container_width=True)
