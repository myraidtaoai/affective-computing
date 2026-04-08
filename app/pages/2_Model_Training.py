"""
Page 2 — Model Training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

from app.utils import (
    load_combined_df, get_or_compute_embeddings,
    get_classifiers, COMBO_KEYS, EMBEDDING_NAMES, CLASSIFIER_NAMES,
    EMB_KEY_MAP, SEED, REPO_ROOT,
)

st.set_page_config(page_title="Model Training", page_icon="🏋️", layout="wide")
st.title("🏋️ Model Training")

MLFLOW_URI  = str(REPO_ROOT / "mlruns")
EXPERIMENT  = "affective_computing_binary_emotion"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

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

# ── Section 1: Approach ───────────────────────────────────────────────────────
st.header("1. Approach")
st.markdown(
    """
We **freeze** both vision-language models and use their output embeddings as fixed feature vectors.
The classifiers are trained purely on these pre-extracted embeddings — no gradient flows through
the backbone.

```
Image  ──►  CLIP / SigLIP2 (frozen)  ──►  512-d / 768-d embedding  ──►  Classifier  ──►  neutral / happy
```

**Why this works:**
CLIP and SigLIP2 were pre-trained on hundreds of millions of image-text pairs. Their representations
already encode high-level semantic features (expressions, textures, shapes) that are discriminative
for emotion classification even without task-specific fine-tuning.
"""
)

# ── Section 2: Load embeddings ────────────────────────────────────────────────
st.header("2. Pre-extracted Embeddings")

if "emb_data" not in st.session_state:
    if st.button("Load / Compute Embeddings"):
        with st.spinner("Loading embeddings…"):
            emb_data = get_or_compute_embeddings(train_df, val_df, test_df)
            st.session_state["emb_data"] = emb_data
        st.success("Embeddings ready.")
    else:
        st.info("Embeddings must be loaded before training. Click the button above.")
        st.stop()

emb_data = st.session_state["emb_data"]
st.success(
    f"CLIP embeddings:    train={emb_data['clip']['X_train'].shape}   "
    f"val={emb_data['clip']['X_val'].shape}   test={emb_data['clip']['X_test'].shape}  \n"
    f"SigLIP2 embeddings: train={emb_data['siglip2']['X_train'].shape}   "
    f"val={emb_data['siglip2']['X_val'].shape}   test={emb_data['siglip2']['X_test'].shape}"
)

# ── Section 3: Train ──────────────────────────────────────────────────────────
st.header("3. Train All 6 Combinations")

st.markdown(
    "Each training run is logged to **MLflow** with parameters, validation metrics, and the fitted model."
)

def get_clf_params(name, clf):
    if "LogReg" in name:
        return {"C": clf.C, "max_iter": clf.max_iter}
    if "XGBoost" in name:
        return {
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "learning_rate": clf.learning_rate,
        }
    if "MLP" in name:
        return {
            "hidden_layer_sizes": str(clf.hidden_layer_sizes),
            "max_iter": clf.max_iter,
        }
    return {}


if st.button("Train All 6 Combinations"):
    classifiers = get_classifiers()
    trained_clfs = {}
    results      = {}
    log_lines    = []

    progress_bar = st.progress(0)
    status_text  = st.empty()
    log_area     = st.expander("Training log", expanded=True)

    total = len(EMBEDDING_NAMES) * len(CLASSIFIER_NAMES)
    step  = 0

    for emb_name in EMBEDDING_NAMES:
        emb_key = EMB_KEY_MAP[emb_name]
        X_tr = emb_data[emb_key]["X_train"]
        X_va = emb_data[emb_key]["X_val"]
        X_te = emb_data[emb_key]["X_test"]
        y_tr = emb_data[emb_key]["y_train"]
        y_va = val_df["label_idx"].values
        y_te = emb_data[emb_key]["y_test"]

        for clf_name, clf_template in classifiers.items():
            key = f"{emb_name} + {clf_name}"
            status_text.text(f"Training: {key}…")

            clf = clone(clf_template)
            t0  = time.time()

            with mlflow.start_run(run_name=key):
                mlflow.set_tags({"embedding": emb_name, "classifier": clf_name})
                mlflow.log_params({"embedding_model": emb_name, "classifier": clf_name,
                                   **get_clf_params(clf_name, clf)})

                clf.fit(X_tr, y_tr)
                elapsed = time.time() - t0

                train_acc = accuracy_score(y_tr, clf.predict(X_tr))
                val_acc   = accuracy_score(y_va, clf.predict(X_va))
                val_f1    = f1_score(y_va,       clf.predict(X_va))
                val_prec  = precision_score(y_va, clf.predict(X_va))
                val_rec   = recall_score(y_va,    clf.predict(X_va))

                mlflow.log_metrics({
                    "train_accuracy": round(train_acc, 4),
                    "val_accuracy":   round(val_acc, 4),
                    "val_f1":         round(val_f1, 4),
                    "val_precision":  round(val_prec, 4),
                    "val_recall":     round(val_rec, 4),
                    "training_time_s": round(elapsed, 2),
                })
                mlflow.sklearn.log_model(clf, artifact_path="model")

            y_pred_test = clf.predict(X_te)
            results[key] = {
                "accuracy":  round(accuracy_score(y_te, y_pred_test), 3),
                "f1":        round(f1_score(y_te, y_pred_test), 3),
                "precision": round(precision_score(y_te, y_pred_test), 3),
                "recall":    round(recall_score(y_te, y_pred_test), 3),
            }
            trained_clfs[key] = (clf, X_te, y_te)

            msg = (f"[{key}]  val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
                   f"test_f1={results[key]['f1']:.3f}  ({elapsed:.1f}s)")
            log_lines.append(msg)
            with log_area:
                st.text("\n".join(log_lines))

            step += 1
            progress_bar.progress(step / total)

    status_text.text("Training complete.")
    st.session_state["trained_clfs"] = trained_clfs
    st.session_state["train_results"] = results
    st.success("All 6 combinations trained and logged to MLflow.")

# ── Section 4: Cross-validation ───────────────────────────────────────────────
st.header("4. 5-Fold Cross-Validation")

if "trained_clfs" not in st.session_state:
    st.info("Run training first.")
else:
    if st.button("Run Cross-Validation on All Combinations"):
        classifiers = get_classifiers()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_rows = []

        with st.spinner("Running cross-validation…"):
            for emb_name in EMBEDDING_NAMES:
                emb_key = EMB_KEY_MAP[emb_name]
                X_tr = emb_data[emb_key]["X_train"]
                y_tr = emb_data[emb_key]["y_train"]

                for clf_name, clf_template in classifiers.items():
                    key = f"{emb_name} + {clf_name}"
                    scores = cross_validate(
                        clone(clf_template), X_tr, y_tr, cv=cv,
                        scoring=["accuracy", "f1"],
                        return_train_score=True,
                    )
                    cv_rows.append({
                        "Combination": key,
                        "CV Acc (mean)": round(scores["test_accuracy"].mean(), 3),
                        "CV Acc (std)":  round(scores["test_accuracy"].std(), 3),
                        "CV F1 (mean)":  round(scores["test_f1"].mean(), 3),
                        "CV F1 (std)":   round(scores["test_f1"].std(), 3),
                        "Train Acc":     round(scores["train_accuracy"].mean(), 3),
                    })

        cv_df = pd.DataFrame(cv_rows).sort_values("CV F1 (mean)", ascending=False)
        st.session_state["cv_df"] = cv_df

    if "cv_df" in st.session_state:
        cv_df = st.session_state["cv_df"]
        best_idx = cv_df["CV F1 (mean)"].idxmax()
        st.dataframe(
            cv_df.style.highlight_max(subset=["CV F1 (mean)"], color="#c6efce"),
            use_container_width=True, hide_index=True,
        )

# ── Section 5: MLflow run explorer ────────────────────────────────────────────
st.header("5. MLflow Run Explorer")

st.markdown(
    f"Tracking URI: `{MLFLOW_URI}`  \n"
    "To open the MLflow UI, run from the repo root:  \n"
    "```bash\nmlflow ui --backend-store-uri mlruns\n```"
)

try:
    runs_df = mlflow.search_runs(experiment_names=[EXPERIMENT])
    if runs_df.empty:
        st.info("No runs logged yet. Train the models above.")
    else:
        display_cols = [
            "tags.mlflow.runName",
            "params.embedding_model", "params.classifier",
            "metrics.val_accuracy", "metrics.val_f1",
            "metrics.val_precision", "metrics.val_recall",
            "metrics.training_time_s",
            "start_time",
        ]
        display_cols = [c for c in display_cols if c in runs_df.columns]
        runs_display = runs_df[display_cols].rename(columns={
            "tags.mlflow.runName":   "Run Name",
            "params.embedding_model": "Embedding",
            "params.classifier":     "Classifier",
            "metrics.val_accuracy":  "Val Acc",
            "metrics.val_f1":        "Val F1",
            "metrics.val_precision": "Val Prec",
            "metrics.val_recall":    "Val Rec",
            "metrics.training_time_s": "Time (s)",
            "start_time":            "Timestamp",
        })
        st.dataframe(runs_display.sort_values("Val F1", ascending=False),
                     use_container_width=True, hide_index=True)

        # Bar chart: Val F1 across runs
        if "Val F1" in runs_display.columns and "Run Name" in runs_display.columns:
            fig_mlflow = px.bar(
                runs_display.dropna(subset=["Val F1"]).sort_values("Val F1"),
                x="Val F1", y="Run Name",
                orientation="h",
                title="MLflow Runs — Validation F1",
                color="Val F1",
                color_continuous_scale="Blues",
            )
            fig_mlflow.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_mlflow, use_container_width=True)

except Exception as e:
    st.warning(f"Could not load MLflow runs: {e}")
