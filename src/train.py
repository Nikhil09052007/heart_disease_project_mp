"""Training pipeline: split -> cross-validated model selection -> test evaluation -> save.

Run:  python src/train.py

Leakage control: ALL preprocessing (imputation, scaling, one-hot encoding) lives inside
a scikit-learn Pipeline, so during cross-validation it is re-fitted on each training
fold only. The held-out test set is touched exactly once, at the end.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, cfg.NUMERIC_FEATURES),
        ("cat", categorical, cfg.CATEGORICAL_FEATURES),
    ])


def build_candidates() -> dict:
    """Each candidate = (pipeline, small hyper-parameter grid)."""
    lr = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, random_state=cfg.RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(random_state=cfg.RANDOM_STATE)),
    ])
    return {
        "Logistic Regression": (lr, {"model__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (rf, {"model__n_estimators": [100, 300],
                               "model__max_depth": [3, 5, None],
                               "model__min_samples_leaf": [1, 3]}),
    }


def main() -> None:
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X = df[cfg.FEATURE_COLUMNS]
    y = df[cfg.TARGET_COL]
    print(f"[train] Rows: {len(df)} | class counts: {y.value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE
    )
    print(f"[train] Train: {X_train.shape}  Test: {X_test.shape}")

    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

    # ---------- model selection with cross-validation on TRAINING data only
    rows, fitted = [], {}
    for name, (pipe, grid) in build_candidates().items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
        search.fit(X_train, y_train)
        best_idx = search.best_index_
        # Also report CV accuracy / F1 of the best params for a fuller comparison.
        from sklearn.model_selection import cross_validate
        cvres = cross_validate(search.best_estimator_, X_train, y_train, cv=cv,
                               scoring=["accuracy", "f1", "roc_auc"])
        rows.append({
            "model": name,
            "best_params": json.dumps(search.best_params_),
            "cv_roc_auc_mean": float(search.cv_results_["mean_test_score"][best_idx]),
            "cv_roc_auc_std": float(search.cv_results_["std_test_score"][best_idx]),
            "cv_accuracy_mean": float(np.mean(cvres["test_accuracy"])),
            "cv_f1_mean": float(np.mean(cvres["test_f1"])),
        })
        fitted[name] = search.best_estimator_
        print(f"[train] {name}: CV ROC-AUC = {rows[-1]['cv_roc_auc_mean']:.4f} "
              f"(+/- {rows[-1]['cv_roc_auc_std']:.4f}) params={search.best_params_}")

    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    comparison.to_csv(cfg.COMPARISON_PATH, index=False)
    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"[train] Selected model (highest CV ROC-AUC): {best_name}")

    # ---------- single evaluation on the held-out test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "selected_model": best_name,
        "best_params": comparison.iloc[0]["best_params"],
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "cv_results": comparison.to_dict(orient="records"),
        "feature_columns": cfg.FEATURE_COLUMNS,
    }
    with open(cfg.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== HELD-OUT TEST METRICS ===")
    for k in ("test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"):
        print(f"{k:16s}: {metrics[k]:.4f}")
    print(f"confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    # ---------- figures
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "07_confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, name=best_name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title("ROC curve (test set)"); ax.legend()
    fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "08_roc_curve.png", dpi=130); plt.close(fig)

    # Interpretability: coefficients (LR) or importances (RF) on the transformed features
    try:
        names = best_model.named_steps["preprocess"].get_feature_names_out()
        mdl = best_model.named_steps["model"]
        vals = mdl.feature_importances_ if hasattr(mdl, "feature_importances_") else np.abs(mdl.coef_[0])
        label = "Importance" if hasattr(mdl, "feature_importances_") else "|Coefficient|"
        imp = pd.Series(vals, index=names).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.plot.barh(color="#2980b9", ax=ax)
        ax.set_title(f"Top features - {best_name}"); ax.set_xlabel(label)
        fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "09_feature_importance.png", dpi=130); plt.close(fig)
    except Exception as exc:
        print(f"[train] Skipped feature-importance plot: {exc}")

    # ---------- save the complete fitted pipeline (preprocessing + model)
    joblib.dump(best_model, cfg.MODEL_PATH)
    print(f"\n[train] Saved pipeline -> {cfg.MODEL_PATH}")
    print(f"[train] Saved metrics  -> {cfg.METRICS_PATH}")


if __name__ == "__main__":
    main()
