"""Exploratory Data Analysis script.

Run:  python src/eda.py
Prints dataset inspection to the console and saves figures to reports/figures/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed; figures are saved to disk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402

sns.set_theme(style="whitegrid")


def save(fig, name: str) -> None:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[eda] saved {path}")


def main() -> None:
    df = load_dataset()

    # ---------------- 1. basic inspection
    print("\n=== SHAPE ===", df.shape)
    print("\n=== DTYPES ===\n", df.dtypes)
    print("\n=== HEAD ===\n", df.head())
    print("\n=== DESCRIBE ===\n", df.describe().T)
    print("\n=== DUPLICATE ROWS ===", int(df.duplicated().sum()))

    # ---------------- 2. missing values
    missing = df.isna().sum()
    print("\n=== MISSING VALUES ===\n", missing[missing > 0] if missing.any() else "None")
    fig, ax = plt.subplots(figsize=(8, 4))
    missing_all = missing.sort_values(ascending=False)
    sns.barplot(x=missing_all.index, y=missing_all.values, color="#c0392b", ax=ax)
    ax.set_title("Missing values per column")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=60)
    save(fig, "01_missing_values.png")

    # ---------------- 3. class distributions
    print("\n=== SOURCE TARGET (num) ===\n", df[cfg.SOURCE_TARGET_COL].value_counts().sort_index())
    print("\n=== BINARY TARGET ===\n", df[cfg.TARGET_COL].value_counts().sort_index())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(x=cfg.SOURCE_TARGET_COL, data=df, color="#2980b9", ax=axes[0])
    axes[0].set_title("Original UCI target (num: 0-4)")
    sns.countplot(x=cfg.TARGET_COL, data=df, palette=["#27ae60", "#c0392b"], hue=cfg.TARGET_COL,
                  legend=False, ax=axes[1])
    axes[1].set_title("Binary label (0 = no disease, 1 = disease present)")
    save(fig, "02_class_distribution.png")

    # ---------------- 4. numeric feature distributions by class
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cfg.NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=cfg.TARGET_COL, kde=(col != "ca"), bins=20,
                     palette=["#27ae60", "#c0392b"], ax=ax)
        ax.set_title(f"{col} by class")
    save(fig, "03_numeric_distributions.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cfg.NUMERIC_FEATURES):
        sns.boxplot(data=df, x=cfg.TARGET_COL, y=col, hue=cfg.TARGET_COL,
                    palette=["#27ae60", "#c0392b"], legend=False, ax=ax)
        ax.set_title(f"{col} vs class")
    save(fig, "04_numeric_boxplots.png")

    # ---------------- 5. categorical features by class
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.ravel(), cfg.CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=cfg.TARGET_COL, palette=["#27ae60", "#c0392b"], ax=ax)
        ax.set_title(f"{col} by class")
    axes.ravel()[-1].axis("off")
    save(fig, "05_categorical_counts.png")

    # ---------------- 6. correlation heatmap (target + all features, NaN pairwise-ignored)
    corr = df[cfg.FEATURE_COLUMNS + [cfg.TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    save(fig, "06_correlation_heatmap.png")

    print("\n=== CORRELATION WITH TARGET ===")
    print(corr[cfg.TARGET_COL].drop(cfg.TARGET_COL).sort_values(ascending=False))
    print("\n[eda] Done. Figures are in", cfg.FIGURES_DIR)


if __name__ == "__main__":
    main()
