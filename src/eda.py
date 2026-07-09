import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import PROCESSED_DIR, FIGURES_DIR, TARGET_COLUMN


def save_eda_figures(df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=TARGET_COLUMN)
    plt.title("Class Balance: Heart Disease Target")
    plt.xlabel("0 = No disease, 1 = Disease")
    plt.ylabel("Patient count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=160)
    plt.close()

    df.hist(figsize=(14, 10), bins=20)
    plt.suptitle("Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "histograms.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    missing = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.index, y=missing.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Missing Value Analysis")
    plt.ylabel("Missing values")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png", dpi=160)
    plt.close()


def main():
    df = pd.read_csv(PROCESSED_DIR / "heart_disease_clean.csv")
    save_eda_figures(df)
    print("EDA plots saved in docs/figures.")


if __name__ == "__main__":
    main()
