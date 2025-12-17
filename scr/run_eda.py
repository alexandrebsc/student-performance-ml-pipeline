# ruff: noqa: D103, T201
"""Script for Exploratory Data Analysis (EDA)."""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
from utils.constants import Col
from utils.etl import get_clean_df


def main() -> None:
    df = get_clean_df()

    df_analysis(df)
    df_vizulization(df)


def df_analysis(df: pd.DataFrame) -> None:
    def shapiro_test() -> None:
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="p-value may not be accurate for N > 5000.",
        )

        stat, p = shapiro(df[Col.inde])

        print(f"Statistics (W): {stat:.4f}\nValue p: {p:.4f}")

    def null_analysis() -> None:
        df_nulls = df.isna()

        print("Quantidade de nulos: ", df_nulls.sum())

    def duplicated_analysis() -> None:
        df_duplicated = df.duplicated()

        print("Quantidade de duplicados: ", df_duplicated.sum())
        print(df[df_duplicated])

    print("> Info:\n")
    df.info()
    print("\n")
    print("> Análise dados nulos:\n\n")
    null_analysis()
    print("> Análise dados duplicados:\n\n")
    duplicated_analysis()
    print(f"> Df sample:\n\n{df.head(3)}", end="\n\n")
    print(f"> Df statistics:\n\n{df.describe()}", end="\n\n")
    print("> Shapiro test on percentual variation by day:\n\n")
    shapiro_test()
    print("\n")


def df_vizulization(df: pd.DataFrame) -> None:
    def hist_plot() -> None:
        _, axs = plt.subplots(5, 1, figsize=(8, 8))

        df[[Col.iaa, Col.inde]].plot.hist(bins=100, alpha=0.5, ax=axs[0])
        axs[0].set_title("iaa vs inde")

        df[[Col.ieg, Col.inde]].plot.hist(bins=100, alpha=0.5, ax=axs[1])
        axs[1].set_title("ieg vs inde")

        df[[Col.level, Col.inde]].plot.hist(bins=100, alpha=0.5, ax=axs[2])
        axs[2].set_title("level vs inde")

        df[[Col.age, Col.inde]].plot.hist(bins=100, alpha=0.5, ax=axs[3])
        axs[3].set_title("age vs inde")

    def correlation_plot() -> None:
        corr_columns = list(df.columns)
        corr_columns.remove(Col.name)
        correlation_matrix = df[corr_columns].corr().round(2)

        _, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(data=correlation_matrix, annot=True, linewidths=0.5, ax=ax)

    hist_plot()
    correlation_plot()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
