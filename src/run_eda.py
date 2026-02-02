"""Script for Exploratory Data Analysis (EDA)."""

import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro

from utils.constants import Col
from utils.pede_passos_loader import PedePassosLoader

logger = logging.getLogger(__name__)


def main() -> None:  # noqa: D103
    logging.basicConfig(level=logging.INFO)

    pd.set_option("display.max_columns", None)

    df = PedePassosLoader().load()

    df_analysis(df)
    df_vizulization(df)


def df_analysis(df: pd.DataFrame) -> None:
    """Log execution of the PEDE Passos dataset analysis."""

    def shapiro_test() -> None:
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="p-value may not be accurate for N > 5000.",
        )

        stat, p = shapiro(df[Col.inde])

        logger.info("Shapiro test statistics (W): %.4f | Value p: %.4f", stat, p)

    def null_analysis() -> None:
        df_nulls = df.isna()
        logger.info("Quantidade de nulos: %s", df_nulls.sum())

    def duplicated_analysis() -> None:
        df_duplicated = df.duplicated()

        duplicated_num = df_duplicated.sum()
        logger.info("Quantidade de duplicados: %s", duplicated_num)
        if duplicated_num > 0:
            logger.info("Linhas duplicadas:\n %s", df[df_duplicated])

    df.info()

    logger.info("DF sample:\n\n%s\n\n", df.head(3))
    logger.info("DF statistics:\n\n%s\n\n", df.describe())

    null_analysis()
    duplicated_analysis()
    shapiro_test()


def df_vizulization(df: pd.DataFrame) -> None:
    """Plot histograms and a correlation graph for the PEDE Passos dataset."""

    def hist_plot() -> None:
        _, axs = plt.subplots(4, 1, figsize=(8, 8))

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
