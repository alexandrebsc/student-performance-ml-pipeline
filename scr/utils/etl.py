# ruff: noqa: D103
"""ETL functions used to get and preapre the data for processing."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle
from utils.constants import DATA_FODLER_PATH, Col


def get_clean_df() -> pd.DataFrame:
    df = get_students_df()
    return data_cleaning(df)


def get_students_df() -> pd.DataFrame:
    file_path = f"{DATA_FODLER_PATH}\\pede_passos.csv"
    pd.set_option("display.max_columns", None)
    return pd.read_csv(
        file_path,
        encoding="utf-8",
        delimiter=";",
        engine="python",
    )


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    def __main(df: pd.DataFrame) -> pd.DataFrame:
        df = df[
            [
                "NOME",
                "IDADE_ALUNO_2020",
                "ANOS_PM_2020",
                "FASE_TURMA_2020",
                "INDE_2020",
                "IAA_2020",
                "IAN_2020",
                "IEG_2020",
                "IDA_2020",
                "SINALIZADOR_INGRESSANTE_2021",
                "FASE_2021",
                "INDE_2021",
                "IEG_2021",
                "IAA_2021",
                "IDA_2021",
                "IAN_2021",
                "ANO_INGRESSO_2022",
                "FASE_2022",
                "INDE_2022",
                "IEG_2022",
                "IAA_2022",
                "IDA_2022",
                "IAN_2022",
                "IPS_2020",
                "IPS_2021",
                "IPS_2022",
                "IPP_2020",
                "IPP_2021",
                "IPP_2022",
                "IPV_2020",
                "IPV_2021",
                "IPV_2022",
            ]
        ]

        df = drop_corrupted_date(df)
        df = fill_age_data(df)
        df = preprocess_is_first_year(df)
        df = preprocess_level_data(df)
        df = cast_values(df)
        df = unpivot_by_year(df)
        df = df.dropna(
            subset=df.columns.difference([Col.name, Col.year, Col.first_year, Col.age]),
            how="all",
        )
        df = df.astype(
            {
                Col.year: int,
                Col.level: int,
                Col.age: int,
                Col.inde: float,
                Col.iaa: float,
                Col.ian: float,
                Col.ida: float,
                Col.ieg: float,
                Col.ipp: float,
                Col.ips: float,
                Col.ipv: float,
                Col.first_year: bool,
            },
        )

        undergraduate_student_level = 8
        return df.loc[df[Col.level] < undergraduate_student_level]

    def drop_corrupted_date(df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("NOME")
        return df.drop(["ALUNO-1259", "ALUNO-506", "ALUNO-71"], axis=0)

    def fill_age_data(df: pd.DataFrame) -> pd.DataFrame:
        def expected_age_in_2021(age_in_2020: int) -> int:
            average_2021_age = 13
            return (age_in_2020 + 1) if not np.isnan(age_in_2020) else average_2021_age

        def expected_age_in_2022(age_in_2020: int) -> int:
            average_2023_age = 13
            return (age_in_2020 + 2) if not np.isnan(age_in_2020) else average_2023_age

        df["IDADE_2020"] = pd.to_numeric(df["IDADE_ALUNO_2020"], downcast="integer")
        df["IDADE_2021"] = df["IDADE_2020"].apply(expected_age_in_2021)
        df["IDADE_2022"] = df["IDADE_2020"].apply(expected_age_in_2022)
        return df

    def preprocess_is_first_year(df: pd.DataFrame) -> pd.DataFrame:
        df["INGRESSANTE_2020"] = df["ANOS_PM_2020"].apply(
            lambda x: x != "0" if x else None,
        )
        df["INGRESSANTE_2021"] = df["SINALIZADOR_INGRESSANTE_2021"].apply(
            lambda x: x == "Ingressante" if x else None,
        )
        df["INGRESSANTE_2022"] = pd.to_numeric(
            df["ANO_INGRESSO_2022"],
            downcast="integer",
        ).apply(lambda x: x == 2022 if x else None)  # noqa: PLR2004
        return df.drop(
            ["ANOS_PM_2020", "ANO_INGRESSO_2022", "SINALIZADOR_INGRESSANTE_2021"],
            axis=1,
        )

    def preprocess_level_data(df: pd.DataFrame) -> pd.DataFrame:
        df["FASE_2020"] = pd.to_numeric(
            df["FASE_TURMA_2020"].str[0],
            downcast="integer",
        )
        df = df.drop("FASE_TURMA_2020", axis=1)
        df["FASE_2021"] = pd.to_numeric(df["FASE_2021"], downcast="integer")
        df["FASE_2022"] = pd.to_numeric(df["FASE_2022"], downcast="integer")
        return df

    def cast_values(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(
            {
                "IDADE_2020": "Int32",
                "IDADE_2021": "Int32",
                "IDADE_2022": "Int32",
                "FASE_2020": "Int32",
                "FASE_2021": "Int32",
                "FASE_2022": "Int32",
                "INDE_2020": float,
                "IAA_2020": float,
                "IAN_2020": float,
                "IEG_2020": float,
                "IDA_2020": float,
                "IPS_2020": float,
                "IPP_2020": float,
                "IPV_2020": float,
                "INDE_2021": float,
            },
        )

    def unpivot_by_year(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index()
        df = pd.melt(
            df,
            id_vars="NOME",
            value_vars=[
                "IDADE_2020",
                "IDADE_2021",
                "IDADE_2022",
                "INGRESSANTE_2020",
                "INGRESSANTE_2021",
                "INGRESSANTE_2022",
                "FASE_2020",
                "FASE_2021",
                "FASE_2022",
                "INDE_2020",
                "INDE_2021",
                "INDE_2022",
                "IAA_2020",
                "IAA_2021",
                "IAA_2022",
                "IAN_2020",
                "IAN_2021",
                "IAN_2022",
                "IEG_2020",
                "IEG_2021",
                "IEG_2022",
                "IDA_2020",
                "IDA_2021",
                "IDA_2022",
                "IPS_2020",
                "IPS_2021",
                "IPS_2022",
                "IPP_2020",
                "IPP_2021",
                "IPP_2022",
                "IPV_2020",
                "IPV_2021",
                "IPV_2022",
            ],
            var_name="COLUNA_ANO",
            value_name="value",
        )

        df[["COLUNA", "ANO"]] = df["COLUNA_ANO"].str.split("_", expand=True)
        df = df.drop("COLUNA_ANO", axis=1)
        return df.pivot_table(
            index=["NOME", "ANO"],
            columns="COLUNA",
            values="value",
        ).reset_index()

    return __main(df)


def data_augmentation(df: pd.DataFrame) -> pd.DataFrame:
    def get_augmented_df(
        base_df: pd.DataFrame,
        column: str,
        value: float,
    ) -> pd.DataFrame:
        __df = base_df.copy()
        __df[column] = __df[column] + value
        __df[column] = __df[column].clip(lower=0.0, upper=10.0)
        return __df

    for augmentation_group in [
        ((Col.iaa, 1), (Col.iaa, -1)),
        ((Col.ieg, 0.66), (Col.ieg, -0.66)),
        ((Col.ida, 1), (Col.ida, -1)),
    ]:
        dfs_to_concat = [
            get_augmented_df(df, augmentation[0], augmentation[1])
            for augmentation in augmentation_group
        ]

        df = pd.concat([*dfs_to_concat, df], ignore_index=True)
        df = df.drop_duplicates()

    def calculate_inde(row: Mapping[Any, Any]) -> list:
        return (
            row[Col.ian] * 0.1
            + row[Col.ida] * 0.2
            + row[Col.ieg] * 0.2
            + row[Col.iaa] * 0.1
            + row[Col.ips] * 0.1
            + row[Col.ipp] * 0.1
            + row[Col.ipv] * 0.2
        )

    df[Col.inde] = df.apply(calculate_inde, axis=1)

    df = df.drop_duplicates()
    return df


def pre_processing(df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    class KeepFeatures(BaseEstimator, TransformerMixin):
        def __init__(self, feature_to_keep: list) -> None:
            self.feature_to_keep = feature_to_keep

        def fit(self, df: pd.DataFrame):  # noqa: ANN202, ARG002
            return self

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            to_columns_drop = set(df.columns) - set(self.feature_to_keep)
            if to_columns_drop:
                return df.drop(to_columns_drop, axis=1)
            return df

    class DivideByScaler(BaseEstimator, TransformerMixin):
        def __init__(self, divisor: float, features: list) -> None:
            self.divisor = divisor
            self.features = features

        def fit(self, df: pd.DataFrame):  # noqa: ANN202, ARG002
            return self

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            for feature in self.features:
                df[feature] = df[feature] / self.divisor
            return df

    class Shuffle(BaseEstimator, TransformerMixin):
        def fit(self, df: pd.DataFrame):  # noqa: ANN202, ARG002
            return self

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            return shuffle(df)

    pre_processor = ColumnTransformer(
        transformers=[
            ("label_encoder", OrdinalEncoder(), [Col.first_year]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    pre_processor.set_output(transform="pandas")

    feature_to_keep = [
        Col.age,
        Col.level,
        Col.inde,
        Col.ian,
        Col.iaa,
        Col.ieg,
        Col.first_year,
    ]
    pipeline = Pipeline(
        [
            ("keep_features", KeepFeatures(feature_to_keep)),
            (
                "divide_by_scaler_i",
                DivideByScaler(10.0, [Col.inde, Col.ian, Col.iaa, Col.ieg]),
            ),
            ("divide_by_scaler_level", DivideByScaler(7.0, [Col.level])),
            ("divide_by_scaler_age", DivideByScaler(30.0, [Col.age])),
            ("pre_processor", pre_processor),
            ("shuffle", Shuffle()),
        ],
    )

    return pipeline.fit_transform(df)
