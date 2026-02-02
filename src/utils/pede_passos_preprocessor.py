"""Script for the module PedePassosPreprocessor."""

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from utils.constants import Col


class PedePassosPreprocessor:
    """Module responsible to preprocess the students historical data making it usable for modeling."""  # noqa: E501

    def __init__(self) -> None:
        """Initialize a PedePassosPreprocessor."""

    def preprocess(
        self,
        df: pd.DataFrame,
        feature_to_keep: list[str] | None = None,
        *,
        augment_data: bool = False,
    ) -> pd.DataFrame:
        """Preprocess the students historical data, making it usable for modeling.

        :param df: Students historical data
        :type df: pd.DataFrame
        :param feature_to_keep: List of columns to keep as features
        :type feature_to_keep: list[str]
        :param augment_data: If True, augments dataset with generated data
        :type augment_data: bool
        :return: Preprocessed students historical data
        :rtype: DataFrame
        """
        if feature_to_keep is None:
            feature_to_keep = df.columns.tolist()

        df = df.copy()

        if augment_data:
            df = self._data_augmentation(df)

        return self._pre_processing(df, feature_to_keep)

    def _pre_processing(
        self,
        df: pd.DataFrame,
        feature_to_keep: list[str],
    ) -> pd.DataFrame:
        pre_processor = ColumnTransformer(
            transformers=[
                (
                    "bool_to_int",
                    FunctionTransformer(lambda x: x.astype(float)),
                    [Col.first_year],
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        pre_processor.set_output(transform="pandas")

        pipeline = Pipeline(
            [
                (
                    "scale_i",
                    FunctionTransformer(
                        self.scale_cols,
                        kw_args={
                            "divisor": 10.0,
                            "cols": [Col.inde, Col.ian, Col.iaa, Col.ieg],
                        },
                    ),
                ),
                (
                    "scale_level",
                    FunctionTransformer(
                        self.scale_cols,
                        kw_args={"divisor": 7.0, "cols": [Col.level]},
                    ),
                ),
                (
                    "scale_age",
                    FunctionTransformer(
                        self.scale_cols,
                        kw_args={"divisor": 30.0, "cols": [Col.age]},
                    ),
                ),
                (
                    "bool_to_float",
                    ColumnTransformer(
                        [
                            (
                                "first_year_cast",
                                FunctionTransformer(lambda x: x.astype(float)),
                                [Col.first_year],
                            ),
                        ],
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                    ).set_output(transform="pandas"),
                ),
                ("keep_features", FunctionTransformer(lambda x: x[feature_to_keep])),
            ],
        )

        return pipeline.fit_transform(df)[feature_to_keep]

    @staticmethod
    def scale_cols(data: pd.DataFrame, divisor: float, cols: list[str]) -> pd.DataFrame:
        """Scale columns by a divisor."""
        data = data.copy()
        data[cols] = data[cols] / divisor
        return data

    def _data_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        for augmentation_group in [
            ((Col.iaa, 1), (Col.iaa, -1)),
            ((Col.ieg, 0.66), (Col.ieg, -0.66)),
            ((Col.ida, 1), (Col.ida, -1)),
        ]:
            dfs_to_concat = [
                self.__get_augmented_df(df, augmentation[0], augmentation[1])
                for augmentation in augmentation_group
            ]

            df = pd.concat([*dfs_to_concat, df], ignore_index=True)
            df = df.drop_duplicates()

        df[Col.inde] = df.apply(self.__calculate_inde, axis=1)

        df = df.drop_duplicates()
        return df

    def __get_augmented_df(
        self,
        base_df: pd.DataFrame,
        column: str,
        value: float,
    ) -> pd.DataFrame:
        __df = base_df.copy()
        __df[column] = __df[column] + value
        __df[column] = __df[column].clip(lower=0.0, upper=10.0)
        return __df

    def __calculate_inde(self, row: Mapping[str, Any]) -> float:
        return (
            row[Col.ian] * 0.1
            + row[Col.ida] * 0.2
            + row[Col.ieg] * 0.2
            + row[Col.iaa] * 0.1
            + row[Col.ips] * 0.1
            + row[Col.ipp] * 0.1
            + row[Col.ipv] * 0.2
        )
