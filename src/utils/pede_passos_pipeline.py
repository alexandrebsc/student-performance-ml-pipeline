"""Script for the module PedePassosPipeline."""

import random
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from utils.evaluator import Evaluator
from utils.pede_passos_loader import PedePassosLoader
from utils.pede_passos_preprocessor import PedePassosPreprocessor
from utils.random_forest_regressor_model import RandomForestRegressorModel


class PedePassosPipeline:
    """Pipeline for the complete ML workflow."""

    def __init__(
        self,
        loader: PedePassosLoader,
        preprocessor: PedePassosPreprocessor,
        model: RandomForestRegressorModel,
        evaluator: Evaluator,
        random_seed: int | None = None,
    ) -> None:
        """Initialize a PedePassosPipeline."""
        self.loader = loader
        self.preprocessor = preprocessor
        self.model = model
        self.evaluator = evaluator
        self.random_seed = random_seed

    def run(
        self,
        features: list[str],
        target: str,
        train_percentage: float = 0.8,
        validation_percentage: float = 0.1,
        *,
        augment_data: bool = False,
    ) -> None:
        """Run pipeline.

        The pipeline load data, preprocess it, split in training, validadtion
        and test datasets, train the model and evaluate it.

        :param features: List of columns to use as features in the model (X)
        :type features: list[str]
        :param target: Column used as target for the model (y)
        :type target: str
        :param train_percentage: Percentage of the dataset used for train. Default: 0.8
        :type train_percentage: float
        :param validation_percentage: Percentage of the dataset used for validation.
            Default: 0.1
        :type validation_percentage: float
        :param augment_data: If True, augments dataset with generated data
        :type augment_data: bool
        """
        self.__set_random_seeds()

        df_raw = self.loader.load()
        df_train, df_validation, df_test = self._split_dataset(
            df_raw,
            train_percentage,
            validation_percentage,
        )

        self.preprocessor_features = [*features, target]
        df_train = self.preprocessor.preprocess(
            df_train,
            self.preprocessor_features,
            augment_data=augment_data,
        )
        df_validation = self.preprocessor.preprocess(
            df_validation,
            self.preprocessor_features,
        )
        df_test = self.preprocessor.preprocess(df_test, self.preprocessor_features)

        get_x_y_from_df = partial(
            self.get_x_y_from_df,
            features=features,
            target=target,
        )

        self.model.fit_by_grid_search(
            df_train,
            df_validation,
            self.evaluator,
            get_x_y_from_df,
        )

        _, train_y = get_x_y_from_df(df_train)
        for header, df in (
            ("Train", df_train),
            ("Validation", df_validation),
            ("Test", df_test),
        ):
            x, y = get_x_y_from_df(df)

            self.evaluator.evaluate(
                y_train=train_y,
                y_actual=y,
                y_predicted=self.model.get_model().predict(x),
                header=header,
            )

    def save_model(
        self, path: str = "models/random_forest_regressor_predict_student_inde.pkl",
    ) -> None:
        """Save model and preprocessor for reusability."""
        joblib.dump(
            {
                "model": self.model.get_model(),
                "preprocessor": self.preprocessor,
                "preprocessor_features": self.preprocessor_features,
            },
            path,
        )

    def _split_dataset(
        self,
        df: pd.DataFrame,
        train_percentage: float = 0.8,
        validation_percentage: float = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if sum([train_percentage, validation_percentage]) > 1:
            msg = (
                "Sum of train_percentage and validation_percentage "
                "must result in less than 1"
            )
            raise RuntimeError(msg)

        df = shuffle(df.copy(), random_state=self.random_seed)
        n = len(df)

        train_end = int(n * train_percentage)
        validation_end = int(n * (train_percentage + validation_percentage))

        df_train = df.iloc[:train_end]
        df_validation = df.iloc[train_end:validation_end]
        df_test = df.iloc[validation_end:]

        return df_train, df_validation, df_test

    @staticmethod
    def get_x_y_from_df(
        df: pd.DataFrame,
        features: list[str],
        target: str,
    ) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        """Get X and Y from df."""
        return (
            df[features].to_numpy(),
            df[target].to_numpy(),
        )

    def __set_random_seeds(self) -> None:
        if self.random_seed is None:
            return

        self.model.set_random_seed(self.random_seed)
        random.seed(self.random_seed)
