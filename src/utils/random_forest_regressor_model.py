"""Script for the module RandomForestRegressorModel."""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

from utils.evaluator import Evaluator


class RandomForestRegressorModel:
    """Module responsible for training a RandomForestRegressor model."""

    def __init__(
        self,
        param_grid_dict: dict | None = None,
    ) -> None:
        """Initialize a configured RandomForestRegressorModel.

        :param features: List of features used for training.
        :type features: list[str]
        :param param_grid_dict: Parameters grid used for training.
        :type param_grid_dict: ParameterGrid
        """
        self.log = logging.getLogger(__name__)
        self.model: RandomForestRegressor | None = None

        if param_grid_dict is None:
            self.param_grid_dict = {
                "n_estimators": [25, 50, 100],
                "max_features": ["sqrt"],
                "max_depth": [7, 8, 9, 10, 11, 12],
            }
        else:
            self.param_grid_dict = dict(param_grid_dict)

        self.random_seed = None

    def set_random_seed(
        self,
        seed: int | None = None,
    ) -> None:
        """Set `random_seed` value."""
        self.random_seed = seed

    def fit_by_grid_search(
        self,
        df_train: pd.DataFrame,
        df_validation: pd.DataFrame,
        evaluator: Evaluator,
        get_x_y_from_df: Callable[
            [pd.DataFrame],
            tuple[np.typing.NDArray, np.typing.NDArray],
        ],
    ) -> None:
        """Search for and fit for the best hyperparams for the model.

        :param df_train: Training Dataframe
        :type df_train: pd.DataFrame
        :param df_validation: Validation Dataframe
        :type df_validation: pd.DataFrame
        :param evaluator: Evaluator object with performance_metrics_info implemented
        :type evaluator: Evaluator
        :param get_x_y_from_df: Callable to get features (X) and target (Y)
        :type get_x_y_from_df: Callable[
            [pd.DataFrame], tuple[np.typing.NDArray, np.typing.NDArray]
        ]
        """
        x_train, y_train = get_x_y_from_df(df_train)
        x_validation, y_validation = get_x_y_from_df(df_validation)

        best_r2 = float("-inf")
        best_param = None
        for param in ParameterGrid(self.param_grid_dict):
            model = RandomForestRegressor(
                random_state=self.random_seed,
                **param,
            )
            model.fit(x_train, y_train)

            evaluator.performance_metrics_info(
                y_actual=y_train,
                y_predicted=model.predict(x_train),
                model_name=f"train [{param}]",
                compact_log=True,
            )

            _, _, r2, _ = evaluator.performance_metrics_info(
                y_actual=y_validation,
                y_predicted=model.predict(x_validation),
                model_name=f"validation [{param}]",
                compact_log=True,
            )

            if r2 > best_r2:
                best_r2 = r2
                best_param = param
                best_model = model

        if best_param is None:
            msg = "Grid search failed to find any valid model."
            raise RuntimeError(msg)

        self.log.debug("Best results %s", best_param)
        self.best_param = best_param
        self.model = best_model

    def get_model(self) -> RandomForestRegressor:
        """Return the model."""
        if self.model is None:
            msg = "Model has not been trained yet. Must fit() or grid_search() first."
            raise RuntimeError(msg)

        return self.model
