"""Script for the module Evaluator."""

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator:
    """Module responsible for evaluating the model predictions."""

    def __init__(self) -> None:
        """Initialize an Evaluator."""
        self.log = logging.getLogger(__name__)

    def evaluate(
        self,
        y_train: NDArray,
        y_actual: NDArray,
        y_predicted: NDArray,
        header: str = "",
    ) -> None:
        """Log model evaluation with baseline comparison.

        :param y_train: Model Y values used for training
        :type y_train: NDArray
        :param y_actual: Model actual Y values
        :type y_actual: NDArray
        :param y_predicted: Model predicted values for Y
        :type y_predicted: NDArray
        :param header: Model name used for logging
        :type header: str
        """
        self.performance_metrics_info(
            y_actual,
            y_predicted,
            header,
            complete_log=True,
        )

        self.evaluate_mean_baseline(y_train, y_actual, header)

    def evaluate_mean_baseline(
        self,
        y_train: NDArray,
        y_actual: NDArray,
        header: str = "",
    ) -> tuple[float, float, float, float]:
        """Evaluate a mean baseline model.

        :param y_train: Model Y values used for training
        :type y_train: NDArray
        :param y_actual: Model actual Y values
        :type y_actual: NDArray
        :param header: Model name used for logging
        :type header: str

        :return: In order return: MSE, MAE, R2, MAX_DIFF
        :rtype: tuple[float, float, float, float]
        """
        mean_predictions = np.full(shape=y_actual.shape, fill_value=y_train.mean())

        return self.performance_metrics_info(
            y_actual=y_actual,
            y_predicted=mean_predictions,
            model_name=f"{header} Mean Baseline",
            complete_log=True,
        )

    def performance_metrics_info(
        self,
        y_actual: NDArray,
        y_predicted: NDArray,
        model_name: str | None = None,
        *,
        complete_log: bool = False,
        compact_log: bool = False,
    ) -> tuple[float, float, float, float]:
        """Get performance metrics for the model.

        :param y_actual: Actual values (optional, uses stored values if None)
        :type y_actual: NDArray | None
        :param y_predicted: Predicted values (optional, uses stored values if None)
        :type y_predicted: NDArray | None
        :param model_name: Name of the model for display purposes
        :type model_name: str
        :param complete_log: If True, log complete metrics details
        :type complete_log: bool
        :param compact_log: If True, log compact metrics details.
            Doesn't log if complete_log is set True
        :type compact_log: bool

        :return: In order return: MSE, MAE, R2, MAX_DIFF
        :rtype: tuple[float, float, float, float]
        """
        mae = mean_absolute_error(y_actual, y_predicted)
        mse = mean_squared_error(y_actual, y_predicted)
        r2 = r2_score(y_actual, y_predicted)
        max_diff = np.max(np.abs(y_predicted - y_actual))

        if complete_log:
            self.log.info("\n%s Performance Metrics:", model_name)
            self.log.info("Mean Squared Error (MSE): %.6f", mse)
            self.log.info("Mean Absolute Error (MAE): %.6f", mae)
            self.log.info("R-squared (R2) Score: %.6f", r2)
        elif compact_log:
            self.log.debug(
                "%s \t-> MAE: %0.3f; MSE: %0.4f; R2: %0.3f; MAX DIFF: %0.3f",
                model_name,
                mae,
                mse,
                r2,
                max_diff,
            )

        return mse, mae, r2, max_diff
