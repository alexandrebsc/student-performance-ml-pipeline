# ruff: noqa: D103, T201
"""Script for training the model."""

import random

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle
from utils.constants import RANDOM_SEED, Col
from utils.etl import get_clean_df, pre_processing


def main() -> None:
    set_random_seeds(RANDOM_SEED)
    df = get_clean_df()

    df_train, df_test = get_df_train_and_test(df)

    # intentionally commented, not optimal
    # df_train = data_augmentation(df_train) # noqa: ERA001

    df_train, df_test, df_complete = (
        pre_processing(df_train),
        pre_processing(df_test),
        pre_processing(df),
    )

    train_and_dump_model(df_train, df_test, df_complete)


def set_random_seeds(seed_value: int) -> None:
    np.random.default_rng(seed_value)
    random.seed(seed_value)


def get_df_train_and_test(
    df: pd.DataFrame,
    test_percentage: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = shuffle(df)
    df_len = len(df)
    head_count = int(df_len * test_percentage)
    return df.head(df_len - head_count), df.tail(head_count)


def train_and_dump_model(  # noqa: C901
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_complete: pd.DataFrame,
) -> None:
    def get_model_results(
        model: RandomForestRegressor,
        x: np.typing.NDArray,
        y: np.typing.NDArray,
    ) -> tuple[float, float, float, float]:
        y_pred = model.predict(x)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        max_diff = np.max(np.abs(y_pred - y))
        return mae, mse, r2, max_diff

    def print_results(  # noqa: PLR0913
        header: str,
        n_estimators: int,
        max_features: str,
        max_depth: int,
        mae: float,
        mse: float,
        r2: float,
        max_diff: float,
    ) -> None:
        print(
            f"{header} -> {n_estimators:03d}, {max_features}, {max_depth:02d},",
            f"{mae:0.3f}, {mse:0.4f}, {r2:0.3f}, {max_diff:0.3f}",
        )

    def grid_search() -> None:
        best_r2 = 0
        for n_estimators in [25, 50, 100]:
            for max_features in ["sqrt"]:
                for max_depth in [7, 8, 9, 10, 11, 12]:
                    model_config = {
                        "n_estimators": n_estimators,
                        "max_features": max_features,
                        "max_depth": max_depth,
                    }
                    model = RandomForestRegressor(
                        random_state=RANDOM_SEED,
                        **model_config,
                    )
                    model.fit(x_train, y_train)

                    results = get_model_results(model, x_train, y_train)
                    print_results("train", *model_config.values(), *results)

                    results = get_model_results(model, x_test, y_test)
                    print_results("test", *model_config.values(), *results)

                    results = get_model_results(model, x_complete, y_complete)
                    print_results("compt", *model_config.values(), *results)
                    print()

                    r2 = results[2]
                    if r2 > best_r2:
                        best_r2 = r2
                        best_result = [n_estimators, max_features, max_depth]

        print(f"Best results {best_result}")

    def train_final_model() -> RandomForestRegressor:
        model = RandomForestRegressor(
            n_estimators=50,
            max_features="sqrt",
            max_depth=11,
            random_state=RANDOM_SEED,
        )
        model.fit(x_train, y_train)

        for log in [
            ("Comp", (x_complete, y_complete)),
            ("Test", (x_test, y_test)),
            ("Train", (x_train, y_train)),
        ]:
            mae, mse, r2, max_diff = get_model_results(model, *log[1])
            print(
                f"{log[0]}  -> MAE: {mae:0.3f}; MSE: {mse:0.4f}; R2: {r2:0.3f};",
                f" MAX DIFF: {max_diff:0.3f}",
            )

        return model

    def get_x_y_from_df(
        df: pd.DataFrame,
    ) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        return (
            df[
                [Col.age, Col.level, Col.ian, Col.iaa, Col.ieg, Col.first_year]
            ].to_numpy(),
            df[Col.inde].to_numpy(),
        )

    x_train, y_train = get_x_y_from_df(df_train)
    x_test, y_test = get_x_y_from_df(df_test)
    x_complete, y_complete = get_x_y_from_df(df_complete)

    grid_search()
    model = train_final_model()
    joblib.dump(model, "random_forest_regressor_predict_student_inde.pkl")


if __name__ == "__main__":
    main()
