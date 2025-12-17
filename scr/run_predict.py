# ruff: noqa: D103, T201
"""Batch prediction to test the model on some sample data."""

import joblib
from utils.constants import Col
from utils.etl import get_clean_df, pre_processing

TRAIN_2020_STUDENTS = [
    "ALUNO-295",
    "ALUNO-708",
    "ALUNO-935",
    "ALUNO-460",
    "ALUNO-450",
    "ALUNO-1067",
    "ALUNO-1222",
]


VALIDATION_2022_STUDENTS = [
    "ALUNO-608",
    "ALUNO-1135",
    "ALUNO-856",
    "ALUNO-405",
    "ALUNO-664",
    "ALUNO-180",
    "ALUNO-967",
    "ALUNO-288",
    "ALUNO-868",
]


def main() -> None:
    model = joblib.load("random_forest_regressor_predict_student_inde.pkl")

    df = get_clean_df()
    df_filtered = df[(df[Col.name].isin(TRAIN_2020_STUDENTS)) & (df[Col.year] == 2020)]  # noqa: PLR2004
    df_processed = pre_processing(df_filtered)

    x = df_processed[
        [Col.age, Col.level, Col.ian, Col.iaa, Col.ieg, Col.first_year]
    ].to_numpy()

    predictions = model.predict(x) * 10
    df_filtered["predicted_inde"] = predictions

    print("\nActual vs Predicted:")
    print("=" * 50)
    df_filtered.set_index(Col.name, inplace=True, drop=True)
    print(df_filtered[[Col.year, Col.inde, "predicted_inde"]].to_string())


if __name__ == "__main__":
    main()
