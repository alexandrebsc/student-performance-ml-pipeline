"""Script for the module PedePassosLoader."""

import pandas as pd

from utils.constants import DATA_FOLDER_PATH, Col


class PedePassosLoader:
    """Module responsible to make the Pede Passos students historical data readable."""

    def __init__(self) -> None:
        """Initialize a PedePassosLoader."""

    def load(self) -> pd.DataFrame:
        """Load the stundents historical data.

        :return: stundents historical data
        :rtype: DataFrame
        """
        df = self._get_students_df()
        return self._data_cleaning(df)

    def _get_students_df(self) -> pd.DataFrame:
        file_path = f"{DATA_FOLDER_PATH}/pede_passos.csv"
        return pd.read_csv(
            file_path,
            encoding="utf-8",
            delimiter=";",
            engine="python",
        )

    def _data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
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

        df = self._drop_corrupted_data(df)
        df = self._fill_age_data(df)
        df = self._normalize_coloumn_ingressante(df)
        df = self._normalize_column_fase(df)
        df = self._cast_values(df)
        df = self._unpivot_by_year(df)
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

    def _drop_corrupted_data(self, df: pd.DataFrame) -> pd.DataFrame:
        invalid_age_mask = (
            pd.to_numeric(df["IDADE_ALUNO_2020"], errors="coerce").isna()
            & df["IDADE_ALUNO_2020"].notna()
        )
        null_mask = df.astype(str).eq("#NULO!").any(axis=1)
        return df[~(invalid_age_mask | null_mask)].reset_index(drop=True)

    def _fill_age_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["IDADE_2020"] = pd.to_numeric(df["IDADE_ALUNO_2020"], downcast="integer")
        average_age = 12
        df["IDADE_2020"] = df["IDADE_2020"].fillna(average_age)

        df["IDADE_2021"] = df["IDADE_2020"] + 1
        df["IDADE_2022"] = df["IDADE_2020"] + 2
        return df

    def _normalize_coloumn_ingressante(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _normalize_column_fase(self, df: pd.DataFrame) -> pd.DataFrame:
        df["FASE_2020"] = pd.to_numeric(
            df["FASE_TURMA_2020"].str[0],
            downcast="integer",
        )
        df = df.drop("FASE_TURMA_2020", axis=1)
        df["FASE_2021"] = pd.to_numeric(df["FASE_2021"], downcast="integer")
        df["FASE_2022"] = pd.to_numeric(df["FASE_2022"], downcast="integer")
        return df

    def _cast_values(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _unpivot_by_year(self, df: pd.DataFrame) -> pd.DataFrame:
        years = ["2020", "2021", "2022"]
        columns = [
            "IDADE",
            "INGRESSANTE",
            "FASE",
            "INDE",
            "IAA",
            "IAN",
            "IEG",
            "IDA",
            "IPS",
            "IPP",
            "IPV",
        ]

        df = df.reset_index()
        df = pd.melt(
            df,
            id_vars="NOME",
            value_vars=[f"{metric}_{year}" for metric in columns for year in years],
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
