"""Script for using the model on streamlit."""

import joblib
import pandas as pd
import streamlit as st

from utils.constants import THRESHOLD_AGATA, THRESHOLD_AMETISTA, THRESHOLD_TOPAZIO, Col
from utils.pede_passos_preprocessor import PedePassosPreprocessor

st.write("### Idade")
input_age = st.slider("Selecione a idade do aluno", 1, 30, 12)

st.write("### Fase")
input_level = st.slider("Selecione a fase do aluno", 0, 7, 0)

st.write("### Ingressante")
input_first_year = st.radio("O aluno é ingressante?", ["Sim", "Não"], index=0) == "Sim"

st.write("### IAA")
input_iaa = st.number_input(
    "Preencha o IAA do aluno",
    0.0,
    10.0,
    0.0,
    0.01,
    "%.2f",
    key="iaa",
)

st.write("### IAN")
input_ian = st.number_input(
    "Preencha o IAN do aluno",
    0.0,
    10.0,
    0.0,
    0.05,
    "%.2f",
    key="ian",
)

st.write("### IEG")
input_ieg = st.number_input(
    "Preencha o IEG do aluno",
    0.0,
    10.0,
    0.0,
    0.01,
    "%.2f",
    key="ieg",
)

if st.button("Enviar"):
    dump = joblib.load("models/random_forest_regressor_predict_student_inde.pkl")
    model = dump["model"]
    preprocessor: PedePassosPreprocessor = dump["preprocessor"]

    tranformed_input = (
        preprocessor.preprocess(
            df=pd.DataFrame(
                data={
                    Col.level: [input_level],
                    Col.iaa: [input_iaa],
                    Col.ian: [input_ian],
                    Col.age: [input_age],
                    Col.ieg: [input_ieg],
                    Col.inde: [0],
                    Col.first_year: [input_first_year],
                },
            ),
            feature_to_keep=dump["preprocessor_features"],
        )
        .drop(Col.inde, axis=1)
        .to_numpy()
    )

    pred_inde = model.predict(tranformed_input)[0] * 10
    if pred_inde > THRESHOLD_TOPAZIO:
        st.success(f"### Topázio! INDE previsto: {pred_inde:.2f}")
        st.balloons()
    elif pred_inde > THRESHOLD_AMETISTA:
        st.success(f"### Ametista! INDE previsto: {pred_inde:.2f}")
    elif pred_inde > THRESHOLD_AGATA:
        st.success(f"### Ágata! INDE previsto: {pred_inde:.2f}")
    else:
        st.error(f"### Quartzo! INDE previsto: {pred_inde:.2f}")
