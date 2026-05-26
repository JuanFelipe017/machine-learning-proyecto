"""Aplicacion web Streamlit - Clasificador de Spam (SVM).

Grupo 7 - Inteligencia Artificial I - Actividad 3.

Modos de entrada:
  1. Pegar texto de un email  -> se extraen las 57 features y se predice.
  2. Probar fila aleatoria del test set (UCI SpamBase)  -> sanity check del modelo.
  3. Editar manualmente las features mas influyentes.

Ejecutar:  streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing import FEATURE_COLUMNS, email_to_features
from src.predict import load_model, predict_from_features

st.set_page_config(
    page_title="Spam Classifier - SVM (Grupo 7)",
    page_icon="MAIL",
    layout="wide",
)

MODEL_PATH = ROOT / "models" / "modelo.pkl"
CARD_PATH = ROOT / "models" / "model_card.json"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_card() -> dict:
    if CARD_PATH.exists():
        return json.loads(CARD_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def get_test_set() -> pd.DataFrame:
    if TEST_PATH.exists():
        return pd.read_csv(TEST_PATH)
    return pd.DataFrame()


def render_result(out: dict) -> None:
    label = out["label_name"]
    proba = out["spam_probability"]
    score = out["decision_score"]
    if label == "spam":
        st.error(f"PREDICCION: SPAM   (prob ~ {proba:.2%}  |  score = {score:+.3f})")
    else:
        st.success(f"PREDICCION: HAM   (prob spam ~ {proba:.2%}  |  score = {score:+.3f})")
    st.caption(
        "score = SVC.decision_function (signo + magnitud de la separacion al hiperplano). "
        "prob ~ sigmoid(score), aproximacion para mostrar confianza."
    )


# ---------- Sidebar ----------
st.sidebar.title("Clasificador de Spam")
st.sidebar.markdown("**Algoritmo:** SVM (RBF) + StandardScaler")
st.sidebar.markdown("**Dataset:** UCI SpamBase (4.601 emails reales)")

card = get_card()
if card:
    st.sidebar.subheader("Metricas del modelo (test)")
    m = card.get("metricas_test", {})
    for k, v in m.items():
        st.sidebar.write(f"- {k}: **{v:.4f}**")
    st.sidebar.caption(f"Entrenado: {card.get('fecha_entrenamiento', '?')}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Repositorio y notebook de entrenamiento en `notebooks/01_training.ipynb`."
)

# ---------- Main ----------
st.title("Clasificador de Spam con Support Vector Machines")
st.write(
    "Aplicacion web del **Grupo 7**. El modelo SVM fue entrenado sobre 3.680 emails "
    "reales del dataset SpamBase (UCI) y alcanza ~92% de accuracy en test."
)

mode = st.radio(
    "Selecciona el modo de entrada:",
    [
        "1. Pegar texto de un email",
        "2. Probar con una fila aleatoria del test set",
    ],
    horizontal=False,
)

model = get_model()

# ----- Modo 1: texto -----
if mode.startswith("1"):
    st.subheader("Pega el texto de un email")
    st.caption(
        "Nota: el modelo aprendio el vocabulario del corpus HP Labs 1999 "
        "(palabras como 'free', 'money', 'credit', 'business', '000', '!', '$' "
        "elevan la probabilidad de spam; palabras como 'hp', 'george', 'lab' "
        "son senales de ham)."
    )

    default = (
        "FREE FREE FREE MONEY!!! Make MONEY fast with our BUSINESS opportunity!!!\n"
        "Your CREDIT is approved! Click REMOVE to unsubscribe. Order NOW!\n"
        "000 dollars in 30 days. 100 percent guaranteed!"
    )
    text = st.text_area("Email", value=default, height=220)

    if st.button("Predecir", type="primary"):
        feats = email_to_features(text)
        out = predict_from_features(feats, MODEL_PATH)
        render_result(out)
        with st.expander("Ver features extraidas (no cero)"):
            row = feats.iloc[0]
            st.dataframe(row[row > 0].rename("valor").to_frame())

# ----- Modo 2: fila aleatoria del test -----
elif mode.startswith("2"):
    st.subheader("Probar con una fila aleatoria del test set")
    test_df = get_test_set()
    if test_df.empty:
        st.warning(
            "No se encuentra `data/processed/test.csv`. Ejecuta primero el notebook "
            "`notebooks/01_training.ipynb`."
        )
    else:
        col1, col2, col3 = st.columns(3)
        if "row_idx" not in st.session_state:
            st.session_state.row_idx = int(np.random.randint(len(test_df)))

        if col1.button("Fila aleatoria"):
            st.session_state.row_idx = int(np.random.randint(len(test_df)))
        if col2.button("Spam aleatorio"):
            spam_idx = test_df.index[test_df["is_spam"] == 1].tolist()
            st.session_state.row_idx = int(np.random.choice(spam_idx))
        if col3.button("Ham aleatorio"):
            ham_idx = test_df.index[test_df["is_spam"] == 0].tolist()
            st.session_state.row_idx = int(np.random.choice(ham_idx))

        idx = st.session_state.row_idx
        row = test_df.iloc[[idx]]
        truth = int(row["is_spam"].iloc[0])
        truth_name = "SPAM" if truth == 1 else "HAM"
        st.info(f"Fila {idx}  |  etiqueta real: **{truth_name}**")

        out = predict_from_features(row[FEATURE_COLUMNS], MODEL_PATH)
        render_result(out)
        match = "CORRECTO" if out["label"] == truth else "ERROR"
        st.write(f"Resultado: **{match}**")

        with st.expander("Ver features de esta fila (no cero)"):
            r = row[FEATURE_COLUMNS].iloc[0]
            st.dataframe(r[r > 0].rename("valor").to_frame())

