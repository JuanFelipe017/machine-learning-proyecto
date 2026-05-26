"""Aplicacion web Streamlit - Clasificador de Spam (SVM).

Grupo 7 - Inteligencia Artificial I - Actividad 3.

Modos de entrada:
  1. Pegar texto de un email  -> se extraen las 57 features y se predice.
  2. Probar fila aleatoria del test set (UCI SpamBase) -> sanity check del modelo.

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

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Clasificador de Spam - Grupo 7",
    page_icon="✉",   # sobre Unicode (no requiere fuente de emoji)
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS para darle un look mas cuidado
st.markdown(
    """
    <style>
      /* tarjetas de metrica */
      .metric-card {
        background: linear-gradient(135deg, #1e2128 0%, #2a2e38 100%);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        border-left: 3px solid #c44e52;
      }
      .metric-name  { color: #9aa0a6; font-size: 0.78rem; text-transform: uppercase;
                      letter-spacing: 0.05em; margin-bottom: 2px; }
      .metric-value { color: #ffffff; font-size: 1.5rem; font-weight: 700;
                      line-height: 1; }
      .metric-hint  { color: #6f7682; font-size: 0.72rem; margin-top: 4px; }

      /* cabecera */
      .hero { background: linear-gradient(135deg, #2a3441 0%, #3d2a3a 100%);
              padding: 24px 28px; border-radius: 14px; margin-bottom: 18px;
              border: 1px solid #3a3f4a; }
      .hero h1 { color: #ffffff; margin: 0; font-size: 1.9rem; }
      .hero p  { color: #c7cad1; margin: 6px 0 0 0; font-size: 0.95rem; }

      /* resultado: HAM */
      .result-ham {
        background: linear-gradient(135deg, #1f3a2a 0%, #1a4a37 100%);
        border-left: 4px solid #3aa063; border-radius: 10px;
        padding: 16px 20px; margin: 10px 0;
      }
      /* resultado: SPAM */
      .result-spam {
        background: linear-gradient(135deg, #3a1f1f 0%, #4a1a1a 100%);
        border-left: 4px solid #d4574e; border-radius: 10px;
        padding: 16px 20px; margin: 10px 0;
      }
      .result-title { color: #ffffff; font-size: 1.4rem; font-weight: 700;
                      margin-bottom: 6px; }
      .result-detail { color: #c7cad1; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = ROOT / "models" / "modelo.pkl"
CARD_PATH = ROOT / "models" / "model_card.json"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

# Glosario corto en espanol para cada metrica
METRIC_HINTS = {
    "accuracy":  "% de aciertos totales",
    "precision": "de los marcados spam, cuantos lo eran",
    "recall":    "de los spam reales, cuantos detecto",
    "f1":        "balance entre precision y recall",
    "roc_auc":   "capacidad de separar las clases (1.0 = perfecto)",
}


# ============================================================
# HELPERS
# ============================================================
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
    """Caja de resultado con estilo (verde para HAM, rojo para SPAM)."""
    label = out["label_name"]
    proba = out["spam_probability"]
    score = out["decision_score"]
    css_class = "result-spam" if label == "spam" else "result-ham"
    title = (
        "PREDICCION: SPAM (correo no deseado)"
        if label == "spam"
        else "PREDICCION: HAM (correo legitimo)"
    )
    st.markdown(
        f"""
        <div class="{css_class}">
          <div class="result-title">{title}</div>
          <div class="result-detail">
            Probabilidad de spam: <b>{proba:.1%}</b> &nbsp;|&nbsp;
            Decision score: <b>{score:+.3f}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "**decision score** (distancia con signo al hiperplano del SVM): "
        "negativo = lado del ham, positivo = lado del spam; "
        "mayor magnitud = mas confianza. "
        "**spam probability** (probabilidad aproximada) = sigmoid del score, "
        "es solo una conversion visual."
    )


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### Clasificador de Spam")
st.sidebar.caption("Grupo 7 - Inteligencia Artificial I")
st.sidebar.markdown("---")

st.sidebar.markdown(
    "**Algoritmo**  \n"
    "SVM *(Support Vector Machine, modelo que traza la mejor frontera entre clases)* "
    "con kernel RBF *(Radial Basis Function, permite fronteras curvas)*"
)
st.sidebar.markdown(
    "**Preprocesamiento**  \n"
    "StandardScaler *(normaliza las features a media 0 y desviacion 1)*"
)
st.sidebar.markdown(
    "**Dataset**  \n"
    "UCI SpamBase - 4.601 emails reales de HP Labs"
)
st.sidebar.markdown("---")

card = get_card()
if card:
    st.sidebar.markdown("### Metricas en test")
    st.sidebar.caption("Sobre 921 emails que el modelo nunca vio en entrenamiento")
    m = card.get("metricas_test", {})
    for k, v in m.items():
        hint = METRIC_HINTS.get(k, "")
        st.sidebar.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-name">{k}</div>
              <div class="metric-value">{v:.4f}</div>
              <div class="metric-hint">{hint}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    fecha = card.get("fecha_entrenamiento", "?")
    hp = card.get("mejores_hiperparametros", {})
    st.sidebar.caption(
        f"Entrenado: `{fecha}`  \n"
        f"Hiperparametros: `C={hp.get('svm__C', '?')}`, "
        f"`gamma='{hp.get('svm__gamma', '?')}'`"
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Repo y notebook de entrenamiento:  \n`notebooks/01_training.ipynb`"
)


# ============================================================
# MAIN
# ============================================================
st.markdown(
    """
    <div class="hero">
      <h1>Clasificador de correos spam con SVM</h1>
      <p>Aplicacion web del <b>Grupo 7</b>. Entrenamos un modelo Support Vector Machine
      sobre 3.680 correos reales del dataset SpamBase (UCI) y alcanzamos cerca
      del <b>92 % de accuracy</b> sobre los 921 correos del test set.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Elige como quieres probar el modelo:",
    [
        "1. Pegar el texto de un email",
        "2. Probar con una fila aleatoria del test set",
    ],
    horizontal=True,
)
st.divider()

model = get_model()

# ---------------- MODO 1: TEXTO ----------------
if mode.startswith("1"):
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Pega el texto de un correo en ingles")
        st.caption(
            "Nota: el modelo aprendio con correos en ingles de 1999. Para clasificar "
            "como spam, el texto debe contener palabras tipicas como `free`, `money`, "
            "`credit`, `business`, `000`, junto con `!` y `$`."
        )
        default = (
            "FREE FREE FREE MONEY!!! Make MONEY fast with our BUSINESS opportunity!!!\n"
            "Your CREDIT is approved! Click REMOVE to unsubscribe. Order NOW!\n"
            "000 dollars in 30 days. 100 percent guaranteed!"
        )
        text = st.text_area("Email", value=default, height=240, label_visibility="collapsed")
        do_predict = st.button("Predecir", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Como funciona")
        st.markdown(
            "1. Tomamos el texto que pegues.\n"
            "2. Calculamos las **57 features** *(porcentajes de palabras y caracteres,\n"
            "   estadisticas de mayusculas)* que aprendio el modelo.\n"
            "3. El pipeline `StandardScaler -> SVM` decide la clase.\n"
            "4. Mostramos: clase + score + probabilidad."
        )

    if do_predict:
        feats = email_to_features(text)
        st.divider()
        render_result(predict_from_features(feats, MODEL_PATH))
        with st.expander("Ver las features que se extrajeron del texto (solo las no cero)"):
            row = feats.iloc[0]
            nz = row[row > 0].rename("valor").to_frame()
            st.dataframe(nz, use_container_width=True)
            st.caption(
                f"Total features extraidas: 57. Con valor distinto de cero: {len(nz)}."
            )

# ---------------- MODO 2: FILA DEL TEST ----------------
elif mode.startswith("2"):
    st.subheader("Probar con un correo real del test set")
    st.caption(
        "Tomamos una fila real del archivo `data/processed/test.csv` "
        "(921 correos que el modelo nunca vio durante el entrenamiento), "
        "ejecutamos la prediccion y comparamos con la etiqueta real."
    )

    test_df = get_test_set()
    if test_df.empty:
        st.warning(
            "No se encuentra `data/processed/test.csv`. Ejecuta el notebook "
            "`notebooks/01_training.ipynb` para regenerarlo."
        )
    else:
        col1, col2, col3 = st.columns(3)
        if "row_idx" not in st.session_state:
            st.session_state.row_idx = int(np.random.randint(len(test_df)))

        if col1.button("Fila aleatoria", use_container_width=True):
            st.session_state.row_idx = int(np.random.randint(len(test_df)))
        if col2.button("Tomar un SPAM", use_container_width=True):
            spam_idx = test_df.index[test_df["is_spam"] == 1].tolist()
            st.session_state.row_idx = int(np.random.choice(spam_idx))
        if col3.button("Tomar un HAM", use_container_width=True):
            ham_idx = test_df.index[test_df["is_spam"] == 0].tolist()
            st.session_state.row_idx = int(np.random.choice(ham_idx))

        idx = st.session_state.row_idx
        row = test_df.iloc[[idx]]
        truth = int(row["is_spam"].iloc[0])
        truth_name = "SPAM" if truth == 1 else "HAM"

        st.divider()
        col_a, col_b = st.columns(2)
        col_a.info(f"Fila numero **{idx}** del test set")
        col_b.info(f"Etiqueta real: **{truth_name}**")

        out = predict_from_features(row[FEATURE_COLUMNS], MODEL_PATH)
        render_result(out)

        if out["label"] == truth:
            st.success("RESULTADO: el modelo acerto.")
        else:
            st.error("RESULTADO: el modelo se equivoco en esta fila.")

        with st.expander("Ver las features de este correo (solo las no cero)"):
            r = row[FEATURE_COLUMNS].iloc[0]
            nz = r[r > 0].rename("valor").to_frame()
            st.dataframe(nz, use_container_width=True)
            st.caption(
                f"Total features: 57. Con valor distinto de cero: {len(nz)}. "
                "Como el modelo acierta el ~92 % de las veces, prueba varias filas "
                "y solo deberia equivocarse 1 de cada 12 aproximadamente."
            )
