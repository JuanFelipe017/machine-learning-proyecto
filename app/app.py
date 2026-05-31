"""Aplicacion web Streamlit - Clasificador de Spam (SVM).

Grupo 7 - Inteligencia Artificial I - Actividad 3.

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
    page_title="Clasificador de Spam",
    page_icon="✉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
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

      .hero { background: linear-gradient(135deg, #2a3441 0%, #3d2a3a 100%);
              padding: 24px 28px; border-radius: 14px; margin-bottom: 18px;
              border: 1px solid #3a3f4a; }
      .hero h1 { color: #ffffff; margin: 0; font-size: 1.9rem; }
      .hero p  { color: #c7cad1; margin: 6px 0 0 0; font-size: 0.95rem; }

      .result-ham {
        background: linear-gradient(135deg, #1f3a2a 0%, #1a4a37 100%);
        border-left: 4px solid #3aa063; border-radius: 10px;
        padding: 16px 20px; margin: 10px 0;
      }
      .result-spam {
        background: linear-gradient(135deg, #3a1f1f 0%, #4a1a1a 100%);
        border-left: 4px solid #d4574e; border-radius: 10px;
        padding: 16px 20px; margin: 10px 0;
      }
      .result-title { color: #ffffff; font-size: 1.4rem; font-weight: 700;
                      margin-bottom: 6px; }
      .result-detail { color: #c7cad1; font-size: 0.88rem; }

      /* Tabla de features con tooltip */
      .feat-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
      .feat-table th { color: #9aa0a6; text-align: left; padding: 6px 10px;
                       border-bottom: 1px solid #3a3f4a; font-weight: 600; }
      .feat-table td { padding: 5px 10px; border-bottom: 1px solid #2a2e38;
                       color: #c7cad1; }
      .feat-table tr:hover td { background: #2a2e38; }
      .feat-name { cursor: help; border-bottom: 1px dashed #6f7682; }
      .feat-val  { color: #ffffff; font-weight: 600; text-align: right; }
      .feat-desc { color: #6f7682; font-size: 0.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = ROOT / "models" / "modelo.pkl"
CARD_PATH  = ROOT / "models" / "model_card.json"
TEST_PATH  = ROOT / "data" / "processed" / "test.csv"

METRIC_HINTS = {
    "accuracy":  "% de aciertos totales",
    "precision": "de los marcados spam, cuantos lo eran",
    "recall":    "de los spam reales, cuantos detecto",
    "f1":        "balance entre precision y recall",
    "roc_auc":   "capacidad de separar las clases (1.0 = perfecto)",
}

# ============================================================
# Descripciones legibles de cada feature para el tooltip
# ============================================================
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # --- palabras ---
    "word_freq_make":       "Qué tan seguido aparece 'make' (hacer/fabricar)",
    "word_freq_address":    "Qué tan seguido aparece 'address' (dirección)",
    "word_freq_all":        "Qué tan seguido aparece 'all' (todo)",
    "word_freq_3d":         "Qué tan seguido aparece '3d' (tecnología 3D, común en spam)",
    "word_freq_our":        "Qué tan seguido aparece 'our' (nuestro) — muy común en spam",
    "word_freq_over":       "Qué tan seguido aparece 'over' (sobre/más de)",
    "word_freq_remove":     "Qué tan seguido aparece 'remove' (eliminar — típico en spam para darse de baja)",
    "word_freq_internet":   "Qué tan seguido aparece 'internet'",
    "word_freq_order":      "Qué tan seguido aparece 'order' (orden/pedido)",
    "word_freq_mail":       "Qué tan seguido aparece 'mail' (correo)",
    "word_freq_receive":    "Qué tan seguido aparece 'receive' (recibir)",
    "word_freq_will":       "Qué tan seguido aparece 'will' (voluntad/futuro)",
    "word_freq_people":     "Qué tan seguido aparece 'people' (gente)",
    "word_freq_report":     "Qué tan seguido aparece 'report' (reporte)",
    "word_freq_addresses":  "Qué tan seguido aparece 'addresses' (direcciones, plural)",
    "word_freq_free":       "Qué tan seguido aparece 'free' (gratis) — señal fuerte de spam",
    "word_freq_business":   "Qué tan seguido aparece 'business' (negocio) — común en spam comercial",
    "word_freq_email":      "Qué tan seguido aparece 'email'",
    "word_freq_you":        "Qué tan seguido aparece 'you' (tú/usted) — spam personaliza mucho el mensaje",
    "word_freq_credit":     "Qué tan seguido aparece 'credit' (crédito) — señal de spam financiero",
    "word_freq_your":       "Qué tan seguido aparece 'your' (tu/su)",
    "word_freq_font":       "Qué tan seguido aparece 'font' (fuente tipográfica)",
    "word_freq_000":        "Qué tan seguido aparece '000' (miles, como $1,000,000) — muy típico de spam",
    "word_freq_money":      "Qué tan seguido aparece 'money' (dinero) — señal fuerte de spam",
    "word_freq_hp":         "Qué tan seguido aparece 'hp' (Hewlett-Packard — ham en este dataset)",
    "word_freq_hpl":        "Qué tan seguido aparece 'hpl' (abreviatura interna de HP — casi siempre ham)",
    "word_freq_george":     "Qué tan seguido aparece 'george' (nombre del recopilador del dataset — casi siempre ham)",
    "word_freq_650":        "Qué tan seguido aparece '650' (código de área de HP Labs — casi siempre ham)",
    "word_freq_lab":        "Qué tan seguido aparece 'lab' (laboratorio)",
    "word_freq_labs":       "Qué tan seguido aparece 'labs' (laboratorios — típico de HP, por eso ham)",
    "word_freq_telnet":     "Qué tan seguido aparece 'telnet' (protocolo de red antiguo)",
    "word_freq_857":        "Qué tan seguido aparece '857' (otro código de área asociado a HP)",
    "word_freq_data":       "Qué tan seguido aparece 'data' (datos)",
    "word_freq_415":        "Qué tan seguido aparece '415' (código de área de San Francisco)",
    "word_freq_85":         "Qué tan seguido aparece '85'",
    "word_freq_technology": "Qué tan seguido aparece 'technology' (tecnología)",
    "word_freq_1999":       "Qué tan seguido aparece '1999' (año del dataset — aparece en spam de la época)",
    "word_freq_parts":      "Qué tan seguido aparece 'parts' (partes/piezas)",
    "word_freq_pm":         "Qué tan seguido aparece 'pm' (tarde/gestión de proyectos)",
    "word_freq_direct":     "Qué tan seguido aparece 'direct' (directo)",
    "word_freq_cs":         "Qué tan seguido aparece 'cs' (ciencias de la computación)",
    "word_freq_meeting":    "Qué tan seguido aparece 'meeting' (reunión — típico de ham corporativo)",
    "word_freq_original":   "Qué tan seguido aparece 'original'",
    "word_freq_project":    "Qué tan seguido aparece 'project' (proyecto — típico de ham)",
    "word_freq_re":         "Qué tan seguido aparece 're' (respuesta — Re: en asunto de email)",
    "word_freq_edu":        "Qué tan seguido aparece 'edu' (dominio educativo — casi siempre ham)",
    "word_freq_table":      "Qué tan seguido aparece 'table' (tabla)",
    "word_freq_conference": "Qué tan seguido aparece 'conference' (conferencia — típico de ham académico)",
    # --- caracteres ---
    "char_freq_;":  "Frecuencia del carácter ';' (punto y coma) en el texto",
    "char_freq_(":  "Frecuencia del carácter '(' (paréntesis de apertura) en el texto",
    "char_freq_[":  "Frecuencia del carácter '[' (corchete de apertura) en el texto",
    "char_freq_!":  "Frecuencia del '!' (exclamación) — señal fuerte de spam: ¡GRATIS! ¡AHORA!",
    "char_freq_$":  "Frecuencia del '$' (dólar) — señal fuerte de spam: '$1,000,000'",
    "char_freq_#":  "Frecuencia del '#' (numeral) en el texto",
    # --- mayúsculas ---
    "capital_run_length_average": "Longitud promedio de las rachas de letras MAYÚSCULAS consecutivas",
    "capital_run_length_longest": "Racha más larga de letras MAYÚSCULAS seguidas en el email (ej: 'GRATIS')",
    "capital_run_length_total":   "Total de letras MAYÚSCULAS en el email — el spam tiende a gritar",
}

def feature_label(col: str) -> str:
    """Nombre corto para la columna Feature (solo el identificador, sin prefijo)."""
    if col.startswith("word_freq_"):
        return col[len("word_freq_"):]
    if col.startswith("char_freq_"):
        return col[len("char_freq_"):]
    labels = {
        "capital_run_length_average": "mayúsculas (promedio)",
        "capital_run_length_longest": "mayúsculas (máximo)",
        "capital_run_length_total":   "mayúsculas (total)",
    }
    return labels.get(col, col)


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


SPAM_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80" width="110" height="73">
  <!-- lata cuerpo -->
  <rect x="10" y="18" width="100" height="50" rx="8" fill="#1a6bb5"/>
  <!-- tapa superior -->
  <ellipse cx="60" cy="18" rx="50" ry="10" fill="#2185d0"/>
  <!-- tapa inferior -->
  <ellipse cx="60" cy="68" rx="50" ry="10" fill="#1558a0"/>
  <!-- reflejo -->
  <rect x="15" y="22" width="18" height="40" rx="4" fill="#3a9de0" opacity="0.25"/>
  <!-- etiqueta fondo -->
  <rect x="10" y="30" width="100" height="28" fill="#f5c842"/>
  <!-- texto SPAM -->
  <text x="60" y="50" text-anchor="middle" font-family="Arial Black, sans-serif"
        font-size="18" font-weight="900" fill="#1a6bb5" letter-spacing="2">SPAM</text>
  <!-- linea decorativa -->
  <rect x="10" y="30" width="100" height="3" fill="#e0a800"/>
  <rect x="10" y="55" width="100" height="3" fill="#e0a800"/>
  <!-- anilla -->
  <ellipse cx="60" cy="12" rx="8" ry="3" fill="none" stroke="#aad4f5" stroke-width="2"/>
  <path d="M60 9 Q68 4 72 8" fill="none" stroke="#aad4f5" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

HAM_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80" width="110" height="73">
  <!-- cuerpo del jamón (pierna) -->
  <ellipse cx="58" cy="50" rx="45" ry="28" fill="#c0533a"/>
  <!-- parte más clara (grasa) -->
  <ellipse cx="58" cy="50" rx="35" ry="20" fill="#d4785e"/>
  <!-- centro rosado -->
  <ellipse cx="58" cy="50" rx="22" ry="13" fill="#e8967a"/>
  <!-- hueso -->
  <ellipse cx="58" cy="50" rx="7" ry="4" fill="#f5e6c8"/>
  <ellipse cx="58" cy="50" rx="3" ry="2" fill="#e8c98a"/>
  <!-- mango del hueso -->
  <rect x="90" y="46" width="22" height="8" rx="4" fill="#f5e6c8"/>
  <ellipse cx="112" cy="50" rx="7" ry="6" fill="#f5e6c8"/>
  <!-- brillo -->
  <ellipse cx="42" cy="40" rx="8" ry="4" fill="#e8967a" opacity="0.5"/>
  <!-- texto HAM -->
  <text x="52" y="54" text-anchor="middle" font-family="Arial Black, sans-serif"
        font-size="13" font-weight="900" fill="#7a1f0a" opacity="0.6">HAM</text>
</svg>
"""

def render_result(out: dict) -> None:
    label  = out["label_name"]
    proba  = out["spam_probability"]
    score  = out["decision_score"]
    is_spam = label == "spam"
    css    = "result-spam" if is_spam else "result-ham"
    title  = (
        "PREDICCION: SPAM (correo no deseado)"
        if is_spam
        else "PREDICCION: HAM (correo legítimo)"
    )
    image_svg = SPAM_SVG if is_spam else HAM_SVG

    st.markdown(
        f"""
        <div class="{css}" style="display:flex; align-items:center; gap:20px;">
          <div style="flex-shrink:0; opacity:0.92">{image_svg}</div>
          <div>
            <div class="result-title">{title}</div>
            <div class="result-detail">
              Probabilidad de spam: <b>{proba:.1%}</b> &nbsp;|&nbsp;
              Decision score: <b>{score:+.3f}</b>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_features_table(feat_series: pd.Series) -> None:
    """Tabla de features no-cero con nombre legible y descripción en tooltip."""
    nz = feat_series[feat_series > 0]
    if nz.empty:
        st.info("Ninguna feature con valor distinto de cero.")
        return

    rows_html = ""
    for col, val in nz.items():
        label = feature_label(col)
        desc  = FEATURE_DESCRIPTIONS.get(col, col)
        rows_html += (
            f"<tr>"
            f"<td>{label}</td>"
            f"<td class='feat-desc'>{desc}</td>"
            f"<td class='feat-val'>{val:.4f}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <table class="feat-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Qué mide</th>
              <th style="text-align:right">Valor (%)</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        <p style="color:#6f7682; font-size:0.78rem; margin-top:8px;">
          Total de features: 57 &nbsp;|&nbsp; Con valor distinto de cero: {len(nz)}
        </p>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### Clasificador de Spam")
st.sidebar.markdown("---")

card = get_card()
if card:
    st.sidebar.markdown("### Métricas en test")
    st.sidebar.caption("Sobre 921 emails que el modelo nunca vio")
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
    hp    = card.get("mejores_hiperparametros", {})
    st.sidebar.caption(
        f"Entrenado: `{fecha}`  \n"
        f"Hiperparámetros: `C={hp.get('svm__C', '?')}`, "
        f"`gamma='{hp.get('svm__gamma', '?')}'`"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("Notebook: `notebooks/01_training.ipynb`")


# ============================================================
# MAIN
# ============================================================
st.markdown(
    """
    <div class="hero">
      <h1>Clasificador de correos spam con SVM</h1>
      <p>SVM (kernel RBF) entrenado sobre 3.680 correos reales
      del dataset SpamBase (UCI) — <b>92 % de accuracy</b> en test.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["✉ Pegar texto de un email", "🎲 Probar con el test set"])

model = get_model()

# ============================================================
# TAB 1: TEXTO
# ============================================================
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Pega el texto de un correo en inglés")
        st.caption(
            "El modelo aprendió con emails de 1999. Para que clasifique como spam, "
            "incluye palabras como `free`, `money`, `credit`, `!` o `$`."
        )
        default = (
            "FREE FREE FREE MONEY!!! Make MONEY fast with our BUSINESS opportunity!!!\n"
            "Your CREDIT is approved! Click REMOVE to unsubscribe. Order NOW!\n"
            "000 dollars in 30 days. 100 percent guaranteed!"
        )
        text = st.text_area("Email", value=default, height=200, label_visibility="collapsed")
        do_predict = st.button("Predecir", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Cómo funciona")
        st.markdown(
            "1. Se extraen **57 features** del texto (frecuencias de palabras, caracteres y mayúsculas).\n"
            "2. El pipeline `StandardScaler → SVM` decide la clase.\n"
            "3. Se muestra la clase, score y probabilidad."
        )

    if do_predict:
        feats = email_to_features(text)
        st.divider()
        render_result(predict_from_features(feats, MODEL_PATH))
        with st.expander("Ver las señales que detectó el modelo (features no-cero)"):
            render_features_table(feats.iloc[0])

# ============================================================
# TAB 2: FILA DEL TEST SET
# ============================================================
with tab2:
    st.subheader("Probar con un correo real del test set")
    st.caption(
        "Selecciona una fila de los 921 correos reales que el modelo nunca vio "
        "durante el entrenamiento y compara la predicción con la etiqueta verdadera."
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

        idx   = st.session_state.row_idx
        row   = test_df.iloc[[idx]]
        truth = int(row["is_spam"].iloc[0])
        truth_name = "SPAM" if truth == 1 else "HAM"

        st.divider()
        col_a, col_b = st.columns(2)
        col_a.info(f"Fila **{idx}** del test set")
        col_b.info(f"Etiqueta real: **{truth_name}**")

        out = predict_from_features(row[FEATURE_COLUMNS], MODEL_PATH)
        render_result(out)

        if out["label"] == truth:
            st.success("✓ El modelo acertó.")
        else:
            st.error("✗ El modelo se equivocó en esta fila.")

        with st.expander("Ver las señales detectadas en este correo (features no-cero)"):
            render_features_table(row[FEATURE_COLUMNS].iloc[0])