"""Preprocesamiento del dataset SpamBase (UCI).

Lee el archivo CSV crudo (sin cabeceras) y devuelve un DataFrame con nombres
de columnas correctos, además de utilidades para construir un vector de
features a partir del texto crudo de un email (usado por la app web).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

WORD_FEATURES = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
    "cs", "meeting", "original", "project", "re", "edu", "table", "conference",
]

CHAR_FEATURES = [";", "(", "[", "!", "$", "#"]

FEATURE_COLUMNS: list[str] = (
    [f"word_freq_{w}" for w in WORD_FEATURES]
    + [f"char_freq_{c}" for c in CHAR_FEATURES]
    + [
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]
)

TARGET_COLUMN = "is_spam"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Carga el CSV crudo sin cabecera y le pone los nombres correctos."""
    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def email_to_features(text: str) -> pd.DataFrame:
    """Convierte un email crudo en el vector de 57 features de SpamBase.

    Replica la definición original del dataset (UCI 1999): frecuencias de
    palabras y caracteres en %, más estadísticas de secuencias de mayúsculas.
    """
    text = text or ""
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    total_words = max(len(tokens), 1)
    total_chars = max(len(text), 1)

    lower_tokens = [t.lower() for t in tokens]
    token_counts: dict[str, int] = {}
    for tok in lower_tokens:
        token_counts[tok] = token_counts.get(tok, 0) + 1

    row: dict[str, float] = {}
    for w in WORD_FEATURES:
        row[f"word_freq_{w}"] = 100.0 * token_counts.get(w, 0) / total_words
    for c in CHAR_FEATURES:
        row[f"char_freq_{c}"] = 100.0 * text.count(c) / total_chars

    runs: list[int] = []
    current = 0
    for ch in text:
        if ch.isupper():
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)

    if runs:
        row["capital_run_length_average"] = float(np.mean(runs))
        row["capital_run_length_longest"] = float(max(runs))
        row["capital_run_length_total"] = float(sum(runs))
    else:
        row["capital_run_length_average"] = 0.0
        row["capital_run_length_longest"] = 0.0
        row["capital_run_length_total"] = 0.0

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
