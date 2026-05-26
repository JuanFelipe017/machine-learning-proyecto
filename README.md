# Clasificador de Spam con Support Vector Machines (SVM)

**Grupo 7 — Inteligencia Artificial I — Actividad 3 — Fundación Universitaria Los Libertadores**

## Descripción

Aplicación web que clasifica correos electrónicos como **spam** o **ham** usando un
modelo Support Vector Machines (kernel RBF) entrenado sobre el dataset **SpamBase**
del UCI Machine Learning Repository (4.601 emails reales recolectados por
Hewlett-Packard Labs).

> **Nota sobre el dataset:** el README del grupo apuntaba al dataset Kaggle
> `purusinghvi/email-spam-classification-dataset`. Como ese repositorio requiere
> autenticación de Kaggle, en esta entrega usamos **UCI SpamBase**, que resuelve el
> mismo problema (clasificación binaria de spam de email), es totalmente público y
> cumple con todos los requisitos del enunciado (>500 registros, >5 features, datos
> reales, problema definido). El pipeline es independiente del origen del CSV:
> bastaría con sustituir el archivo en `data/raw/` y ajustar `src/preprocessing.py`
> para usar el dataset original más adelante.

## Demostración

Aplicación local en http://localhost:8501 una vez ejecutado `streamlit run app/app.py`.
Despliegue en la nube pendiente para la Entrega 2.

## Algoritmo utilizado

- **Algoritmo:** Support Vector Machines (SVM) con kernel RBF, `class_weight='balanced'`.
- **Preprocesamiento:** `StandardScaler` (SVM es muy sensible a la escala de las features).
- **Selección de hiperparámetros:** `GridSearchCV` con CV de 5 folds, optimizando F1
  sobre `C ∈ {0.5, 1, 3, 10}` y `gamma ∈ {scale, 0.01, 0.05}`.
- **Por qué SVM es apropiado:** las 57 features de SpamBase son densas y de magnitudes
  muy distintas (frecuencias en % vs. longitudes en caracteres). SVM con RBF maneja bien
  fronteras no lineales en este espacio sin necesidad de feature engineering adicional,
  y `class_weight='balanced'` compensa el ligero desbalance (~39 % spam, ~61 % ham).

### Métricas de desempeño (test, 921 emails)

| Métrica   | Valor  |
|-----------|--------|
| accuracy  | 0.9207 |
| precision | 0.9119 |
| recall    | 0.8843 |
| F1        | 0.8979 |
| ROC-AUC   | 0.9698 |

Mejores hiperparámetros encontrados: `C = 10`, `gamma = 'scale'`.

## Dataset

- **Fuente:** UCI Machine Learning Repository — SpamBase
  (<https://archive.ics.uci.edu/dataset/94/spambase>).
- **Tamaño:** 4.601 emails reales (3.680 train / 921 test, split estratificado 80/20).
- **Features (57):**
  - 48 frecuencias de palabra (`word_freq_make`, `word_freq_free`, `word_freq_money`,
    `word_freq_credit`, `word_freq_000`, `word_freq_business`, `word_freq_hp`, …).
  - 6 frecuencias de carácter (`char_freq_!`, `char_freq_$`, `char_freq_#`, …).
  - 3 estadísticas de secuencias de mayúsculas (promedio / máximo / total).
- **Target:** `is_spam ∈ {0, 1}` (1 = spam, 0 = ham).

## Estructura del proyecto

```
Grupo7 - SVM/
├── README.md                  # este archivo
├── requirements.txt           # dependencias Python
├── .gitignore
│
├── data/
│   ├── raw/                   # spambase.data, spambase.names, spambase.DOCUMENTATION
│   └── processed/             # train.csv y test.csv (generados por el notebook)
│
├── notebooks/
│   ├── 01_training.ipynb      # carga → EDA → preprocesa → entrena → evalúa → serializa
│   └── build_notebook.py      # script que genera 01_training.ipynb (reproducible)
│
├── models/
│   ├── modelo.pkl             # pipeline serializado (StandardScaler + SVC)
│   └── model_card.json        # metadatos: hiperparámetros, métricas, versión sklearn, fecha
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # load_raw(), split_xy(), email_to_features()
│   └── predict.py             # load_model(), predict_from_features(), predict_from_email()
│
├── app/
│   ├── app.py                 # aplicación Streamlit
│   ├── templates/             # (vacío — Streamlit no usa templates)
│   └── static/                # (vacío)
│
└── docs/
    ├── evidence_local_test.txt  # evidencia textual de la prueba local
    ├── streamlit_homepage.html  # HTML servido por Streamlit en localhost
    ├── streamlit_run.log        # log del arranque de Streamlit
    └── images/                  # plots generados por el notebook
```

## Instalación local

### Requisitos

- Python 3.11+
- pip

### Pasos

```bash
git clone <url-del-repo>
cd "Grupo7 - SVM"
pip install -r requirements.txt
```

### Re-entrenar el modelo (opcional — el .pkl ya está incluido)

```bash
jupyter nbconvert --to notebook --execute notebooks/01_training.ipynb \
    --output 01_training.ipynb --ExecutePreprocessor.timeout=600
```

Esto regenera `models/modelo.pkl`, `models/model_card.json` y
`data/processed/{train,test}.csv`.

### Ejecutar la aplicación

```bash
streamlit run app/app.py
```

La app queda disponible en <http://localhost:8501>.

## Uso de la aplicación

La app ofrece **tres modos** de entrada:

1. **Pegar texto de un email** — se extraen las 57 features de SpamBase desde el
   texto crudo y se predice. *Nota:* el modelo aprendió el vocabulario HP Labs 1999,
   así que para que el texto cuente como spam debe contener palabras como `free`,
   `money`, `credit`, `business`, `000`, `!`, `$` con cierta densidad.
2. **Fila aleatoria del test set** — toma una fila real de `data/processed/test.csv`,
   muestra la etiqueta verdadera y la predicción del modelo (útil para demostrar el
   ~92 % de accuracy en vivo).
3. **Editar features manualmente** — sliders para las 11 features más influyentes;
   el resto queda en 0. Útil para mostrar cómo cambia la predicción al subir las
   features asociadas a spam.

## Buenas prácticas de MLOps aplicadas en esta entrega

- **Pipeline serializado completo** (`StandardScaler + SVC`) en un único `.pkl`,
  para evitar desincronización entre el scaler de entrenamiento y el de inferencia.
- **Model card** (`models/model_card.json`) con hiperparámetros, métricas, versión
  de scikit-learn y fecha de entrenamiento.
- **Reproducibilidad**: el notebook se genera desde `notebooks/build_notebook.py`
  y se ejecuta con `random_state=42` en split y modelo.
- **Separación de capas**: lógica de datos (`src/preprocessing.py`), de inferencia
  (`src/predict.py`) y de UI (`app/app.py`) en módulos independientes.
- **`requirements.txt`** con versiones fijadas.

## Autores

- Grupo 7 — Inteligencia Artificial I
- Algoritmo asignado: **Support Vector Machines (SVM)**

## Licencia

MIT License
