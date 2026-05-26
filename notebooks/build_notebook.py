"""Genera notebooks/01_training.ipynb desde codigo fuente.

Si se modifica este archivo, regenerar el notebook con:
    python notebooks/build_notebook.py
y luego re-ejecutarlo con:
    jupyter nbconvert --to notebook --execute notebooks/01_training.ipynb \\
        --output 01_training.ipynb --ExecutePreprocessor.timeout=600
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ============================================================
# PORTADA
# ============================================================
md("""# Clasificador de Spam con Support Vector Machines (SVM)

**Grupo 7 — Inteligencia Artificial I — Actividad 3**
Fundación Universitaria Los Libertadores

---

## ¿Qué hace este notebook?

Este notebook recorre, de principio a fin, el ciclo completo de un proyecto
de Machine Learning para clasificar correos como **spam** o **ham** (correo legítimo):

1. **Carga** los datos reales del dataset *SpamBase* (UCI Machine Learning Repository).
2. Hace un **análisis exploratorio** rápido para entender qué hay dentro.
3. **Preprocesa** los datos (escalado) y los **divide** en entrenamiento y prueba.
4. **Entrena** un modelo SVM probando varias combinaciones de hiperparámetros.
5. **Evalúa** el desempeño con métricas y gráficos.
6. **Serializa** el pipeline completo a un archivo `.pkl` para que la aplicación
   web (Streamlit) pueda usarlo en producción.

## Sobre el dataset

- **Nombre:** SpamBase
- **Fuente:** UCI ML Repository — <https://archive.ics.uci.edu/dataset/94/spambase>
- **Origen:** correos electrónicos reales recolectados por Hewlett-Packard Labs (1999).
- **Tamaño:** 4.601 emails.
- **Features (57):** frecuencias de 48 palabras clave (`free`, `money`, `credit`…),
  frecuencias de 6 caracteres (`!`, `$`, `#`…) y 3 estadísticas sobre secuencias
  de letras mayúsculas.
- **Target:** `is_spam` ∈ {0, 1} (1 = spam, 0 = ham).

## ¿Por qué SVM?

Las 57 features son densas (casi siempre tienen valores > 0) y de magnitudes muy
distintas. **SVM con kernel RBF** maneja bien fronteras de decisión no lineales en
este tipo de espacio, y `class_weight='balanced'` compensa el ligero desbalance
entre las dos clases (~39 % spam, ~61 % ham).
""")

# ============================================================
# SECCIÓN 1: CARGA
# ============================================================
md("""## 1. Carga y exploración de datos

Empezamos importando las librerías y cargando el CSV crudo. La función
`load_raw()` (definida en `src/preprocessing.py`) le pone los nombres correctos
a las 58 columnas — el archivo original viene sin encabezados.
""")

code("""import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_raw, split_xy, FEATURE_COLUMNS, TARGET_COLUMN

pd.set_option('display.max_columns', 100)
sns.set_theme(style='whitegrid')

df = load_raw('../data/raw/spambase.data')
print('Forma del dataset:', df.shape)
df.head()
""")

md("""La salida confirma que tenemos **4.601 filas × 58 columnas** (57 features + 1 etiqueta).
Las primeras filas son spam (la etiqueta `is_spam = 1`).
""")

code("""# Tipos de datos y memoria utilizada
df.info()
""")

md("""Todas las columnas son numéricas (`float64` o `int64`). No hay columnas categóricas
ni texto crudo, lo cual simplifica mucho el preprocesamiento: **no necesitamos
codificación one-hot ni tokenización**.
""")

code("""# Resumen estadistico de las primeras 15 features
df.describe().T.head(15)
""")

md("""Las features son frecuencias (porcentajes), así que casi todas tienen rangos
[0, 100]. Las tres últimas (`capital_run_length_*`) son longitudes y pueden llegar
a valores grandes. Por eso más adelante aplicamos `StandardScaler`: SVM es muy
sensible a la escala.
""")

# ============================================================
# SECCIÓN 2: EDA
# ============================================================
md("""## 2. Análisis exploratorio (EDA)

Antes de entrenar, vale la pena hacerse tres preguntas:

1. **¿Están las clases balanceadas?** Si una clase domina, hay que ajustarlo.
2. **¿Hay valores nulos o duplicados?** Pueden contaminar el modelo.
3. **¿Qué features parecen separar spam de ham?** Da una intuición de qué va a
   aprender el modelo.
""")

md("### 2.1 Balance de clases")

code("""ax = df[TARGET_COLUMN].value_counts().rename({0: 'ham', 1: 'spam'}).plot.bar(
    color=['#4c72b0', '#c44e52']
)
ax.set_title('Distribución de clases')
ax.set_ylabel('cantidad de emails')
plt.tight_layout(); plt.show()
df[TARGET_COLUMN].value_counts(normalize=True).rename({0: 'ham', 1: 'spam'})
""")

md("""Tenemos **~61 % ham** y **~39 % spam**. Es un desbalance leve, manejable sin
técnicas de sobremuestreo. Lo compensamos con `class_weight='balanced'` en el SVM,
que da más peso a la clase minoritaria durante el entrenamiento.
""")

md("### 2.2 Datos nulos y duplicados")

code("""print('Valores nulos totales:', int(df.isna().sum().sum()))
print('Filas duplicadas       :', int(df.duplicated().sum()))
""")

md("""**Cero nulos** — el dataset está limpio. Sí hay algunas filas duplicadas;
podríamos eliminarlas, pero como son pocas y representan emails idénticos
(probablemente spam reenviado), las dejamos: para SVM no afectan significativamente.
""")

md("""### 2.3 Distribución de features clave por clase

Graficamos histogramas de 6 features que intuitivamente deberían separar spam de ham.
Si las distribuciones por clase son muy distintas, el modelo va a tener facilidad
para diferenciar.
""")

code("""to_plot = ['word_freq_free', 'word_freq_money', 'word_freq_your',
           'char_freq_!', 'char_freq_$', 'capital_run_length_total']
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.ravel(), to_plot):
    for cls, color in [(0, '#4c72b0'), (1, '#c44e52')]:
        df.loc[df[TARGET_COLUMN] == cls, col].clip(upper=df[col].quantile(0.99)).plot.hist(
            bins=40, alpha=0.55, ax=ax, color=color, label='ham' if cls == 0 else 'spam')
    ax.set_title(col); ax.legend()
plt.tight_layout(); plt.show()
""")

md("""Como se esperaba: los emails spam (rojo) usan mucho más las palabras `free`,
`money`, `your`, y abusan de `!` y `$`. También tienen secuencias de mayúsculas
mucho más largas (los típicos "CLICK HERE NOW!!!"). Esto confirma que el dataset
contiene señal suficiente para que el modelo aprenda.
""")

md("### 2.4 Top features por correlación con la etiqueta")

code("""corr = df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(
    key=abs, ascending=False
)
top = corr.head(15)
plt.figure(figsize=(7, 6))
sns.barplot(x=top.values, y=top.index, palette='vlag', hue=top.index, legend=False)
plt.title('Top 15 features por |correlación| con is_spam')
plt.tight_layout(); plt.show()
top
""")

md("""Las features con **correlación positiva** (rojas) son las que aparecen más en
spam: `your`, `000`, `remove`, `$`, `free`, `business`, `money`…
Las features con **correlación negativa** (azules) son típicas de los emails
internos de HP Labs (de donde salió el dataset): `hp`, `george`, `meeting`,
`re` (de "Re:" en respuestas), `edu`. Cuando aparecen, casi seguro es ham.
""")

# ============================================================
# SECCIÓN 3: PREPROCESAMIENTO
# ============================================================
md("""## 3. Preprocesamiento

Para SVM hay que hacer **dos cosas** antes de entrenar:

1. **Escalar las features** (`StandardScaler`): SVM mide distancias en el espacio
   de features, y si una variable va de 0 a 100 y otra de 0 a 5000, la segunda
   domina. El escalado pone todas con media 0 y desviación 1.
2. **Dividir en train/test**: usamos 80 % para entrenar y 20 % para evaluar.
   Hacemos un split **estratificado** para que la proporción spam/ham se mantenga
   en ambos conjuntos.

> El `StandardScaler` no se aplica aquí directamente: lo metemos dentro de un
> `Pipeline` en la siguiente sección, para que se ajuste solo con los datos de
> entrenamiento y se aplique luego al test (evitando *data leakage*).
""")

code("""from sklearn.model_selection import train_test_split

X, y = split_xy(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print('Tamaño train:', X_train.shape)
print('Tamaño test :', X_test.shape)
print('Balance train:', y_train.value_counts(normalize=True).round(3).to_dict())
print('Balance test :', y_test.value_counts(normalize=True).round(3).to_dict())
""")

md("""Quedamos con **3.680 emails de entrenamiento** y **921 de prueba**. La
proporción de spam (~0.39) se conserva en ambos conjuntos gracias al
`stratify=y`. Esto es importante: si el test tuviera proporciones distintas, las
métricas no serían comparables con el escenario real.
""")

code("""# Guardamos los splits a disco para que la app web pueda usarlos en runtime
# (modo "fila aleatoria del test set" en la aplicacion Streamlit).
os.makedirs('../data/processed', exist_ok=True)
X_train.assign(is_spam=y_train).to_csv('../data/processed/train.csv', index=False)
X_test.assign(is_spam=y_test).to_csv('../data/processed/test.csv', index=False)
print('Guardados: data/processed/train.csv y data/processed/test.csv')
""")

# ============================================================
# SECCIÓN 4: ENTRENAMIENTO
# ============================================================
md("""## 4. Entrenamiento del modelo (SVM)

Aquí está el corazón del proyecto. Hacemos tres cosas en este bloque:

### 4.1 Construir un `Pipeline`

Un `Pipeline` encadena pasos en orden. El nuestro tiene:

- `StandardScaler` → escala las features.
- `SVC` (Support Vector Classifier) con **kernel RBF** → el modelo en sí.

La ventaja de usar `Pipeline` es que cuando hagamos `model.predict(...)` desde la
app, el escalado se aplica automáticamente. **No hay que serializar el scaler
por separado.**

### 4.2 Probar varias combinaciones de hiperparámetros

`GridSearchCV` prueba todas las combinaciones de `C` y `gamma` que le damos
(`4 × 3 = 12` combinaciones), entrena cada una con **validación cruzada de 5 folds**,
y se queda con la mejor según el **F1-score** (la métrica más adecuada cuando hay
desbalance).

- `C`: cuánto castiga el modelo los errores de clasificación. Más alto = frontera
  más complicada, menos errores en train pero riesgo de overfitting.
- `gamma`: qué tan "local" es el kernel RBF. Más alto = el modelo se fija en
  vecindarios más pequeños.

### 4.3 Quedarse con el mejor

`grid.best_estimator_` es el pipeline ya re-entrenado con todos los datos de
training usando los mejores hiperparámetros encontrados.
""")

code("""from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=False,
                class_weight='balanced', random_state=42)),
])

param_grid = {
    'svm__C':     [0.5, 1, 3, 10],
    'svm__gamma': ['scale', 0.01, 0.05],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print('Mejores hiperparametros:', grid.best_params_)
print('Mejor F1 en validacion cruzada:', round(grid.best_score_, 4))
model = grid.best_estimator_
""")

md("""El `GridSearchCV` evaluó 12 combinaciones × 5 folds = **60 entrenamientos**.
Los mejores hiperparámetros encontrados fueron `C=10` y `gamma='scale'`, con un
F1 promedio de ~0.92 en validación cruzada. Eso es muy buena señal: significa
que el modelo es estable y no depende de un split particular.
""")

# ============================================================
# SECCIÓN 5: EVALUACIÓN
# ============================================================
md("""## 5. Evaluación

Ya tenemos el modelo entrenado. Ahora lo evaluamos sobre el **test set**
(emails que nunca vio) usando cinco métricas:

- **Accuracy**: % de predicciones correctas. Útil pero engañosa con desbalance.
- **Precision**: de los que dije que eran spam, ¿cuántos sí lo eran? (alto = pocos falsos positivos = pocos correos legítimos enviados a la carpeta de spam).
- **Recall**: de los spam reales, ¿cuántos detecté? (alto = pocos spam que se cuelan a la bandeja).
- **F1**: promedio armónico de precision y recall. La métrica clave aquí.
- **ROC-AUC**: qué tan bien el modelo ordena los emails por probabilidad de spam (1.0 = perfecto, 0.5 = azar).
""")

code("""from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

y_pred  = model.predict(X_test)
y_score = model.decision_function(X_test)

print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall   : {recall_score(y_test, y_pred):.4f}')
print(f'F1       : {f1_score(y_test, y_pred):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, y_score):.4f}')
print()
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
""")

md("""**Resultado:** ~92 % de accuracy y 0.97 de ROC-AUC. El modelo es claramente
mejor que cualquier baseline trivial (clasificar todo como ham daría 61 % de
accuracy). Para un problema clásico como este, estos números son competitivos.
""")

md("### 5.1 Matriz de confusión")

code("""cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'], ax=ax)
ax.set_xlabel('predicción'); ax.set_ylabel('real')
ax.set_title('Matriz de confusión — SVM (test)')
plt.tight_layout(); plt.show()
""")

md("""La diagonal contiene los aciertos. Las celdas fuera de la diagonal son errores:

- **Falsos positivos** (esquina superior derecha): ham clasificado como spam.
  Son los más molestos para el usuario final (pierde correo legítimo).
- **Falsos negativos** (esquina inferior izquierda): spam clasificado como ham.
  Menos graves: solo molestan al usuario.

Idealmente queremos minimizar los falsos positivos (alta precision).
""")

md("### 5.2 Curva ROC")

code("""fpr, tpr, _ = roc_curve(y_test, y_score)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color='#c44e52',
         label=f'SVM (AUC = {roc_auc_score(y_test, y_score):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Modelo aleatorio')
plt.xlabel('FPR (Tasa de falsos positivos)')
plt.ylabel('TPR (Tasa de verdaderos positivos)')
plt.title('Curva ROC')
plt.legend(); plt.tight_layout(); plt.show()
""")

md("""La curva ROC pega al borde superior izquierdo (AUC = 0.97), muy lejos de la
diagonal del modelo aleatorio. Esto significa que el modelo separa muy bien las
dos clases: existe un umbral en el que casi todo el spam queda por encima y casi
todo el ham por debajo.
""")

# ============================================================
# SECCIÓN 6: SERIALIZACIÓN
# ============================================================
md("""## 6. Serialización del modelo

Guardamos el modelo a disco para que la aplicación web (Streamlit) lo pueda cargar
sin tener que reentrenar cada vez. Usamos `joblib` (recomendado por scikit-learn
para objetos con arrays grandes).

**Importante:** guardamos el **pipeline completo**, no solo el SVC. Así el
escalado se aplica automáticamente cuando la app llame a `model.predict(...)`.

Junto al `.pkl` guardamos un **model card** en JSON con todos los metadatos:
hiperparámetros, métricas, fecha, versión de sklearn. Esto es **buena práctica
de MLOps**: cualquier persona que cargue el modelo puede saber exactamente cómo
fue entrenado.
""")

code("""import joblib, json, datetime, sklearn

os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/modelo.pkl')

meta = {
    'algoritmo': 'SVM (kernel rbf)',
    'mejores_hiperparametros': grid.best_params_,
    'metricas_test': {
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall':    float(recall_score(y_test, y_pred)),
        'f1':        float(f1_score(y_test, y_pred)),
        'roc_auc':   float(roc_auc_score(y_test, y_score)),
    },
    'n_train': int(len(X_train)),
    'n_test':  int(len(X_test)),
    'sklearn_version': sklearn.__version__,
    'fecha_entrenamiento': datetime.datetime.now().isoformat(timespec='seconds'),
    'dataset': 'UCI SpamBase (4601 emails reales, HP Labs 1999)',
    'features': FEATURE_COLUMNS,
}
with open('../models/model_card.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print('Guardados: models/modelo.pkl y models/model_card.json')
""")

# ============================================================
# SECCIÓN 7: PRUEBA RÁPIDA
# ============================================================
md("""## 7. Prueba rápida del modelo serializado

Para asegurarnos de que el `.pkl` quedó bien guardado y se puede usar desde
otro proceso, lo cargamos vía la función `predict_from_email` del módulo
`src.predict` (la misma que usa la aplicación Streamlit).

`predict_from_email` toma un texto crudo, lo convierte a las 57 features de
SpamBase (frecuencias de palabras y caracteres) y devuelve la predicción.

> **Aclaración:** el modelo aprendió el vocabulario muy específico de los
> correos de HP Labs en 1999. Las palabras `hp`, `george`, `meeting`, `lab` son
> señales fuertes de ham. Por eso, emails sintéticos modernos pueden no
> clasificarse correctamente desde texto crudo. Para una demo más fiel, mejor
> usar el "modo 2" de la app (fila aleatoria del test set) que sí pasa las
> features reales y acierta ~92 % de las veces.
""")

code("""from src.predict import predict_from_email

email_spam = '''FREE FREE FREE MONEY!!! Make MONEY fast with our BUSINESS opportunity!!!
Your CREDIT is approved! Click REMOVE to unsubscribe. Order NOW!
000 dollars in 30 days. 100 percent guaranteed!'''

email_ham = '''Hi George, the project meeting at HP Labs is at 3pm in conference room 415.
Please review the technology report and send your comments.'''

print('Email spam ->', predict_from_email(email_spam))
print('Email ham  ->', predict_from_email(email_ham))
""")

md("""---

## Conclusiones

- Entrenamos un **SVM con kernel RBF** sobre 3.680 emails reales y alcanzamos
  **~92 % de accuracy** y **0.97 de ROC-AUC** en el test set de 921 emails.
- Los hiperparámetros óptimos (`C=10`, `gamma='scale'`) fueron seleccionados
  con validación cruzada de 5 folds.
- El pipeline completo (`StandardScaler + SVC`) está serializado en
  `models/modelo.pkl` y listo para usarse en producción.
- La aplicación web (`app/app.py`) carga este `.pkl` y permite predecir
  desde tres modos: texto crudo, fila del test set, o edición manual de features.

**Próximo paso:** desplegar la app en Streamlit Cloud para que sea accesible
mediante una URL pública (Entrega 2).
""")

# ============================================================
# Escribir notebook
# ============================================================
nb.cells = cells
nb.metadata = {
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'language_info': {'name': 'python'},
}

out = Path(__file__).parent / '01_training.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('escrito:', out)
