"""Genera notebooks/01_training.ipynb desde codigo fuente."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Clasificador de Spam con Support Vector Machines (SVM)

**Grupo 7 - Inteligencia Artificial I - Actividad 3**

Entrenamiento del modelo a partir del dataset **SpamBase** del UCI Machine Learning Repository.

- Fuente: <https://archive.ics.uci.edu/dataset/94/spambase>
- 4.601 emails reales recolectados por Hewlett-Packard Labs
- 57 features numericas + 1 etiqueta binaria (1 = spam, 0 = ham)
""")

md("## 1. Carga y exploracion de datos")

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
print('shape:', df.shape)
df.head()
""")

code("df.info()")

code("df.describe().T.head(15)")

md("## 2. EDA (Analisis Exploratorio)")

code("""ax = df[TARGET_COLUMN].value_counts().rename({0: 'ham', 1: 'spam'}).plot.bar(color=['#4c72b0', '#c44e52'])
ax.set_title('Distribucion de clases')
ax.set_ylabel('cantidad de emails')
plt.tight_layout(); plt.show()
df[TARGET_COLUMN].value_counts(normalize=True).rename({0:'ham',1:'spam'})
""")

code("""print('nulos por columna (totales):', int(df.isna().sum().sum()))
print('duplicados:', int(df.duplicated().sum()))
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

code("""corr = df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(key=abs, ascending=False)
top = corr.head(15)
plt.figure(figsize=(7, 6))
sns.barplot(x=top.values, y=top.index, palette='vlag', hue=top.index, legend=False)
plt.title('Top 15 features por |corr| con is_spam')
plt.tight_layout(); plt.show()
top
""")

md("""## 3. Preprocesamiento

- Las 57 features ya son numericas (no requieren codificacion categorica).
- SVM es sensible a la escala, por eso aplicamos `StandardScaler`.
- Division train/test estratificada 80/20.
- Guardamos una copia procesada en `data/processed/` para reproducibilidad.
""")

code("""from sklearn.model_selection import train_test_split

X, y = split_xy(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print('train:', X_train.shape, 'test:', X_test.shape)
print('balance train:', y_train.value_counts(normalize=True).to_dict())
""")

code("""os.makedirs('../data/processed', exist_ok=True)
X_train.assign(is_spam=y_train).to_csv('../data/processed/train.csv', index=False)
X_test.assign(is_spam=y_test).to_csv('../data/processed/test.csv', index=False)
print('guardado train.csv y test.csv')
""")

md("""## 4. Entrenamiento del modelo (SVM)

Entrenamos un pipeline `StandardScaler -> SVC(kernel='rbf')`. Usamos `GridSearchCV`
(busqueda corta) sobre `C` y `gamma`, optimizando F1 con validacion cruzada de 5 folds.
""")

code("""from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=False, class_weight='balanced', random_state=42)),
])

param_grid = {
    'svm__C':     [0.5, 1, 3, 10],
    'svm__gamma': ['scale', 0.01, 0.05],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print('mejores hiperparametros:', grid.best_params_)
print('mejor f1 cv:', round(grid.best_score_, 4))
model = grid.best_estimator_
""")

md("## 5. Evaluacion")

code("""from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

y_pred  = model.predict(X_test)
y_score = model.decision_function(X_test)

print(f'accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'precision: {precision_score(y_test, y_pred):.4f}')
print(f'recall   : {recall_score(y_test, y_pred):.4f}')
print(f'f1       : {f1_score(y_test, y_pred):.4f}')
print(f'roc-auc  : {roc_auc_score(y_test, y_score):.4f}')
print()
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
""")

code("""cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'], ax=ax)
ax.set_xlabel('prediccion'); ax.set_ylabel('real')
ax.set_title('Matriz de confusion - SVM (test)')
plt.tight_layout(); plt.show()
""")

code("""fpr, tpr, _ = roc_curve(y_test, y_score)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color='#c44e52', label=f'SVM (AUC = {roc_auc_score(y_test, y_score):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Curva ROC')
plt.legend(); plt.tight_layout(); plt.show()
""")

md("""## 6. Serializacion del modelo

Guardamos el **pipeline completo** (`StandardScaler` + `SVC`) en `models/modelo.pkl`.
No hace falta serializar el scaler por separado porque esta dentro del `Pipeline`.
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
print('modelo.pkl y model_card.json guardados')
""")

md("## 7. Prueba rapida del modelo serializado")

code("""from src.predict import predict_from_email

sample_spam = '''FREE!!! Get your CREDIT report NOW for $0. Click here to remove your debt and earn MONEY fast!!!
Your address has been selected. Receive your free order today. CALL 1-800-555-0000!!!'''

sample_ham = '''Hi George, attached is the agenda for tomorrow's project meeting at HP Labs.
Please review the technology section and send your comments by 5pm. Thanks.'''

print('SPAM ejemplo ->', predict_from_email(sample_spam))
print('HAM  ejemplo ->', predict_from_email(sample_ham))
""")

nb.cells = cells
nb.metadata = {
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'language_info': {'name': 'python'},
}

out = Path(__file__).parent / '01_training.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('escrito:', out)
