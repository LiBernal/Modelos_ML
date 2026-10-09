import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
# from sklearn.metrics import f1_score, balanced_accuracy_score

# 1. Cargar el dataset
df = pd.read_csv("http://posgrado.itlp.edu.mx/datasets/winequality-red.csv", sep=";")

# 2. Agrupar la calidad del vino en 3 clases: Low (0), Medium (1), High (2)
df["quality"] = df["quality"].replace({3: 0, 4: 0, 5: 0, 6: 1, 7: 2, 8: 2, 9: 2})

# 3. Separar características (X) y variable objetivo (y)
X = df.values[:, 0:11]
y = df.values[:, 11]

# Codificar la etiqueta y escalar las características
label_encoder = preprocessing.LabelEncoder()
y_encoded = label_encoder.fit_transform(np.ravel(y))

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Definir nombres y modelos a comparar
names = ["MLP", "RFC", "XGB"]

models = [
    (
        "Multi-Layer Perceptron",
        MLPClassifier(
            hidden_layer_sizes=(100, 30), # 2 capas ocultas: 100 y 30 neuronas
            max_iter=500,                # Máximo 150 épocas por pliegue
            random_state=42,
            early_stopping = True,
            verbose=False              # Cambiar a True para ver el progreso época por época
        ),
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=100,             # 100 árboles
            max_depth=4,                  # Profundidad max 4
            random_state=42,
            n_jobs=-1,                    # Paraleliza la creación de árboles
            verbose=True
        ),
    ),
    (
        "XGBoost",
        XGBClassifier(
            n_estimators=100,             # 100 árboles
            max_depth=4,
            learning_rate=0.1,            # tasa de aprendizaje
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1                    # Paraleliza el entrenamiento
        ),
    ),
]

# 5. Evaluación con Validación Cruzada Repetida
results = []
for name, model in models:
    kfold = model_selection.RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=42
    )
    cv_results = model_selection.cross_val_score(
        model, X_scaled, y_encoded, cv=kfold, scoring="accuracy"
    )
    results.append(cv_results)
    msg = "%s: mean %f (std %f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# 6. Gráfica de comparación (Boxplot)
fig = plt.figure(figsize=(8, 6))
fig.suptitle("Comparación de Algoritmos (MLP vs RF vs XGB)")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_ylabel("Exactitud (Accuracy)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()