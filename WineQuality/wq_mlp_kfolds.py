import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
import keras

df = pd.read_csv("http://posgrado.itlp.edu.mx/datasets/winequality-red.csv",sep=";")

# Group wine quality into 3 classes
# Low (3-5), Medium (6), High (7-8)
df["quality"] = df["quality"].replace({3:0, 4:0, 5:0, 6:1, 7:2, 8:2, 9:2})

X = df.values[:,0:11]
y = df.values[:,11].astype(int)

#  ¿Cuantos hay de cada clase?
print(df.groupby('quality').size())

# Separar los datos de TEST (10%)
# Los de test NO se usan para validación cruzada
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

# k-folds estratificado
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# listas para guardar métricas por fold
accuracies = []
f1_macros = []
precisions = []
recalls = []

# VALIDACIÓN CRUZADA
for fold, (train_index, valid_index) in enumerate(
        kf.split(X_train, y_train), start=1):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    # datos del fold
    X_train_fold = X_train[train_index]
    X_valid_fold = X_train[valid_index]

    y_train_fold = y_train[train_index]
    y_valid_fold = y_train[valid_index]

    # Normalizar los datos
    scaler = preprocessing.StandardScaler()
    
    # SOLO aprende del train
    X_train_fold = scaler.fit_transform(X_train_fold)

    # Transformamos validación usando el scaler anterior
    X_valid_fold = scaler.transform(X_valid_fold)

    # balancear el TRAIN con SMOTE
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_fold,
        y_train_fold
    )

    print("Distribución antes de SMOTE:")
    print(np.bincount(y_train_fold))

    print("Distribución después de SMOTE:")
    print(np.bincount(y_train_resampled))

    # Definir el modelo (MLP)
    model = keras.models.Sequential()
    model.add(keras.layers.Input([11]))
    model.add(keras.layers.Dense(250, activation= "relu"))
    model.add(keras.layers.Dense(50, activation= "relu"))
    model.add(keras.layers.Dense(3, activation= "softmax"))

    # Compilar el modelo
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.sparse_categorical_accuracy])
    model.summary()

    # callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint("rwine.keras", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # Entrenar el modelo
    model.fit(X_train_resampled, y_train_resampled, epochs=100, validation_data=(X_valid_fold, y_valid_fold), callbacks=[early_stopping_cb], verbose=0)

    # Prediccion
    predicted = np.argmax(model.predict(X_valid_fold), axis=-1)

    # Metricas
    accuracy = metrics.accuracy_score(y_valid_fold, predicted)
    f1_macro = metrics.f1_score(y_valid_fold, predicted, average="macro")
    precision = metrics.precision_score(y_valid_fold, predicted, average="macro")
    recall = metrics.recall_score(y_valid_fold, predicted,average="macro")

    accuracies.append(accuracy)
    f1_macros.append(f1_macro)
    precisions.append(precision)
    recalls.append(recall)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Macro:  {f1_macro:.4f}")

    # Matriz de confusion
    print("\nMatriz de confusión:")
    print(metrics.confusion_matrix(y_valid_fold, predicted))

# 8. RESULTADOS DE VALIDACIÓN CRUZADA
print("\n")
print("=" * 60)
print("RESULTADOS MLP - VALIDACIÓN CRUZADA")
print("=" * 60)

print(f"Accuracy promedio:  {np.mean(accuracies):.4f}")
print(f"Accuracy std:       {np.std(accuracies):.4f}")

print(f"Precision promedio: {np.mean(precisions):.4f}")
print(f"Recall promedio:    {np.mean(recalls):.4f}")
print(f"F1 Macro promedio:  {np.mean(f1_macros):.4f}")

# 9. ENTRENAMIENTO FINAL

print("\n")
print("=" * 60)
print("ENTRENAMIENTO FINAL Y TEST")
print("=" * 60)

# Normalización usando TODO el train
scaler_final = preprocessing.StandardScaler()

X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

# SMOTE solamente sobre TRAIN
smote_final = SMOTE()

X_train_resampled, y_train_resampled = smote_final.fit_resample(X_train_scaled, y_train)

# Crear modelo final
model_final = keras.models.Sequential([
    keras.layers.Input(shape=(11,)),
    keras.layers.Dense(250, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

model_final.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(),
    metrics=["sparse_categorical_accuracy"]
)

model_final.fit(
    X_train_resampled,
    y_train_resampled,
    epochs=100,
    verbose=0
)

# 10. EVALUACIÓN FINAL EN TEST
predicted_test = np.argmax(
    model_final.predict(X_test_scaled, verbose=0),
    axis=1
)

print("\nMatriz de confusión TEST:")
print(metrics.confusion_matrix(y_test, predicted_test))

print("\nClassification Report TEST:")
print(metrics.classification_report(y_test, predicted_test, target_names=["Malo","Regular", "Bueno"]))