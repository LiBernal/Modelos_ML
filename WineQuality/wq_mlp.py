import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
import keras

df = pd.read_csv("http://posgrado.itlp.edu.mx/datasets/winequality-red.csv",sep=";")

# Group wine quality into 3 classes
# Low (3-5), Medium (6), High (7-8)
df["quality"] = df["quality"].replace({3:0, 4:0, 5:0, 6:1, 7:2, 8:2, 9:2})

X = df.values[:,0:11]
y = df.values[:,11]

#  ¿Cuantos hay de cada clase?
print(df.groupby('quality').size())

#Normalizar los datos, y dividirlos en 
#datos de entrenamiento `train` (60%) , 
#validación `valid` (20%) usado para el early-stopping) y 
#prueba `test` (20%) no usado en ninguna fase del entrenamiento)

scaler = preprocessing.StandardScaler()
scaler.fit(X)
Xnormalizada = scaler.transform(X)

label_encoder = preprocessing.LabelEncoder()
y_integer_encoded = label_encoder.fit_transform(np.ravel(y))
X_train, X_test, y_train, y_test = model_selection.train_test_split(Xnormalizada, y_integer_encoded, test_size=0.10, stratify=y_integer_encoded)

balancer=SMOTE()

X_resampled, y_resampled = balancer.fit_resample(X_train, y_train)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_resampled, y_resampled, test_size=0.20, stratify=y_resampled)

# Entrenar el modelo

checkpoint_cb = keras.callbacks.ModelCheckpoint("rwine.keras", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

model = keras.models.Sequential()
model.add(keras.layers.Input([11]))
model.add(keras.layers.Dense(250, activation= "relu"))
model.add(keras.layers.Dense(50, activation= "relu"))
model.add(keras.layers.Dense(3, activation= "softmax"))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=[keras.metrics.sparse_categorical_accuracy])
model.summary()

model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
          callbacks=[checkpoint_cb, early_stopping_cb])
e = model.evaluate(X_valid, y_valid)
print(e)

# Probar el modelo
from sklearn import metrics

predicted = np.argmax(model.predict(X_valid), axis=-1)
print("Confusion Matrix Validation")
print(metrics.confusion_matrix(y_valid, predicted))

print("Classification Report Validation")
print(metrics.classification_report(y_valid, predicted))

predicted = np.argmax(model.predict(X_test), axis=-1)
print("Confusion Matrix Test")
print(metrics.confusion_matrix(y_test, predicted))

print("Classification Report Test")
print(metrics.classification_report(y_test, predicted))