# Producción de un modelo
import pandas as pd
import numpy as np
import pickle
import keras
from sklearn.metrics import confusion_matrix

df = pd.read_csv("http://posgrado.itlp.edu.mx/datasets/winequality-red.csv",sep=";")

# --- Normalizar datos "nuevos" para poder usar el modelo pre-guardado ---

# Group wine quality into 3 classes
# Low (3-5), Medium (6), High (7-8)
# del dataframe modifico quality los valores del 0 al 2
df["quality"] = df["quality"].replace({3:0, 4:0, 5:0, 6:1, 7:2, 8:2, 9:2})

X = df.values[:,0:11] # entradas
y = df.values[:,11] # salida el modelo

# abrir en modo read binary
with open("scaler.pkl","rb") as handler:
    scaler = pickle.load(handler)

z = scaler.transform(X)

model = keras.saving.load_model("mlp.keras")
print(model.summary())

# para ver si los pesos andan bien
model.evaluate(z,y)
# axis = -1 
yp = np.argmax(model.predict(z), axis=-1)
print(confusion_matrix(y,yp))