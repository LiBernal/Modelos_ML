# Producción de un modelo XGBoost
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
import xgboost as xgb

df = pd.read_csv("http://posgrado.itlp.edu.mx/datasets/winequality-red.csv",sep=";")

# --- Normalizar datos "nuevos" para poder usar el modelo pre-guardado ---
# Group wine quality into 3 classes
# Low (3-5), Medium (6), High (7-8)
df["quality"] = df["quality"].replace({3:0, 4:0, 5:0, 6:1, 7:2, 8:2, 9:2})

X = df.values[:,0:11]
y = df.values[:,11]

# abrir en modo read binary
with open("scalerXGB.pkl","rb") as handler:
    scaler = pickle.load(handler)

# abrir encoder
with open("label_encoder.pkl", "rb") as handler:
    label_encoder = pickle.load(handler)
y_encoded = label_encoder.transform(y)    

z = scaler.transform(X)

model = xgb.XGBClassifier()
model.load_model("xgb.json")
print(model)

# Evaluación
acc = model.score(z, y)
print(f"Accuracy: {acc:.4f}")

yp = model.predict(z)   # ya son clases, NO aplicar argmax
print("Confusion Matrix")
print(metrics.confusion_matrix(y, yp))
print("\nClassification Report")
print(metrics.classification_report(y, yp))