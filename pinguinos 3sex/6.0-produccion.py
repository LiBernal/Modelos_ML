import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import keras
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_confusion_matrix(classes, labels, pred_labels):

    fig = plt.figure(figsize=(classes, classes))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(classes))
    cm.plot(values_format='d', cmap='Blues', ax=ax)


# Cargar datos
df = pd.read_csv("penguins.csv")

# Descartar inválidos
df_clean = df.dropna().reset_index(drop=True)

# Codificamos como 0 y 1 el sexo
encoder=LabelEncoder()
encoder.fit(['male','female'])
df_clean['sex_encoded'] = encoder.transform(df_clean['sex'])

features = ['bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'sex_encoded'] #Ignoramos el tamaño de aleta
X = df_clean[features].values
y_species = df_clean['species'].values

# Normalizar
with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)
file.close()
X_scaled = scaler.transform(X)
# Mapear especies a números para comparación
encoder.fit(['Adelie','Chinstrap','Gentoo'])
y_numeric = encoder.transform(y_species)

model = keras.models.load_model("best_model.keras")

##### Evaluación del modelo contra los datos totales
model.evaluate(X_scaled,y_numeric)

predicted = np.argmax(model.predict(X_scaled), axis=-1) #argmax devuelve el mayor valor de un vctor
print("Confusion Matrix Test")
print(metrics.confusion_matrix(y_numeric, predicted))
plot_confusion_matrix(3, y_numeric, predicted)
plt.show()

print("Classification Report Test")
print(metrics.classification_report(y_numeric, predicted))
