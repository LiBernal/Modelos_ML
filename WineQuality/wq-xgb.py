import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
import xgboost as xgb

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
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, early_stopping_rounds=5)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

# Probar el modelo
from sklearn import metrics

predicted = model.predict(X_valid)
print("Confusion Matrix Validation")
print(metrics.confusion_matrix(y_valid, predicted))

print("Classification Report Validation")
print(metrics.classification_report(y_valid, predicted))

predicted = model.predict(X_test)
print("Confusion Matrix Test")
print(metrics.confusion_matrix(y_test, predicted))

print("Classification Report Test")
print(metrics.classification_report(y_test, predicted))
