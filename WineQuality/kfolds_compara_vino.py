import pandas as pd
import numpy as np
import keras
import statistics
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt

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

print("Antes de SMOTE:", np.bincount(y_train))
print("Después de SMOTE:", np.bincount(y_resampled))
# ----------------------------------------------------------
# X_resampled y y_resampled

# MODELOS
def compile_model(nombre):
	if nombre == "MLP":
		# MLP (Multilayered Perceptron)
		model = keras.models.Sequential()
		model.add(keras.layers.Input([11]))
		model.add(keras.layers.Dense(100, activation= "leaky_relu"))
		model.add(keras.layers.Dense(50, activation= "leaky_relu"))
		model.add(keras.layers.Dense(3, activation= "softmax"))

		model.compile(loss=keras.losses.sparse_categorical_crossentropy,
						optimizer=keras.optimizers.Adam(),
						metrics=[keras.metrics.sparse_categorical_accuracy])
		return model
	
	# Random Forest (RF)
	elif nombre == "RF":
		return RandomForestClassifier(n_estimators=100, max_depth=4)
		#model.fit(X_train, y_train)

	# Extreme Gradient Boosting (XG Boost)
	elif nombre == "XGB":
		return xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, early_stopping_rounds=5)
	# model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

# -------------------------------------------
# parametros para el k-folds
K = 5  # n-splits -> numero de folds
repeats = 2

# k-foldssss (RepeatedKFold)
modelos =  ["MLP", "RF", "XGB"]
results = {m: [] for m in modelos}   # <-- resultados por modelo

kfold = model_selection.RepeatedKFold(n_splits=K, n_repeats=repeats)

# Iterate through each fold
# Desempeño del modelo en cada iteracion
for train, test in kfold.split(X_resampled, y_resampled):
	xt = X_resampled[train]
	yt = y_resampled[train]
	xv = X_resampled[test]
	yv = y_resampled[test]
	
	for nombre in modelos:
		model = compile_model(nombre)
		
		if nombre == "MLP":
			early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
			model.fit(xt,yt, epochs=250, validation_data=(xv,yv), batch_size=16, verbose=0,callbacks=[early_stopping_cb])
			acc = model.evaluate(xv, yv)[1] # compara entrada y salida esperada, devuelve funcion de perdida osea error
		
		elif nombre == "RF":
			model.fit(xt, yt)
			acc = model.score(xv, yv)
		
		elif nombre == "XGB":
			model.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
			acc = model.score(xv, yv)
		  
		results[nombre].append(acc) # accuracy

# ---------------- Estadísticas ----------------
print("\n===== Resultados por modelo =====")
for nombre in modelos:
	r = results[nombre]
	print(f"\n{nombre}")
	print("  Accuracies:", [f"{x:.4f}" for x in r])
	print(f"  Mean : {statistics.mean(r):.4f}")
	print(f"  STD  : {statistics.stdev(r):.4f}")
	print(f"  Min  : {min(r):.4f}  Max: {max(r):.4f}")


# ---------------- Diagrama de cajas ----------------
plt.figure(figsize=(8, 5))
plt.boxplot([results[m] for m in modelos],
			labels=modelos,
			showmeans=True,
			meanline=True)
plt.ylabel("Accuracy en validación (K-Fold)")
plt.title("Comparación de modelos — Repeated K-Fold")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()