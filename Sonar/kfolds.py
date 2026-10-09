import keras
import pandas as pd
from sklearn import model_selection
import statistics

K = 5  # n-splits -> numero de folds
repeats = 2

def compile_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Input([60]))
    model.add(keras.layers.Dense(40, activation="leaky_relu"))
    model.add(keras.layers.Dense(5, activation="leaky_relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[keras.metrics.BinaryAccuracy()])
    return model

# k-foldssss (RepeatedKFold)
kfold = model_selection.RepeatedKFold(n_splits=K, n_repeats=repeats)

# Load training data (datos ya preparados)
# Con estos datos se evalua el modelo
X = pd.read_csv("train-data/X_train.csv").values
Y = pd.read_csv("train-data/Y_train.csv").values

results=[]
# Iterate through each fold
# Desempeño del modelo en cada iteracion
for train, test in kfold.split(X, Y):
	xt = X[train]
	yt = Y[train]
	xv = X[test]
	yv = Y[test]

	model = compile_model()
	# checkpoint_cb = keras.callbacks.ModelCheckpoint("sonar.keras", save_best_only=True)
	early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

	model.fit(xt,yt, epochs=500, validation_data=(xv,yv), batch_size=16, verbose=0,
	                    callbacks=[early_stopping_cb])

	a = model.evaluate(xv, yv)[1] # compara entrada y salida esperada, devuelve funcion de perdida osea error
	results.append(a) # accuracy

# Calculate and print statistics
mean = statistics.mean(results)
std = statistics.stdev(results)
minimum = min(results)
maximum = max(results)
print("\nResults")
print("Validation Accuracy",results)
print("Mean",mean)
print("Range (",minimum,"-",maximum,")")
print("STD", std)