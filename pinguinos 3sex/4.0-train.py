import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from scipy import stats

x_train = pd.read_csv("data/X_train.csv",header=None).values
y_train = pd.read_csv("data/Y_train.csv",header=None).values
x_valid = pd.read_csv("data/X_valid.csv",header=None).values
y_valid = pd.read_csv("data/Y_valid.csv",header=None).values

best = 1

for trial in range(5):
   model = keras.models.Sequential()
   model.add(keras.layers.Input([4]))
   model.add(keras.layers.Dense(3, activation="sigmoid"))
   model.add(keras.layers.Dense(3, activation="softmax"))
   model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  metrics=[keras.metrics.sparse_categorical_accuracy])
   model.summary()
   checkpoint_cb = keras.callbacks.ModelCheckpoint("penguins.keras", save_best_only=True)
   early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)


   history = model.fit(x_train,y_train, epochs=500, validation_data=(x_valid,y_valid),
	                       callbacks=[checkpoint_cb, early_stopping_cb])
   e=model.evaluate(x_valid,y_valid)
   if (e[0]<best):
      best = e[0]
      best_model = model
   print(trial+1,e)

best_model.save("best_model.keras")
best_model.evaluate(x_valid,y_valid)
