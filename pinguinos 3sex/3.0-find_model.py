import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

def model_builder(hp):
  hp_units1 = hp.Int('units1', min_value=3, max_value=5, step=1)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
  hp_activation = hp.Choice('activation', values=['leaky_relu','sigmoid'])
  
  model = keras.Sequential()
  model.add(keras.layers.Input([4]))
  model.add(keras.layers.Dense(units=hp_units1, activation=hp_activation))
  model.add(keras.layers.Dense(3, activation='softmax'))
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['sparse_categorical_accuracy'])
  return model

x_train = pd.read_csv("data/X_train.csv",header=None).values
y_train = pd.read_csv("data/Y_train.csv",header=None).values

tuner = kt.GridSearch(model_builder,
                     objective='val_sparse_categorical_accuracy',
                     directory='Experimentos',
                     project_name='penguins1')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

tuner.search(x_train, y_train, epochs=500, validation_split=0.2, batch_size=16, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units1')} with {best_hps.get('activation')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')} .
""")
