"""
Preprocesses the Sonar dataset and splits it into training and validation sets.

This script reads the 'sonar.all-data' file, separates features and labels,
normalizes the features, encodes the labels, and splits the data into
training and validation sets, saving them as CSV files.
"""

import numpy
import pandas
from sklearn import model_selection
from sklearn import preprocessing

filename = "sonar.all-data"

dataset = pandas.read_csv(filename, header=None)

array = dataset.values
X = array[:, 0:60]  # columnas 0...10 (Variables de entrada)
Y = array[:, 60]  # columna 60 (R  o M)

# Normalize features
scaled_X = preprocessing.normalize(X, norm="max", axis=0)

# Encode labels
label_encoder = preprocessing.LabelEncoder()
Y_integer_encoded = label_encoder.fit_transform(Y)
validation_size = 0.20

X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(scaled_X, Y_integer_encoded,
                                                                      test_size=validation_size)
numpy.savetxt("train-data/X_full.csv", scaled_X, delimiter=",")
numpy.savetxt("train-data/X_train.csv", X_train, delimiter=",")
numpy.savetxt("train-data/X_valid.csv", X_valid, delimiter=",")
numpy.savetxt("train-data/Y_train.csv", Y_train, delimiter=",")
numpy.savetxt("train-data/Y_valid.csv", Y_valid, delimiter=",")
numpy.savetxt("train-data/Y_full.csv", Y_integer_encoded, delimiter=",")
