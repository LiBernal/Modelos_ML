# cargar librerias
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

x_train = pd.read_csv("data/X_train.csv",header=None).values
y_train = pd.read_csv("data/Y_train.csv",header=None).values.ravel()
x_valid = pd.read_csv("data/X_valid.csv",header=None).values
y_valid = pd.read_csv("data/Y_valid.csv",header=None).values.ravel()
x_test = pd.read_csv("data/X_test.csv",header=None).values
y_test = pd.read_csv("data/Y_test.csv",header=None).values.ravel()

models = [('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
          ('K-Nearest Neighbors', KNeighborsClassifier()),
          ('Desicion Tree Classifier', DecisionTreeClassifier()),
          ('Gaussian Naive Bayes', GaussianNB()),
          ('Support Vector Machine', SVC(gamma='auto')),
          ('Multi Layer Perceptron', MLPClassifier(solver='adam', hidden_layer_sizes=(3), max_iter=2000, learning_rate_init=0.01))]

results = []
names = ["LDA", "KNN", "DTC", "NB", "SVC", "MLP"]

for name, model in models:
    kfold = model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, n_jobs=-1, scoring='accuracy')
    results.append(cv_results)
    msg = "Method %s: mean %f std(%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Comparacion de Algoritmos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

species=['Adelie','Chinstrap','Gentoo']

lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
predictions = lda.predict(x_valid)
print("Linear Discriminant Analysis Validation")
print(accuracy_score(y_valid, predictions))
print(confusion_matrix(y_valid, predictions))
print(classification_report(y_valid, predictions))

predictions = lda.predict(x_test)
print("Linear Discriminant Analysis Test")
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('*'*60)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_valid)
print("K-Nearest Neighbors Validation")
print(accuracy_score(y_valid, predictions))
print(confusion_matrix(y_valid, predictions))
print(classification_report(y_valid, predictions))

predictions = knn.predict(x_test)
print("K-Nearest Neighbors Test")
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('*'*60)

