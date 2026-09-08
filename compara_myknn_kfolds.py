#        -*- coding: utf-8 -*-

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import statistics

url = "http://posgrado.itlp.edu.mx/datasets/iris.data"
#file = "iris.data"
 
df = pd.read_csv(url, header=None)

x = df.values[:,0:4] #todos los renglones, columnas 0 a 3 pero python es rarito y se pone uno mas
y = df.values[:,4]

kf = model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=20) #divide los datos

pruebas = [2,5,10,20]
means = []

for k in pruebas:
    modelo = knc(n_neighbors=k)

    results = []

    for train_index, test_index in kf.split(x,y):
        xt = x[train_index]
        xv = x[test_index]
        yt = y[train_index]
        yv = y[test_index]

    #xt = xtrain, xt = xvalidacion, yt = ytrain, yv = yvalidacion  --- stratify y es para que agarre datos uniformes de cada clase
    #xt, xv, yt, yv = model_selection.train_test_split(x, y, test_size=0.2, stratify=y) #test_size es el porcentaje para la prueba

        modelo.fit(xt,yt) #fit para entrenar el modelo

        predictions = modelo.predict(xv)
        acc = accuracy_score(yv, predictions) #debe ser porcentaje
        results.append(acc)
        #print('Datos de validacion', acc) #debe ser porcentaje
    
    mean = statistics.mean(results)
    print("K=", k, "Mean accuracy=", mean)
    means.append(mean)
    
index = means.index(max(means))
print(max(means))
print(pruebas[index])

#print(results)
