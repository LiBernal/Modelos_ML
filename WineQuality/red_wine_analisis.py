import pandas as pd
import matplotlib.pyplot as plt

url = "http://posgrado.itlp.edu.mx/datasets/winequality-red.csv"

dataset = pd.read_csv(url, sep=';')

print("Tamaño de los datos: ",dataset.shape)
print("Nombres de las columnas",dataset.columns)
descriptor = dataset.describe()
print("\n\nEstadísticos:\n",descriptor[['fixed acidity', 'volatile acidity', 'citric acid']])
print("\n\nEstadísticos:\n",descriptor[['residual sugar', 'chlorides', 'free sulfur dioxide']])
print("\n\nEstadísticos:\n",descriptor[['total sulfur dioxide','density', 'pH']])
print("\n\nEstadísticos:\n",descriptor[['sulphates', 'alcohol','quality']])
print("\n\n Datos de cada clase: \n",dataset.groupby('quality').size())

clases=[3,4,5,6,7,8,9]

plt.hist(dataset[['quality']],bins=clases)
plt.suptitle("Histogramas")
plt.show()

dataset.plot(kind='box', column=['fixed acidity','residual sugar', 'alcohol'])
plt.suptitle("Diagramas de cajas de áreas")
plt.show()

dataset.plot(kind='box', column=['pH'])
plt.suptitle("Diagramas de cajas de áreas")
plt.show()

dataset.plot(kind='box', column=['volatile acidity', 'citric acid', 'sulphates'])
plt.suptitle("Diagramas de cajas")
plt.show()

dataset.plot(kind='box', column=['density'])
plt.suptitle("Diagramas de cajas")
plt.show()

dataset.plot(kind='box', column=['chlorides'])
plt.suptitle("Diagramas de cajas")
plt.show()

dataset.plot(kind='box', column=['free sulfur dioxide'])
plt.suptitle("Diagramas de cajas")
plt.show()

dataset.plot(kind='box', column=['total sulfur dioxide'])
plt.suptitle("Diagramas de cajas")
plt.show()

framel = dataset[['fixed acidity','residual sugar', 'alcohol']]
pd.plotting.scatter_matrix(framel)
plt.suptitle("Diagrama de dispersión (histogramas en el diagonal)")
plt.show()

framel = dataset[['volatile acidity', 'citric acid', 'sulphates']]
pd.plotting.scatter_matrix(framel)
plt.suptitle("Diagrama de dispersión (histogramas en el diagonal)")
plt.show()

framel = dataset[['pH', 'density', 'chlorides']]
pd.plotting.scatter_matrix(framel)
plt.suptitle("Diagrama de dispersión (histogramas en el diagonal)")
plt.show()

framel = dataset[['free sulfur dioxide', 'total sulfur dioxide']]
pd.plotting.scatter_matrix(framel)
plt.suptitle("Diagrama de dispersión (histogramas en el diagonal)")
plt.show()