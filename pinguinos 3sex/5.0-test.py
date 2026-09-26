import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
def plot_confusion_matrix(classes, labels, pred_labels, class_names):
    fig = plt.figure(figsize=(classes * 2, classes * 2))
    ax  = fig.add_subplot(1, 1, 1)
    cm  = metrics.confusion_matrix(labels, pred_labels)
    cm  = metrics.ConfusionMatrixDisplay(cm, display_labels=class_names)
    cm.plot(values_format="d", cmap="Blues", ax=ax)
    plt.tight_layout()

x_test  = pd.read_csv("data/X_test.csv",header=None).values
y_test  = pd.read_csv("data/Y_test.csv",header=None).values
model = keras.models.load_model("best_model.keras")
model.evaluate(x_test,y_test)
predicted = np.argmax(model.predict(x_test), axis=-1) #argmax devuelve el mayor valor de un vctor
print("Confusion Matrix Test")
print(metrics.confusion_matrix(y_test, predicted))
species=['Adelie','Chinstrap','Gentoo']
plot_confusion_matrix(3, y_test, predicted, species)
plt.show()

print("Classification Report Test")
print(metrics.classification_report(y_test, predicted))
