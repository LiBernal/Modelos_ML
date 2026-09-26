import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def plot_confusion_matrix(classes, labels, pred_labels, class_names):
    fig = plt.figure(figsize=(classes * 2, classes * 2))
    ax  = fig.add_subplot(1, 1, 1)
    cm  = metrics.confusion_matrix(labels, pred_labels)
    cm  = metrics.ConfusionMatrixDisplay(cm, display_labels=class_names)
    cm.plot(values_format="d", cmap="Blues", ax=ax)
    plt.tight_layout()

x_train = pd.read_csv("data/X_train.csv",header=None).values
y_train = pd.read_csv("data/Y_train.csv",header=None).values
x_test  = pd.read_csv("data/X_test.csv",header=None).values
y_test  = pd.read_csv("data/Y_test.csv",header=None).values

features = ['bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'sex_encoded']
# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
plt.figure(figsize=(20,20))
species=['Adelie','Chinstrap','Gentoo']
tree.plot_tree(clf, feature_names=features, class_names=species)
plt.show()

# Get feature importances
importances = clf.feature_importances_
print("Features importances")
print(features)
print(importances)

predicted=clf.predict(x_test)
plot_confusion_matrix(3, y_test, predicted, species)
plt.show()

print("Classification Report Test")
print(metrics.classification_report(y_test, predicted))
