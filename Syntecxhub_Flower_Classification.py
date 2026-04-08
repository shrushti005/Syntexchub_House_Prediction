import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("First 5 Rows:")
print(df.head())


print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

plt.figure()
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['species'])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Width")
plt.show()

plt.figure()
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['species'])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length vs Width")
plt.show()

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, y_pred_log)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nModel Accuracy Comparison:")
print("Logistic Regression Accuracy:", log_accuracy)
print("Decision Tree Accuracy:", tree_accuracy)

cm = confusion_matrix(y_test, y_pred_tree)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()

print("\n--- Flower Species Prediction ---")

try:
    sl = float(input("Enter Sepal Length: "))
    sw = float(input("Enter Sepal Width: "))
    pl = float(input("Enter Petal Length: "))
    pw = float(input("Enter Petal Width: "))

    new_data = np.array([[sl, sw, pl, pw]])

    prediction = tree_model.predict(new_data)

    print("Predicted Species:", iris.target_names[prediction[0]])

except:
    print("Invalid input! Please enter numeric values.")
