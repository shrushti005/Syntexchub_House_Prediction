import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("C:\Users\shru\OneDrive\Desktop\housedata.csv")

print("\nFirst 5 Rows of Dataset:\n")
print(df.head())

print("\nDataset Information:\n")
print(df.info())
df = df.drop(["date", "street", "city", "statezip", "country"], axis=1)

df = df.dropna()

X = df.drop("price", axis=1) 
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("RMSE:", rmse)
print("R2 Score:", r2)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients:\n")
print(coefficients)

joblib.dump(model, "house_price_model.pkl")
print("\nModel saved successfully as 'house_price_model.pkl'")

sample_data = X_test.iloc[0:1]
prediction = model.predict(sample_data)

print("\nExample Prediction:")
print("Actual Price:", y_test.iloc[0])
print("Predicted Price:", prediction[0])

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()
