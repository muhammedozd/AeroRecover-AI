import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.data.load_flights import load_flights
from src.data.preprocess_data import preprocess_data



X_train, X_test, y_train, y_test = preprocess_data()

model = LogisticRegression(max_iter=1000)
model = joblib.load("models/logistic_regression.pkl")
y_pred = model.predict(X_test)
print("İlk 10 tahmin:")
print(y_pred[:10])