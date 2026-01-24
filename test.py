import pandas as pd
from src.preprocessing import preprocess
from src.model import train_logistic_model

df = pd.read_csv("data/diabetes.csv")
df = preprocess(df)

model, X_test, y_test, y_proba = train_logistic_model(df)
