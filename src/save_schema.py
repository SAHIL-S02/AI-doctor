import joblib

def save_feature_schema(X):
    joblib.dump(list(X.columns), "models/feature_schema.pkl")
