import joblib

def save_models(base_model, calibrated_model, X_background):
    joblib.dump(base_model, "models/logistic_base.pkl")
    joblib.dump(calibrated_model, "models/logistic_calibrated.pkl")
    joblib.dump(X_background, "models/background.pkl")
