from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

def calibrate_model(base_model, X_train, y_train, X_test, y_test):
    calibrated = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv=5
    )

    calibrated.fit(X_train, y_train)

    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_proba_cal)

    print("Brier Score (lower is better):", round(brier, 4))

    return calibrated, y_proba_cal
