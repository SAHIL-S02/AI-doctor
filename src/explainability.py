import shap
import pandas as pd

def explain_patient(base_model, X_background, X_sample, top_k=5):
    """
    base_model: uncalibrated LogisticRegression model
    X_background: background dataset (e.g. X_train sample)
    X_sample: single-row DataFrame
    """

    explainer = shap.LinearExplainer(
        base_model,
        X_background,
        feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(X_sample)

    explanation = pd.DataFrame({
        "feature": X_sample.columns,
        "impact": shap_values[0]
    })

    explanation["abs_impact"] = explanation["impact"].abs()

    explanation = explanation.sort_values(
        by="abs_impact", ascending=False
    )

    return explanation.head(top_k)
