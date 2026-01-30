import shap
import pandas as pd

def explain_patient(base_model, X_background, X_sample, top_k=5):

    explainer = shap.LinearExplainer(
        base_model,
        X_background
    )

    shap_values = explainer.shap_values(X_sample)

    explanation = pd.DataFrame({
        "feature": X_sample.columns,
        "impact": shap_values[0]
    })

    explanation["abs_impact"] = explanation["impact"].abs()

    explanation = explanation.sort_values(
        by="abs_impact",
        ascending=False
    )

    return explanation.head(top_k)
