import pandas as pd
from src.preprocessing import preprocess
from src.model import train_logistic_model
from src.calibration import calibrate_model
from src.explainability import explain_patient
df = pd.read_csv("data/dataset.csv")
df = preprocess(df)


model, X_train, X_test, y_train, y_test, y_proba = train_logistic_model(df)

calibrated_model, y_proba_cal = calibrate_model(
    model, X_train, y_train, X_test, y_test
)




# background for SHAP (small sample is enough)
X_background = X_train.sample(100, random_state=42)

# explain one patient
sample = X_test.iloc[[0]]

explanation = explain_patient(
    base_model=model,          # UNCALIBRATED model
    X_background=X_background,
    X_sample=sample
)

print(explanation)

from src.bias_checks import subgroup_auc

# We already have these from earlier steps:
# X_train, X_test, y_train, y_test

# Gender bias check
gender_auc = subgroup_auc(
    X=X_test,
    y=y_test,
    model=calibrated_model,
    subgroup_series=df.loc[X_test.index, 'gender']
)

print("Gender AUC:", gender_auc)

# Age bucket bias check
age_bucket = pd.cut(
    df.loc[X_test.index, 'age'],
    bins=[0, 40, 60, 100],
    labels=['<40', '40–60', '60+']
)

age_auc = subgroup_auc(
    X=X_test,
    y=y_test,
    model=calibrated_model,
    subgroup_series=age_bucket
)

print("Age-group AUC:", age_auc)


from src.save_model import save_models

save_models(
    base_model=model,
    calibrated_model=calibrated_model,
    X_background=X_train.sample(100, random_state=42)
)
from src.save_schema import save_feature_schema

save_feature_schema(X_train)
