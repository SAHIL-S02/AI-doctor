import pandas as pd
from sklearn.metrics import roc_auc_score

def subgroup_auc(X, y, model, subgroup_series):
    results = {}

    for group in subgroup_series.unique():
        idx = subgroup_series == group

        if idx.sum() < 500:
            continue

        X_sub = X[idx]
        y_sub = y[idx]

        y_proba = model.predict_proba(X_sub)[:, 1]
        auc = roc_auc_score(y_sub, y_proba)

        results[group] = round(auc, 4)

    return results
