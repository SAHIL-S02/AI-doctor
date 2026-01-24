import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

def train_logistic_model(df: pd.DataFrame):
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Metrics
    auc = roc_auc_score(y_test, y_proba)

    print("ROC-AUC:", round(auc, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test, y_proba
