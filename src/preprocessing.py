import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # 1. Encode gender
    # -----------------------------
    df['gender'] = df['gender'].map({
        'Male': 1,
        'Female': 0
    })

    # -----------------------------
    # 2. Encode smoking history (one-hot)
    # -----------------------------
    df = pd.get_dummies(
        df,
        columns=['smoking_history'],
        drop_first=True
    )

    # -----------------------------
    # 3. BMI category (clinically interpretable)
    # -----------------------------
    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]  # under, normal, overweight, obese
    )

    # -----------------------------
    # 4. Glycemic risk flags
    # -----------------------------
    df['prediabetes_flag'] = (
        (df['HbA1c_level'] >= 5.7) &
        (df['HbA1c_level'] < 6.5)
    ).astype(int)

    df['high_glucose_flag'] = (
        df['blood_glucose_level'] >= 140
    ).astype(int)

    # -----------------------------
    # 5. Final cleanup
    # -----------------------------
    df = df.dropna()  # safe here; dataset is clean

    return df
