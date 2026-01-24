import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

from src.preprocessing import preprocess
from src.explainability import explain_patient

# -------------------------------
# Load models
# -------------------------------
base_model = joblib.load("models/logistic_base.pkl")
calibrated_model = joblib.load("models/logistic_calibrated.pkl")
X_background = joblib.load("models/background.pkl")
feature_schema = joblib.load("models/feature_schema.pkl")


# -------------------------------
# Risk band logic
# -------------------------------
def risk_band(p):
    if p < 0.10:
        return "Low"
    elif p < 0.25:
        return "Moderate"
    else:
        return "High"

# -------------------------------
# Predict function
# -------------------------------
def predict():
    try:
        patient = {
            "gender": gender_var.get(),
            "age": int(age_var.get()),
            "hypertension": int(hyper_var.get()),
            "heart_disease": int(heart_var.get()),
            "smoking_history": smoke_var.get(),
            "bmi": float(bmi_var.get()),
            "HbA1c_level": float(hba1c_var.get()),
            "blood_glucose_level": float(glucose_var.get())
        }

        df = pd.DataFrame([patient])
        df_p = preprocess(df)

        # ---- ALIGN FEATURES ----
        for col in feature_schema:
            if col not in df_p.columns:
                df_p[col] = 0

        df_p = df_p[feature_schema]


        prob = calibrated_model.predict_proba(df_p)[0][1]
        band = risk_band(prob)

        explanation = explain_patient(
            base_model,
            X_background,
            df_p
        )

        result_text.set(
            f"Early Diabetes Risk: {band}\n"
            f"Estimated Risk: {prob:.1%}\n\n"
            f"Key Risk Drivers:\n" +
            "\n".join(
                f"- {row.feature}"
                for _, row in explanation.iterrows()
            )
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

# -------------------------------
# Tkinter UI
# -------------------------------
root = tk.Tk()
root.title("Early Diabetes Risk Screening Tool")
root.geometry("520x520")

# Variables
gender_var = tk.StringVar(value="Female")
smoke_var = tk.StringVar(value="never")

age_var = tk.StringVar()
bmi_var = tk.StringVar()
hba1c_var = tk.StringVar()
glucose_var = tk.StringVar()

hyper_var = tk.IntVar()
heart_var = tk.IntVar()

result_text = tk.StringVar()

# -------------------------------
# Layout
# -------------------------------
ttk.Label(root, text="Patient Information", font=("Arial", 14, "bold")).pack(pady=10)

form = ttk.Frame(root)
form.pack(pady=5)

def field(label, widget):
    row = ttk.Frame(form)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text=label, width=22).pack(side="left")
    widget.pack(side="right", fill="x", expand=True)

field("Gender", ttk.Combobox(form, textvariable=gender_var,
      values=["Male", "Female"], state="readonly"))

field("Age", ttk.Entry(form, textvariable=age_var))
field("BMI", ttk.Entry(form, textvariable=bmi_var))
field("HbA1c", ttk.Entry(form, textvariable=hba1c_var))
field("Blood Glucose", ttk.Entry(form, textvariable=glucose_var))

field("Smoking History", ttk.Combobox(
    form,
    textvariable=smoke_var,
    values=["never", "former", "current", "No Info"],
    state="readonly"
))

ttk.Checkbutton(form, text="Hypertension", variable=hyper_var).pack(anchor="w")
ttk.Checkbutton(form, text="Heart Disease", variable=heart_var).pack(anchor="w")

ttk.Button(root, text="Predict Risk", command=predict).pack(pady=10)

ttk.Label(root, textvariable=result_text, wraplength=480,
          font=("Arial", 10)).pack(pady=10)

root.mainloop()
