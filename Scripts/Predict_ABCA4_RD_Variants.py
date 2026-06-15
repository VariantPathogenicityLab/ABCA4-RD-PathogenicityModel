import pandas as pd
import joblib
import numpy as np
import os

# ----------------------- CONFIG -----------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "..", "results", "models")
VARIANTS_DIR = os.path.join(BASE_DIR, "..", "data", "variants")
PREDS_DIR    = os.path.join(BASE_DIR, "..", "results", "predictions")

os.makedirs(PREDS_DIR, exist_ok=True)
# ------------------------------------------------------

# Check model files exist before proceeding
model_files = {
    "full":    os.path.join(MODELS_DIR, "final_model_full.pkl"),
    "reduced": os.path.join(MODELS_DIR, "final_model_reduced.pkl")
}

missing = [f for f in model_files.values() if not os.path.exists(f)]
if missing:
    print("Error: The following model files are missing:")
    for f in missing:
        print(f"  - {f}")
    print("\nPlease run Train_ABCA4_RD_Model.py first to generate the models.")
    exit(1)


# ----------------------- Functions -----------------------

def classify_with_abstain(proba, t_pos=0.9, t_neg=0.1):
    if proba >= t_pos:
        return 1
    elif proba <= t_neg:
        return 0
    else:
        return -1


def categorize_probability(p):
    if p >= 0.90:
        return "Likely Pathogenic"
    elif 0.75 <= p < 0.90:
        return "VUS-Likely Pathogenic"
    elif 0.25 <= p < 0.75:
        return "VUS"
    elif 0.11 <= p < 0.25:
        return "VUS-Likely Benign"
    else:
        return "Likely Benign"


def predict_and_save(input_path, model_dict, output_path, t_pos=0.9, t_neg=0.1):
    model    = model_dict["model"]
    features = model_dict["features"]
    data     = pd.read_csv(input_path)
    X_new    = data[features].values
    probas   = model.predict_proba(X_new)[:, 1]
    labels   = np.array([classify_with_abstain(p, t_pos, t_neg) for p in probas])

    data["Predicted_Label_HC"]      = labels
    data["Predicted_Probability"]   = probas
    data["Predicted_ACMG_Category"] = [categorize_probability(p) for p in probas]
    data.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


# ----------------------- Load models -----------------------

model_full    = joblib.load(model_files["full"])
model_reduced = joblib.load(model_files["reduced"])

# ----------------------- Run predictions -----------------------
# RD1 — Structured region (no intrinsic disorder) → uses full model (18 features)
# RD1 — Disordered region (intrinsically disordered) → uses reduced model (11 features)
# RD2 — Structured region → uses full model
# RD2 — Disordered region → uses reduced model

external_files = [
    (
        os.path.join(VARIANTS_DIR, "RD1_Structured_Variants.csv"),
        os.path.join(PREDS_DIR,    "Predictions_RD1_Structured.csv"),
        model_full
    ),
    (
        os.path.join(VARIANTS_DIR, "RD1_Disordered_Variants.csv"),
        os.path.join(PREDS_DIR,    "Predictions_RD1_Disordered.csv"),
        model_reduced
    ),
    (
        os.path.join(VARIANTS_DIR, "RD2_Structured_Variants.csv"),
        os.path.join(PREDS_DIR,    "Predictions_RD2_Structured.csv"),
        model_full
    ),
    (
        os.path.join(VARIANTS_DIR, "RD2_Disordered_Variants.csv"),
        os.path.join(PREDS_DIR,    "Predictions_RD2_Disordered.csv"),
        model_reduced
    )
]

for infile, outfile, model_used in external_files:
    predict_and_save(infile, model_used, outfile)

print("\nDone! All variant predictions saved to results/predictions/")
