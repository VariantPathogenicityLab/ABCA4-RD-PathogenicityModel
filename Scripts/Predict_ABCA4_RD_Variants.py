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


# ----------------------- Features -----------------------

# Full feature set (18 features) — used when structure is available
features_full = [
    'SIFT', 'PolyPhen-2_HumDiv', 'AlphaMissense', 'GRANTHAM_SCORE', 'REVEL',
    'PolyPhen-2_HumVar', 'MetaRNN', 'MutScore', 'GERP',
    'MutationTaster', 'Mutation_Assessor', 'PROVEAN', 'VEST4', 'MutPred', 'gMVP',
    'MPC', 'DEOGEN2', 'LIST-S2'
]

# Reduced feature set (14 features) — used when no structure available (Protein_Structure = 0)
features_reduced = [
    'SIFT', 'GRANTHAM_SCORE', 'REVEL', 'MetaRNN', 'MutScore',
    'GERP', 'MutationTaster', 'Mutation_Assessor', 'PROVEAN',
    'VEST4', 'MutPred', 'gMVP', 'DEOGEN2', 'LIST-S2'
]


# ----------------------- Functions -----------------------

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


def get_recommended_model(protein_structure):
    """
    Determine recommended model based on Protein_Structure flag.
    1 = structure available → Full model
    0 = no structure available → Reduced model
    """
    return "Full" if protein_structure == 1 else "Reduced"


def predict_domain(input_path, model_full_dict, model_reduced_dict, output_path):
    """
    Run both full and reduced models on every variant.
    Output includes both model predictions side by side,
    with a Recommended_Model column based on Protein_Structure.
    """
    data = pd.read_csv(input_path)

    # Validate Protein_Structure column exists
    if 'Protein_Structure' not in data.columns:
        raise ValueError(
            f"'Protein_Structure' column not found in {input_path}. "
            f"Please add this column with values: 1 (structure available) or 0 (no structure)."
        )

    # Validate Protein_Structure values
    invalid = set(data['Protein_Structure'].unique()) - {0, 1}
    if invalid:
        raise ValueError(
            f"Invalid Protein_Structure values found: {invalid}. "
            f"Valid values are: 0 (no structure) or 1 (structure available)."
        )

    # ---- Full model predictions ----
    model_full  = model_full_dict["model"]
    X_full      = data[features_full].values
    probas_full = model_full.predict_proba(X_full)[:, 1]

    # ---- Reduced model predictions ----
    model_reduced  = model_reduced_dict["model"]
    X_reduced      = data[features_reduced].values
    probas_reduced = model_reduced.predict_proba(X_reduced)[:, 1]

    # ---- Recommended model based on Protein_Structure ----
    recommended_model = data['Protein_Structure'].apply(get_recommended_model)
    recommended_proba = np.where(recommended_model == "Full", probas_full, probas_reduced)

    # ---- Build output dataframe ----
    output = pd.DataFrame()
    output['Protein_Change']              = data['Protein_Change']
    output['Protein_Structure']           = data['Protein_Structure']
    output['Recommended_Model']           = recommended_model

    # Full model columns
    output['Full_Model_Probability']      = probas_full.round(4)
    output['Full_Model_ACMG_Category']    = [categorize_probability(p) for p in probas_full]

    # Reduced model columns
    output['Reduced_Model_Probability']   = probas_reduced.round(4)
    output['Reduced_Model_ACMG_Category'] = [categorize_probability(p) for p in probas_reduced]

    # Recommended prediction columns
    output['Recommended_Probability']     = recommended_proba.round(4)
    output['Recommended_ACMG_Category']   = [categorize_probability(p) for p in recommended_proba]

    output.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    print(f"  Total variants:      {len(output)}")
    print(f"  Recommended Full:    {(recommended_model == 'Full').sum()} variants (Protein_Structure = 1)")
    print(f"  Recommended Reduced: {(recommended_model == 'Reduced').sum()} variants (Protein_Structure = 0)")
    print()


# ----------------------- Load models -----------------------

model_full    = joblib.load(model_files["full"])
model_reduced = joblib.load(model_files["reduced"])

# ----------------------- Run predictions -----------------------

domains = [
    (
        os.path.join(VARIANTS_DIR, "RD1_Variants.csv"),
        os.path.join(PREDS_DIR,    "Predictions_RD1.csv")
    ),
    (
        os.path.join(VARIANTS_DIR, "RD2_Variants.csv"),
        os.path.join(PREDS_DIR,    "Predictions_RD2.csv")
    )
]

for infile, outfile in domains:
    if os.path.exists(infile):
        print(f"Processing: {os.path.basename(infile)}")
        predict_domain(infile, model_full, model_reduced, outfile)
    else:
        print(f"Warning: Input file not found — {infile}")

print("Done! All variant predictions saved to results/predictions/")
