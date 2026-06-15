# ABCA4 Regulatory Domain Pathogenicity Model

## Overview
This repository contains the code, data, and results for a domain-specific 
machine learning model designed to predict the pathogenicity of missense variants 
in the regulatory domains (RD1 and RD2) of the ABCA4 gene. A Random Forest 
classifier trained with Leave-One-Out Cross-Validation (LOOCV) and isotonic 
calibration is applied separately to structured and intrinsically disordered 
regions, using full (18-feature) and reduced (11-feature) feature sets respectively.

This work is part of a manuscript currently under review. Citation information 
will be added upon publication.

## Model Description
Two models are trained and saved:
- **Full model (18 features):** Applied to variants in structured regions of RD1 
  and RD2 where all scores are available
- **Reduced model (11 features):** Applied to variants in intrinsically disordered 
  regions of RD1 and RD2 where structural scores are excluded

Predictions are classified into five ACMG-informed categories based on predicted 
probability:

| Probability | Category |
|---|---|
| ≥ 0.90 | Likely Pathogenic |
| 0.75 – 0.89 | VUS-Likely Pathogenic |
| 0.25 – 0.74 | VUS |
| 0.11 – 0.24 | VUS-Likely Benign |
| < 0.11 | Likely Benign |

## Training Data
The model was trained on 15 curated ABCA4 regulatory domain variants (11 PLP, 
4 BLB) from ClinVar. Variants are classified as Pathogenic/Likely Pathogenic 
(PLP = 1) or Benign/Likely Benign (BLB = 0).

## Repository Structure
```
ABCA4-RD-PathogenicityModel/
├── data/
│   ├── training/
│   │   └── ABCA4_Training_Variants.csv     # Training variants with scores
│   └── variants/
│       ├── RD1_Structured_Variants.csv     # RD1 structured region VUS
│       ├── RD1_Disordered_Variants.csv     # RD1 disordered region VUS
│       ├── RD2_Structured_Variants.csv     # RD2 structured region VUS
│       └── RD2_Disordered_Variants.csv     # RD2 disordered region VUS
├── scripts/
│   ├── Train_ABCA4_RD_Model.py             # Train model, LOOCV and permutation tests
│   ├── Predict_ABCA4_RD_Variants.py        # Run predictions on new variants
│   └── Plot_Permutations.py                # Generate publication permutation figures
├── results/
│   ├── figures/
│   │   ├── loocv/                          # ROC, PR, and calibration plots
│   │   └── permutations/                   # Permutation test figures
│   ├── tables/                             # LOOCV metrics and permutation results
│   ├── models/                             # Saved .pkl model files
│   └── predictions/                        # Variant prediction outputs
└── requirements.txt
```

## Important Note on SIFT Scores
SIFT scores in this model are **inverted** from their original scale. Typically 
SIFT scores below 0.05 indicate pathogenicity. In this dataset all SIFT scores 
have been transformed as:

SIFT_SCORE = 1 - SIFT_SCORE_original

This means scores **above 0.95** indicate pathogenicity, consistent with the 
directionality of all other scores in the feature set. If you are applying this 
model to new variants you must invert your SIFT scores before running predictions.

## Input Data Format
Variant input files must contain the following columns. Score columns used by 
the full model are marked (F), reduced model (R), and both (F+R):

| Column | Model | Notes |
|---|---|---|
| `Protein_Change` | — | Variant identifier |
| `SIFT_SCORE` | F+R | Must be inverted (see above) |
| `PolyPhen-2_D_SCORE` | F | PolyPhen-2 HumDiv |
| `AM_SCORE` | F+R | AlphaMissense |
| `GRANTHAM_SCORE` | F+R | |
| `REVEL_SCORE` | F+R | |
| `PolyPhen-2_V_SCORE` | F | PolyPhen-2 HumVar |
| `MetaRNN_SCORE` | F+R | |
| `MutScore_SCORE` | F+R | |
| `GERP_SCORE` | F+R | |
| `MT_SCORE` | F+R | MutationTaster |
| `MA_SCORE` | F+R | MutationAssessor |
| `PROVEAN_SCORE` | F+R | |
| `VEST4_SCORE` | F | |
| `MutPred_SCORE` | F | |
| `gMVP_SCORE` | F | |
| `MPC_SCORE` | F | |
| `DEOGEN2_SCORE` | F+R | |
| `LIST-S2_SCORE` | F+R | |

## Requirements
- Python 3.13
- All dependencies listed in `requirements.txt`

## Installation
```bash
git clone git@github.com:VariantPathogenicityLab/ABCA4-RD-PathogenicityModel.git
cd ABCA4-RD-PathogenicityModel
pip install -r requirements.txt
```

## Usage

**Step 1 — Train the model and run validation:**
```bash
python Scripts/Train_ABCA4_RD_Model.py
```

**Step 2 — Generate publication permutation figures:**
```bash
python Scripts/Plot_Permutations.py
```

**Step 3 — Run predictions on new variants:**
```bash
python Scripts/Predict_ABCA4_RD_Variants.py
```

Steps 2 and 3 require Step 1 to be completed first.

## Results
- LOOCV performance metrics are in `results/tables/LOOCV_Metrics_FullSet.csv`
- Permutation test results are in `results/tables/Permutation_Test_Summary.csv`
- ROC, PR, and calibration figures are in `results/figures/loocv/`
- Publication permutation figures are in `results/figures/permutations/`
- Variant predictions are in `results/predictions/`

## Citation
This repository accompanies a manuscript currently under review. Citation 
information will be added upon publication.

## License
This project is licensed under the Apache License 2.0 — see the LICENSE file 
for details.