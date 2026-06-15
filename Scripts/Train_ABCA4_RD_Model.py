import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from collections import Counter

# ----------------------- CONFIG -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input data
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "training", "ABCA4_Training_Variants.csv")

# Output directories
FIGURES_LOOCV_DIR   = os.path.join(BASE_DIR, "..", "results", "figures", "loocv")
FIGURES_PERM_DIR    = os.path.join(BASE_DIR, "..", "results", "figures", "permutations")
TABLES_DIR          = os.path.join(BASE_DIR, "..", "results", "tables")
MODELS_DIR          = os.path.join(BASE_DIR, "..", "results", "models")

for d in [FIGURES_LOOCV_DIR, FIGURES_PERM_DIR, TABLES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

N_BOOT        = 5000
CALIB_METHOD  = "isotonic"
CALIB_CV      = 3
RANDOM_STATE  = 42
PERMUTATIONS  = 1365
# ------------------------------------------------------

# Load data
df = pd.read_csv(DATA_PATH)

features_full = [
    'SIFT_SCORE', 'PolyPhen-2_D_SCORE', 'AM_SCORE', 'GRANTHAM_SCORE', 'REVEL_SCORE',
    'PolyPhen-2_V_SCORE', 'MetaRNN_SCORE', 'MutScore_SCORE', 'GERP_SCORE',
    'MT_SCORE', 'MA_SCORE', 'PROVEAN_SCORE', 'VEST4_SCORE', 'MutPred_SCORE', 'gMVP_SCORE',
    'MPC_SCORE', 'DEOGEN2_SCORE', 'LIST-S2_SCORE'
]
features_reduced = [
    'SIFT_SCORE', 'GRANTHAM_SCORE', 'REVEL_SCORE', 'MetaRNN_SCORE', 'MutScore_SCORE',
    'GERP_SCORE', 'MT_SCORE', 'MA_SCORE', 'PROVEAN_SCORE', 'DEOGEN2_SCORE', 'LIST-S2_SCORE'
]

X_full    = df[features_full].values
X_reduced = df[features_reduced].values
y         = df['Classification'].values
loo       = LeaveOneOut()

rc('font', family='Arial', size=12)


# ------------------ Helper functions ------------------

def reliability_diagram(y_true, p, n_bins=10, out_path=None):
    fig, ax1 = plt.subplots(figsize=(5, 5))
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy='uniform')
    ax1.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("True Probability")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_loocv_with_calibration(X, y, feature_set_name="full", rf_params=None, return_preds=True):
    all_y_test, all_y_proba = [], []

    if rf_params is None:
        rf_params = {}

    for train_index, test_index in loo.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf         = RandomForestClassifier(**rf_params, class_weight="balanced_subsample")
        clf        = make_pipeline(StandardScaler(), rf)
        calibrated = CalibratedClassifierCV(estimator=clf, method=CALIB_METHOD, cv=CALIB_CV)
        calibrated.fit(X_train, y_train)

        y_proba = calibrated.predict_proba(X_test)[:, 1]
        all_y_test.append(y_test[0])
        all_y_proba.append(y_proba[0])

    all_y_test  = np.array(all_y_test)
    all_y_proba = np.array(all_y_proba)
    y_pred      = (all_y_proba >= 0.5).astype(int)

    try:    roc_auc     = roc_auc_score(all_y_test, all_y_proba)
    except: roc_auc     = np.nan
    try:    pr_auc      = average_precision_score(all_y_test, all_y_proba)
    except: pr_auc      = np.nan
    try:    accuracy    = accuracy_score(all_y_test, y_pred)
    except: accuracy    = np.nan
    try:    precision   = precision_score(all_y_test, y_pred, zero_division=0)
    except: precision   = np.nan
    try:    recall      = recall_score(all_y_test, y_pred, zero_division=0)
    except: recall      = np.nan
    try:    f1          = f1_score(all_y_test, y_pred, zero_division=0)
    except: f1          = np.nan
    try:    mcc         = matthews_corrcoef(all_y_test, y_pred)
    except: mcc         = np.nan

    try:
        tn, fp, fn, tp = confusion_matrix(all_y_test, y_pred).ravel()
        specificity    = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except:
        specificity = np.nan

    # ROC figure → results/figures/loocv/
    try:
        fpr, tpr, _ = roc_curve(all_y_test, all_y_proba)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, lw=2, label=f'ROC AUC={roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_LOOCV_DIR, f"ROC_{feature_set_name}.png"))
        plt.close()
    except Exception:
        pass

    # PR curve → results/figures/loocv/
    try:
        prec, rec, _ = precision_recall_curve(all_y_test, all_y_proba)
        plt.figure(figsize=(5, 5))
        plt.plot(rec, prec, lw=2, label=f'PR AUC={pr_auc:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_LOOCV_DIR, f"PR_{feature_set_name}.png"))
        plt.close()
    except Exception:
        pass

    # Calibration plot → results/figures/loocv/
    try:
        reliability_diagram(
            all_y_test, all_y_proba,
            out_path=os.path.join(FIGURES_LOOCV_DIR, f"Calibration_{feature_set_name}.png")
        )
    except Exception:
        pass

    results = {
        "Feature_Set":  feature_set_name,
        "ROC_AUC":      roc_auc,
        "PR_AUC":       pr_auc,
        "Accuracy":     accuracy,
        "Precision":    precision,
        "Sensitivity":  recall,
        "Specificity":  specificity,
        "F1":           f1,
        "MCC":          mcc,
        "y_true":       all_y_test,
        "y_proba":      all_y_proba,
        "y_pred":       y_pred
    }

    if return_preds:
        results["per_sample"] = pd.DataFrame({
            "y_true":  all_y_test,
            "y_proba": all_y_proba,
            "y_pred":  y_pred
        })

    return results


def train_and_save(X, y, features, filename, rf_params=None):
    if rf_params is None:
        rf_params = {}
    rf         = RandomForestClassifier(**rf_params, class_weight="balanced_subsample")
    clf        = make_pipeline(StandardScaler(), rf)
    calibrated = CalibratedClassifierCV(estimator=clf, method=CALIB_METHOD, cv=CALIB_CV)
    calibrated.fit(X, y)
    save_path  = os.path.join(MODELS_DIR, filename)
    joblib.dump({"model": calibrated, "features": features}, save_path)
    print(f"Model saved to: {save_path}")
    return calibrated


def extract_rf_from_calibrated(calibrated_model):
    calibrated_clf  = calibrated_model.calibrated_classifiers_[0]
    pipeline_fitted = calibrated_clf.estimator
    return pipeline_fitted.named_steps['randomforestclassifier']


def plot_first_splitting_tree(rf_model, features, filename, max_depth=3):
    tree_to_plot = None
    for i, estimator in enumerate(rf_model.estimators_):
        if estimator.tree_.node_count > 1:
            tree_to_plot = estimator
            print(f"Using tree #{i} for plotting (node_count={estimator.tree_.node_count})")
            break

    if tree_to_plot is None:
        print("Warning: All trees are single-node stumps.")
        tree_to_plot = rf_model.estimators_[0]

    plt.figure(figsize=(20, 10))
    from sklearn.tree import plot_tree
    plot_tree(
        tree_to_plot,
        feature_names=features,
        class_names=["Benign", "Pathogenic"],
        filled=True,
        rounded=True,
        max_depth=max_depth
    )
    plt.title("Example Decision Tree from Random Forest")
    plt.savefig(os.path.join(FIGURES_LOOCV_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_usage(rf_model, features, fig_filename, csv_filename=None):
    feature_counts = Counter()
    for estimator in rf_model.estimators_:
        for f in estimator.tree_.feature:
            if f != -2:
                feature_counts[features[f]] += 1

    feat_usage_df = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['Count'])
    feat_usage_df = feat_usage_df.sort_values('Count', ascending=False)

    plt.figure(figsize=(8, 8))
    sns.barplot(x=feat_usage_df['Count'], y=feat_usage_df.index, color="skyblue")
    plt.xlabel("Number of Splits Across Forest")
    plt.ylabel("Feature")
    plt.title("Feature Usage Across Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_LOOCV_DIR, fig_filename), dpi=300)
    plt.close()

    if csv_filename:
        feat_usage_df.to_csv(os.path.join(TABLES_DIR, csv_filename))


def permutation_test_3fold(
    X, y, pipe, feature_set_name,
    n_permutations=1365,
    random_state=RANDOM_STATE,
    n_splits=3,
    save_plots=True
):
    rng = np.random.default_rng(random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    true_proba = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        true_proba[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    true_bin = (true_proba >= 0.5).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y, true_bin).ravel()
        true_specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except Exception:
        true_specificity = np.nan

    true_metrics = {
        "roc_auc":     roc_auc_score(y, true_proba),
        "accuracy":    accuracy_score(y, true_bin),
        "precision":   precision_score(y, true_bin, zero_division=0),
        "sensitivity": recall_score(y, true_bin, zero_division=0),
        "specificity": true_specificity
    }

    permuted = {k: np.zeros(n_permutations) for k in true_metrics.keys()}

    for i in range(n_permutations):
        y_perm     = rng.permutation(y)
        perm_proba = np.zeros(len(y))

        for train_idx, test_idx in skf.split(X, y_perm):
            pipe.fit(X[train_idx], y_perm[train_idx])
            perm_proba[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

        perm_bin = (perm_proba >= 0.5).astype(int)
        try:
            tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_perm, perm_bin).ravel()
            spec_p = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else 0.0
        except Exception:
            spec_p = 0.0

        permuted["roc_auc"][i]     = roc_auc_score(y_perm, perm_proba)
        permuted["accuracy"][i]    = accuracy_score(y_perm, perm_bin)
        permuted["precision"][i]   = precision_score(y_perm, perm_bin, zero_division=0)
        permuted["sensitivity"][i] = (tp_p / (tp_p + fn_p)) if (tp_p + fn_p) > 0 else 0.0
        permuted["specificity"][i] = spec_p

    p_values = {}
    for metric in true_metrics:
        p_values[metric] = (np.sum(permuted[metric] >= true_metrics[metric]) + 1) / (n_permutations + 1)

    # Save permuted arrays → results/tables/
    perm_df      = pd.DataFrame(permuted)
    perm_csv     = os.path.join(TABLES_DIR, f"Permutation_SelectedMetrics_{feature_set_name}_3fold.csv")
    perm_df.to_csv(perm_csv, index=False)

    # Save summary → results/tables/
    summary      = {f"true_{k}": v for k, v in true_metrics.items()}
    summary.update({f"p_{k}": v for k, v in p_values.items()})
    summary_csv  = os.path.join(TABLES_DIR, f"Permutation_Summary_{feature_set_name}_3fold.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    # Standard permutation histograms → results/figures/permutations/
    if save_plots:
        for metric in true_metrics:
            plt.figure(figsize=(8, 8))
            plt.hist(permuted[metric], bins=40, facecolor='gray', edgecolor='black', alpha=0.7)
            plt.axvline(true_metrics[metric], color='black', linestyle='--', linewidth=2,
                        label=f"True {metric}: {true_metrics[metric]:.3f}")
            plt.title(f"Permutation Distribution: {metric} ({feature_set_name}, 3-fold)\np = {p_values[metric]:.4f}")
            plt.xlabel(metric)
            plt.ylabel("Frequency")
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_PERM_DIR, f"Perm_{metric}_{feature_set_name}_3fold.png"), dpi=300, bbox_inches="tight")
            plt.close()

    return {
        "true_metrics": true_metrics,
        "permuted":     permuted,
        "p_values":     p_values,
        "perm_csv":     perm_csv,
        "summary_csv":  summary_csv
    }


# ----------------- Run everything -----------------

rf_base_params = {
    "n_estimators":     100,
    "min_samples_split": 2,
    "min_samples_leaf":  1,
    "random_state":      RANDOM_STATE
}

# 1) LOOCV evaluation
metrics_full = evaluate_loocv_with_calibration(X_full, y, "full", rf_params=rf_base_params)
metrics_red  = evaluate_loocv_with_calibration(X_reduced, y, "reduced", rf_params=rf_base_params)

# Save LOOCV metrics → results/tables/
metrics_summary = pd.DataFrame([
    {
        "Feature_Set": metrics_full["Feature_Set"],
        "ROC_AUC": metrics_full["ROC_AUC"], "PR_AUC": metrics_full["PR_AUC"],
        "Accuracy": metrics_full["Accuracy"], "Precision": metrics_full["Precision"],
        "Sensitivity": metrics_full["Sensitivity"], "Specificity": metrics_full["Specificity"],
        "F1": metrics_full["F1"], "MCC": metrics_full["MCC"]
    },
    {
        "Feature_Set": metrics_red["Feature_Set"],
        "ROC_AUC": metrics_red["ROC_AUC"], "PR_AUC": metrics_red["PR_AUC"],
        "Accuracy": metrics_red["Accuracy"], "Precision": metrics_red["Precision"],
        "Sensitivity": metrics_red["Sensitivity"], "Specificity": metrics_red["Specificity"],
        "F1": metrics_red["F1"], "MCC": metrics_red["MCC"]
    }
])
metrics_summary.to_csv(os.path.join(TABLES_DIR, "LOOCV_Metrics_FullSet.csv"), index=False)
print(f"LOOCV metrics saved to: {os.path.join(TABLES_DIR, 'LOOCV_Metrics_FullSet.csv')}")

# Save per-sample predictions → results/tables/
metrics_full["per_sample"].to_csv(os.path.join(TABLES_DIR, "LOOCV_Per_Sample_Preds_full.csv"), index=False)
metrics_red["per_sample"].to_csv(os.path.join(TABLES_DIR, "LOOCV_Per_Sample_Preds_reduced.csv"), index=False)


# 2) Permutation tests
pipe_full = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(**rf_base_params))])
perm_results_full = permutation_test_3fold(X_full, y, pipe_full, "full", n_permutations=PERMUTATIONS, save_plots=False)

pipe_red = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(**rf_base_params))])
perm_results_red = permutation_test_3fold(X_reduced, y, pipe_red, "reduced", n_permutations=PERMUTATIONS, save_plots=False)

# Combined permutation summary → results/tables/
perm_summary = pd.DataFrame([
    {"Feature_Set": "full",    **{f"true_{k}": v for k, v in perm_results_full["true_metrics"].items()}, **{f"p_{k}": v for k, v in perm_results_full["p_values"].items()}},
    {"Feature_Set": "reduced", **{f"true_{k}": v for k, v in perm_results_red["true_metrics"].items()},  **{f"p_{k}": v for k, v in perm_results_red["p_values"].items()}}
])
perm_summary.to_csv(os.path.join(TABLES_DIR, "Permutation_Test_Summary.csv"), index=False)
print(f"Permutation summary saved to: {os.path.join(TABLES_DIR, 'Permutation_Test_Summary.csv')}")


# 3) Train and save final models → results/models/
final_model_full    = train_and_save(X_full,    y, features_full,    "final_model_full.pkl",    rf_params=rf_base_params)
final_model_reduced = train_and_save(X_reduced, y, features_reduced, "final_model_reduced.pkl", rf_params=rf_base_params)


# 4) Extract RF and plot trees / feature usage → results/figures/loocv/
rf_fitted_full    = extract_rf_from_calibrated(final_model_full)
rf_fitted_reduced = extract_rf_from_calibrated(final_model_reduced)

plot_first_splitting_tree(rf_fitted_full,    features_full,    "Example_Tree_Full.png")
plot_first_splitting_tree(rf_fitted_reduced, features_reduced, "Example_Tree_Reduced.png")

plot_feature_usage(rf_fitted_full,    features_full,    "Feature_Usage_Full.png",    csv_filename="Feature_Usage_Full.csv")
plot_feature_usage(rf_fitted_reduced, features_reduced, "Feature_Usage_Reduced.png", csv_filename="Feature_Usage_Reduced.csv")


# 5) Print final model scores
def print_final_model_scores(metrics_dict):
    name = metrics_dict['Feature_Set']
    print(f"\n--- LOOCV Overall Performance: {name.upper()} ---")
    print(f"ROC AUC  : {metrics_dict['ROC_AUC']:.4f}")
    print(f"Accuracy : {metrics_dict['Accuracy']:.4f}")
    print(f"MCC      : {metrics_dict['MCC']:.4f}")

print_final_model_scores(metrics_full)
print_final_model_scores(metrics_red)


# 6) Save individual variant scores → results/tables/
def save_variant_scores(metrics_dict, original_df, id_col, out_name):
    results              = metrics_dict['per_sample'].copy()
    results.insert(0, id_col, original_df[id_col].values)
    results['Prediction'] = results['y_pred'].map({1: 'Pathogenic', 0: 'Benign'})
    save_path            = os.path.join(TABLES_DIR, out_name)
    results.to_csv(save_path, index=False)
    print(f"Saved individual variant scores to: {save_path}")

save_variant_scores(metrics_full, df, 'Protein_change', "Individual_Variant_Scores_Full.csv")
save_variant_scores(metrics_red,  df, 'Protein_change', "Individual_Variant_Scores_Reduced.csv")

# Misclassification check
misclassified = metrics_full['per_sample'][metrics_full['per_sample']['y_true'] != metrics_full['per_sample']['y_pred']]
print(f"\n--- Analysis ---")
print(f"Total misclassified variants in Full Set: {len(misclassified)} out of {len(df)}")
print(f"\nDone! All outputs saved to results/ subfolders.")
