import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ----------------------- CONFIG -----------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE_DIR, "..", "results", "tables")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "results", "figures", "permutations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global font settings
plt.rcParams['font.family']    = 'Arial'
plt.rcParams['font.weight']    = 'bold'
plt.rcParams['axes.linewidth'] = 2.0

# Plot configurations — axis limits and tick intervals per metric and feature set
PLOT_CONFIGS = {
    ('full', 'accuracy'): {
        'xlim': (0.3, 1.0), 'ylim': (0, 1300),
        'x_maj': 0.1,  'x_min': 0.05,  'y_maj': 100, 'y_min': 20
    },
    ('full', 'precision'): {
        'xlim': (0.5, 1.0), 'ylim': (0, 1000),
        'x_maj': 0.05, 'x_min': 0.025, 'y_maj': 100, 'y_min': 20
    },
    ('full', 'roc_auc'): {
        'xlim': (0.0, 1.0), 'ylim': (0, 220),
        'x_maj': 0.1,  'x_min': 0.05,  'y_maj': 20, 'y_min': 5
    },
    ('reduced', 'accuracy'): {
        'xlim': (0.3, 1.0), 'ylim': (0, 1300),
        'x_maj': 0.1,  'x_min': 0.05,  'y_maj': 100, 'y_min': 20
    },
    ('reduced', 'precision'): {
        'xlim': (0.5, 1.0), 'ylim': (0, 1000),
        'x_maj': 0.05, 'x_min': 0.025, 'y_maj': 100, 'y_min': 20
    },
    ('reduced', 'roc_auc'): {
        'xlim': (0.0, 1.0), 'ylim': (0, 220),
        'x_maj': 0.1,  'x_min': 0.05,  'y_maj': 20, 'y_min': 5
    }
}
# ------------------------------------------------------


def create_pretty_plot(feature_set, metric_name, perm_data, true_val, p_val, cfg):
    fig, ax = plt.subplots(figsize=(7, 7))

    clean_name = "ROC AUC" if metric_name.lower() == 'roc_auc' else metric_name.replace("_", " ").title()

    counts, bins = np.histogram(perm_data, bins=35)
    ax.hist(perm_data, bins=bins, color='#a0a0a0', edgecolor='#333333', alpha=0.8, linewidth=1.2)

    ax.axvline(true_val, color='red', linestyle='--', linewidth=3,
               label=f'True {clean_name}: {true_val:.3f}')

    model_label = f"{feature_set.title()} Model"
    ax.set_title(f"Permutation Test: {clean_name}\n({model_label}, p = {p_val:.4f})",
                 fontsize=16, fontweight='bold', pad=25)

    ax.set_xlabel(clean_name,  fontsize=14, fontweight='bold', labelpad=18)
    ax.set_ylabel("Frequency", fontsize=14, fontweight='bold', labelpad=18)

    ax.set_xlim(cfg['xlim'])
    ax.set_ylim(cfg['ylim'])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(cfg['x_maj']))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(cfg['x_min']))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(cfg['y_maj']))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(cfg['y_min']))

    ax.tick_params(axis='both', which='major', labelsize=12, width=2,   length=8)
    ax.tick_params(axis='both', which='minor', width=1.5,   length=4,   color='#333333')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.legend(loc='upper left', frameon=True, edgecolor='black',
              prop={'weight': 'bold', 'size': 11})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    save_name = f"Publication_Perm_{feature_set}_{metric_name}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}")


def main():
    for (f_set, m_name), config in PLOT_CONFIGS.items():
        m_path = os.path.join(TABLES_DIR, f"Permutation_SelectedMetrics_{f_set}_3fold.csv")
        s_path = os.path.join(TABLES_DIR, f"Permutation_Summary_{f_set}_3fold.csv")

        if os.path.exists(m_path) and os.path.exists(s_path):
            df_p = pd.read_csv(m_path)
            df_s = pd.read_csv(s_path)

            if m_name in df_p.columns:
                try:
                    create_pretty_plot(
                        f_set, m_name,
                        df_p[m_name].values,
                        df_s[f"true_{m_name}"].iloc[0],
                        df_s[f"p_{m_name}"].iloc[0],
                        config
                    )
                except Exception as e:
                    print(f"Error on {f_set} {m_name}: {e}")
        else:
            print(f"Missing data files for {f_set} {m_name} — run Train_ABCA4_RD_Model.py first")

    print(f"\nDone! Publication permutation plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()