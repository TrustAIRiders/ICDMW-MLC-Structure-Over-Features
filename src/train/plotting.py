import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.calibration import calibration_curve


ARCH_COLORS = {
    "GATv2": "#1f77b4",
    "SAGE": "#ff7f0e",
    "SAGE+Edge": "#2ca02c",
    "GIN": "#d62728",
}

ARCH_LINESTYLES = {
    "GATv2": "-",
    "SAGE": "--",
    "SAGE+Edge": "-.",
    "GIN": ":",
}


def _group_by_arch(results):
    grouped = {}
    for r in results:
        arch = r["arch"]
        if arch not in grouped:
            grouped[arch] = []
        grouped[arch].append(r)
    return grouped


def plot_roc_curves(results, output_path, with_confidence=True):
    grouped = _group_by_arch(results)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    mean_fpr = np.linspace(0, 1, 200)

    for arch, runs in grouped.items():
        color = ARCH_COLORS.get(arch, "#333333")
        ls = ARCH_LINESTYLES.get(arch, "-")

        all_tprs = []
        aucs = []

        for r in runs:
            labels = np.array(r["test_labels"])
            probs = np.array(r["test_probs"])
            fpr, tpr, _ = roc_curve(labels, probs)
            auc_val = roc_auc_score(labels, probs)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            all_tprs.append(interp_tpr)
            aucs.append(auc_val)

        mean_tpr = np.mean(all_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        label = f"{arch} (AUC={mean_auc:.3f}"
        if len(runs) > 1:
            label += f"±{std_auc:.3f}"
        label += ")"

        ax.plot(mean_fpr, mean_tpr, color=color, linestyle=ls, lw=2, label=label)

        if with_confidence and len(all_tprs) > 1:
            std_tpr = np.std(all_tprs, axis=0)
            ax.fill_between(mean_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            alpha=0.15, color=color)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Architecture Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ROC curves saved to: {output_path}")


def plot_pr_curves(results, output_path, with_confidence=True):
    grouped = _group_by_arch(results)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    mean_recall = np.linspace(0, 1, 200)

    for arch, runs in grouped.items():
        color = ARCH_COLORS.get(arch, "#333333")
        ls = ARCH_LINESTYLES.get(arch, "-")

        all_precisions = []
        aucs = []

        for r in runs:
            labels = np.array(r["test_labels"])
            probs = np.array(r["test_probs"])
            precision, recall, _ = precision_recall_curve(labels, probs)
            ap = average_precision_score(labels, probs)

            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            all_precisions.append(interp_precision)
            aucs.append(ap)

        mean_precision = np.mean(all_precisions, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        label = f"{arch} (AP={mean_auc:.3f}"
        if len(runs) > 1:
            label += f"±{std_auc:.3f}"
        label += ")"

        ax.plot(mean_recall, mean_precision, color=color, linestyle=ls, lw=2, label=label)

        if with_confidence and len(all_precisions) > 1:
            std_precision = np.std(all_precisions, axis=0)
            ax.fill_between(mean_recall,
                            np.clip(mean_precision - std_precision, 0, 1),
                            np.clip(mean_precision + std_precision, 0, 1),
                            alpha=0.15, color=color)

    all_labels = np.array(results[0]["test_labels"])
    prevalence = np.mean(all_labels)
    ax.axhline(y=prevalence, color='k', linestyle='--', lw=1, alpha=0.5, label=f'Baseline ({prevalence:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — Architecture Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  PR curves saved to: {output_path}")


def plot_calibration_curves(results, output_path, n_bins=10):
    grouped = _group_by_arch(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_cal = axes[0]
    ax_hist = axes[1]

    for arch, runs in grouped.items():
        color = ARCH_COLORS.get(arch, "#333333")
        ls = ARCH_LINESTYLES.get(arch, "-")

        all_fractions = []
        all_mean_preds = []

        for r in runs:
            labels = np.array(r["test_labels"])
            probs = np.array(r["test_probs"])

            fraction_positives, mean_predicted = calibration_curve(
                labels, probs, n_bins=n_bins, strategy='uniform'
            )
            all_fractions.append(fraction_positives)
            all_mean_preds.append(mean_predicted)

        ax_cal.plot(all_mean_preds[0], all_fractions[0],
                    color=color, linestyle=ls, lw=2, marker='o', markersize=4,
                    label=arch)

        probs = np.array(runs[0]["test_probs"])
        ax_hist.hist(probs, bins=50, alpha=0.4, color=color, label=arch, density=True)

    ax_cal.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect')
    ax_cal.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax_cal.set_ylabel('Fraction of Positives', fontsize=12)
    ax_cal.set_title('Calibration Curve (Reliability Diagram)', fontsize=13)
    ax_cal.legend(loc='lower right', fontsize=10)
    ax_cal.set_xlim([0, 1])
    ax_cal.set_ylim([0, 1.05])
    ax_cal.grid(True, alpha=0.3)

    ax_hist.set_xlabel('Predicted Probability', fontsize=12)
    ax_hist.set_ylabel('Density', fontsize=12)
    ax_hist.set_title('Prediction Distribution', fontsize=13)
    ax_hist.legend(loc='upper right', fontsize=10)
    ax_hist.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Calibration curves saved to: {output_path}")


def generate_all_plots(results_path, output_dir):
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating evaluation plots")
    plot_roc_curves(results, os.path.join(output_dir, "roc_curves.png"))
    plot_pr_curves(results, os.path.join(output_dir, "pr_curves.png"))
    plot_calibration_curves(results, os.path.join(output_dir, "calibration_curves.png"))
    print("All plots generated.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--results", type=str, default="data/results/full_results.json",
                        help="Path to full_results.json")
    parser.add_argument("--output-dir", type=str, default="data/results/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    generate_all_plots(args.results, args.output_dir)
