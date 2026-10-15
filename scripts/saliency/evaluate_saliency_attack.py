import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


ARCH_NAMES = {
    "gatv2":     "GATv2",
    "sage":      "SAGE",
    "sage_edge": "SAGE+Edge",
    "gin":       "GIN",
}

ARCH_ORDER = ["gatv2", "sage", "sage_edge", "gin"]

MODEL_COLORS = {
    "gatv2":     "#2196F3",
    "sage":      "#4CAF50",
    "sage_edge": "#FF9800",
    "gin":       "#E91E63",
}

MODEL_MARKERS = {
    "gatv2":     "o",
    "sage":      "s",
    "sage_edge": "^",
    "gin":       "D",
}


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_rows(results: dict) -> list[dict]:
    rows = []
    for conv_type, mdata in results["models"].items():
        arch_order = ARCH_ORDER.index(conv_type) if conv_type in ARCH_ORDER else 99
        for top_k_str, tk_data in mdata.get("topk_results", {}).items():
            top_k = int(top_k_str)
            for eps_str, stats in tk_data.get("epsilon_results", {}).items():
                row = {
                    "conv_type":  conv_type,
                    "model":      ARCH_NAMES.get(conv_type, conv_type),
                    "top_k":      top_k,
                    "epsilon":    float(eps_str),
                    "arch_order": arch_order,
                }
                row.update(stats)
                row["f1_drop"]     = (row.get("post_f1_macro",           float("nan"))
                                      - row.get("clean_f1_macro",        float("nan")))
                row["recall_drop"] = (row.get("post_recall_suspicious",  float("nan"))
                                      - row.get("clean_recall_suspicious", float("nan")))
                rows.append(row)

    rows.sort(key=lambda r: (r["arch_order"], r["top_k"], r["epsilon"]))

    baseline: dict = {}
    for r in rows:
        key = (r["conv_type"], r["epsilon"])
        if key not in baseline or r["top_k"] > baseline[key][0]:
            baseline[key] = (r["top_k"], r.get("asr", float("nan")))
    for r in rows:
        key    = (r["conv_type"], r["epsilon"])
        b_asr  = baseline[key][1]
        r_asr  = r.get("asr", float("nan"))
        r["efficiency"] = (r_asr / b_asr
                           if (not np.isnan(b_asr) and b_asr > 0)
                           else float("nan"))

    return rows


def models_in_order(rows: list[dict]) -> list[str]:
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["arch_order"]):
        if r["conv_type"] not in seen:
            seen.add(r["conv_type"])
            out.append(r["conv_type"])
    return out


def plot_asr_vs_topk(rows: list[dict], outdir: str) -> None:
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))
    top_k_vals = sorted(set(r["top_k"] for r in rows))
    n_eps      = len(epsilons)

    fig, axes = plt.subplots(1, n_eps, figsize=(6 * n_eps, 4.5), sharey=True)
    if n_eps == 1:
        axes = [axes]

    for ax, eps in zip(axes, epsilons):
        for ct in conv_types:
            sub = sorted(
                [r for r in rows if r["conv_type"] == ct and r["epsilon"] == eps],
                key=lambda r: r["top_k"],
            )
            if not sub:
                continue
            xs  = [r["top_k"] for r in sub]
            ys  = [r.get("asr", float("nan")) * 100 for r in sub]

            ax.plot(
                xs, ys,
                marker=MODEL_MARKERS.get(ct, "o"),
                linewidth=2, markersize=8,
                color=MODEL_COLORS.get(ct),
                label=ARCH_NAMES.get(ct, ct),
            )
            for x, y in zip(xs, ys):
                if not np.isnan(y):
                    ax.annotate(
                        f"{y:.1f}%", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8,
                        color=MODEL_COLORS.get(ct),
                    )

        ax.set_xlabel("top_k features perturbed", fontsize=12)
        ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
        ax.set_title(f"ε = {eps:.2f}", fontsize=12, fontweight="bold")
        ax.set_xticks(top_k_vals)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(bottom=0)
        ax.legend(title="Model", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.35)

    fig.suptitle(
        "Saliency-Guided Attack — ASR vs top_k features\n"
        "(top_k=43 reproduces uniform PGD baseline)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, outdir, "saliency_asr_vs_topk.png")


def plot_efficiency(rows: list[dict], outdir: str) -> None:
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))
    max_k      = max(r["top_k"] for r in rows)
    top_k_vals = sorted(k for k in set(r["top_k"] for r in rows) if k < max_k)

    if not top_k_vals:
        return

    n_eps  = len(epsilons)
    fig, axes = plt.subplots(1, n_eps, figsize=(6 * n_eps, 4.5), sharey=True)
    if n_eps == 1:
        axes = [axes]

    for ax, eps in zip(axes, epsilons):
        for ct in conv_types:
            sub = sorted(
                [r for r in rows
                 if r["conv_type"] == ct and r["epsilon"] == eps and r["top_k"] < max_k],
                key=lambda r: r["top_k"],
            )
            if not sub:
                continue
            xs = [r["top_k"] for r in sub]
            ys = [r.get("efficiency", float("nan")) * 100 for r in sub]

            ax.plot(
                xs, ys,
                marker=MODEL_MARKERS.get(ct, "o"),
                linewidth=2, markersize=8,
                color=MODEL_COLORS.get(ct),
                label=ARCH_NAMES.get(ct, ct),
            )
            for x, y in zip(xs, ys):
                if not np.isnan(y):
                    ax.annotate(
                        f"{y:.0f}%", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8,
                        color=MODEL_COLORS.get(ct),
                    )

        ax.axhline(100, linestyle="--", color="gray", linewidth=1,
                   label=f"Baseline (k={max_k})")
        ax.set_xlabel("top_k features perturbed", fontsize=12)
        ax.set_ylabel(f"Efficiency vs k={max_k} (%)", fontsize=12)
        ax.set_title(f"ε = {eps:.2f}", fontsize=12, fontweight="bold")
        ax.set_xticks(top_k_vals)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(bottom=0)
        ax.legend(title="Model", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.35)

    fig.suptitle(
        f"Attack Efficiency  (ASR(k) / ASR(k={max_k}))\n"
        "How much of full-PGD power is achieved with only k features?",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, outdir, "saliency_efficiency.png")


def plot_asr_heatmap(rows: list[dict], outdir: str) -> None:
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))
    top_k_vals = sorted(set(r["top_k"] for r in rows))

    n_eps  = len(epsilons)
    fig, axes = plt.subplots(1, n_eps, figsize=(4 * n_eps + 2, 0.7 * len(conv_types) + 2.5))
    if n_eps == 1:
        axes = [axes]

    y_labels = [ARCH_NAMES.get(ct, ct) for ct in conv_types]
    x_labels = [str(k) for k in top_k_vals]

    for ax, eps in zip(axes, epsilons):
        matrix = np.full((len(conv_types), len(top_k_vals)), float("nan"))
        for i, ct in enumerate(conv_types):
            for j, k in enumerate(top_k_vals):
                match = [r for r in rows
                         if r["conv_type"] == ct and r["top_k"] == k
                         and r["epsilon"] == eps]
                if match:
                    matrix[i, j] = match[0].get("asr", float("nan")) * 100

        sns.heatmap(
            matrix, ax=ax,
            xticklabels=x_labels, yticklabels=y_labels,
            annot=True, fmt=".1f",
            cmap="RdYlGn_r",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "ASR (%)"},
            vmin=0, vmax=100,
        )
        ax.set_title(f"ASR (%)  — ε={eps:.2f}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("top_k features perturbed", fontsize=10)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Saliency-Guided Attack — ASR Heatmap  (model x top_k)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, outdir, "saliency_asr_heatmap.png")


def plot_importance_heatmap(results: dict, outdir: str) -> None:
    conv_types = [ct for ct in ARCH_ORDER if ct in results["models"]]
    if not conv_types:
        return

    n_features = results["node_feat_dim"]
    matrix     = np.zeros((len(conv_types), n_features))

    for i, ct in enumerate(conv_types):
        scores = np.array(
            results["models"][ct]["feature_importance"]["gradient_scores"]
        )
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            matrix[i] = (scores - s_min) / (s_max - s_min)
        else:
            matrix[i] = scores

    y_labels = [ARCH_NAMES.get(ct, ct) for ct in conv_types]
    x_labels = [str(i + 1) for i in range(n_features)]

    fig, ax = plt.subplots(figsize=(max(14, n_features * 0.35), len(conv_types) * 1.2 + 1.5))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.6, "label": "Normalised gradient saliency"},
    )
    ax.set_xlabel("Feature index", fontsize=11)
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(
        "Gradient Saliency per Feature (normalised per model)\n"
        "Darker = model relies more on this feature for 'suspicious' prediction",
        fontsize=12, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    _save(fig, outdir, "saliency_importance_heatmap.png")


def plot_method_correlation(results: dict, outdir: str) -> None:
    conv_types = [ct for ct in ARCH_ORDER if ct in results["models"]]
    n_models   = len(conv_types)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, ct in zip(axes, conv_types):
        fi      = results["models"][ct]["feature_importance"]
        grad    = np.array(fi["gradient_scores"])
        ig      = np.array(fi["ig_scores"])
        n_feat  = len(grad)

        def _norm(v):
            lo, hi = v.min(), v.max()
            return (v - lo) / (hi - lo) if hi > lo else v

        grad_n, ig_n = _norm(grad), _norm(ig)

        top10      = set(np.argsort(grad)[-10:].tolist())
        colors     = ["#E53935" if i in top10 else "#90CAF9" for i in range(n_feat)]

        ax.scatter(grad_n, ig_n, c=colors, s=40, alpha=0.85, edgecolors="none")

        corr = float(np.corrcoef(grad, ig)[0, 1])
        ax.text(0.05, 0.93, f"r = {corr:.2f}", transform=ax.transAxes,
                fontsize=10, color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
                linewidth=1, alpha=0.6)

        top5 = np.argsort(grad)[::-1][:5]
        for idx in top5:
            ax.annotate(
                f"f{idx + 1}", (grad_n[idx], ig_n[idx]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=7, color="#B71C1C", fontweight="bold",
            )

        ax.set_xlabel("Gradient saliency (norm.)", fontsize=10)
        ax.set_ylabel("Integrated gradients (norm.)", fontsize=10)
        ax.set_title(ARCH_NAMES.get(ct, ct), fontsize=11, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    from matplotlib.patches import Patch
    handles = [
        Patch(color="#E53935", label="Top-10 (gradient)"),
        Patch(color="#90CAF9", label="Others"),
    ]
    axes[-1].legend(handles=handles, fontsize=9, framealpha=0.9,
                    loc="lower right")

    fig.suptitle(
        "Feature Importance Method Correlation  (Gradient vs Integrated Gradients)\n"
        "Points near the diagonal = both methods agree",
        fontsize=12, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    _save(fig, outdir, "saliency_method_correlation.png")


def print_robustness_analysis(rows: list[dict]) -> None:
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))
    top_k_vals = sorted(set(r["top_k"] for r in rows))
    max_k      = max(top_k_vals)

    print("\n" + "=" * 65)
    print("  ROBUSTNESS ANALYSIS — Saliency-Guided Attack")
    print("=" * 65)

    model_stats: dict = {}
    for ct in conv_types:
        m_rows = [r for r in rows if r["conv_type"] == ct]
        asrs   = [r.get("asr", float("nan")) for r in m_rows]
        model_stats[ct] = {
            "mean_asr": np.nanmean(asrs),
            "max_asr":  np.nanmax(asrs),
        }

    ranked = sorted(model_stats.items(), key=lambda kv: kv[1]["mean_asr"])
    print(f"\n  {'Rank':<5} {'Model':<14} {'Mean ASR':>9} {'Max ASR':>9}")
    print("  " + "-" * 42)
    for rank, (ct, st) in enumerate(ranked, 1):
        tag = "  ← most robust"   if rank == 1          else \
              "  ← least robust"  if rank == len(ranked) else ""
        print(f"  {rank:<4} {ARCH_NAMES.get(ct, ct):<14} "
              f"{st['mean_asr']:>8.1%} {st['max_asr']:>8.1%}{tag}")

    print(f"\n  ASR at k={max_k} (uniform PGD baseline) vs smallest k:")
    for eps in epsilons:
        min_k = min(top_k_vals)
        print(f"    ε = {eps:.2f}:")
        for ct in conv_types:
            base = next(
                (r.get("asr", float("nan")) for r in rows
                 if r["conv_type"] == ct and r["top_k"] == max_k
                 and r["epsilon"] == eps), float("nan")
            )
            selective = next(
                (r.get("asr", float("nan")) for r in rows
                 if r["conv_type"] == ct and r["top_k"] == min_k
                 and r["epsilon"] == eps), float("nan")
            )
            eff = selective / base if (not np.isnan(base) and base > 0) else float("nan")
            print(f"      {ARCH_NAMES.get(ct, ct):<12} "
                  f"k={min_k}: {selective:.1%}  k={max_k}: {base:.1%}  "
                  f"efficiency={eff:.0%}")


def print_feature_agreement(results: dict, conv_types: list[str]) -> None:
    print("\n" + "=" * 65)
    print("  FEATURE AGREEMENT ACROSS MODELS (top-10 gradient, 1-based)")
    print("=" * 65)

    all_top10: dict = {}
    for ct in conv_types:
        if ct not in results["models"]:
            continue
        top10 = results["models"][ct]["feature_importance"][
            "top_features_gradient"
        ][:10]
        all_top10[ct] = set(top10)
        print(f"\n  {ARCH_NAMES.get(ct, ct):12s}: {sorted(top10)}")

    if len(all_top10) >= 2:
        shared = set.intersection(*all_top10.values())
        print(f"\n  Shared by ALL {len(all_top10)} models: "
              f"{sorted(shared) if shared else '∅ (none)'}")

        models_list = list(all_top10.keys())
        if len(models_list) >= 2:
            print("\n  Pairwise Jaccard (top-10):")
            for i in range(len(models_list)):
                for j in range(i + 1, len(models_list)):
                    a, b = all_top10[models_list[i]], all_top10[models_list[j]]
                    jac  = len(a & b) / len(a | b)
                    print(f"    {ARCH_NAMES[models_list[i]]:12s} ∩ "
                          f"{ARCH_NAMES[models_list[j]]:12s}  = "
                          f"{len(a & b)}/10  Jaccard = {jac:.2f}")


def _save(fig, outdir: str, filename: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saliency-guided feature attack results"
    )
    parser.add_argument(
        "--results",
        default="data/results/saliency_attack_results.json",
        help="Path to saliency_attack_results.json",
    )
    parser.add_argument(
        "--plots_dir",
        default="data/results/plots/saliency",
        help="Output directory for PNG plots",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        print("Run scripts/run_saliency_attack.py first.")
        sys.exit(1)

    os.makedirs(args.plots_dir, exist_ok=True)
    setup_style()

    print(f"\n=== Saliency Attack Evaluation ===")
    print(f"  Reading: {args.results}")

    results    = load_results(args.results)
    rows       = build_rows(results)
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))
    top_k_vals = sorted(set(r["top_k"] for r in rows))

    if not rows:
        print("ERROR: No result rows found. Check that the attack ran successfully.")
        sys.exit(1)

    print(f"  Models:    {[ARCH_NAMES.get(ct, ct) for ct in conv_types]}")
    print(f"  Epsilons:  {epsilons}")
    print(f"  top_k:     {top_k_vals}")
    print(f"  Total rows (model x top_k x ε): {len(rows)}")


    print("\n--- [1/5] ASR vs top_k curve ---")
    plot_asr_vs_topk(rows, args.plots_dir)

    print("\n--- [2/5] Efficiency ratio ---")
    plot_efficiency(rows, args.plots_dir)

    print("\n--- [3/5] ASR heatmap (model x top_k) ---")
    plot_asr_heatmap(rows, args.plots_dir)

    print("\n--- [4/5] Feature importance heatmap ---")
    plot_importance_heatmap(results, args.plots_dir)

    print("\n--- [5/5] Method correlation (gradient vs IG) ---")
    plot_method_correlation(results, args.plots_dir)

    print_robustness_analysis(rows)
    print_feature_agreement(results, conv_types)

    print("\n" + "=" * 65)
    print("Done. Outputs:")
    print(f"  {args.summary}")
    for fname in [
        "saliency_asr_vs_topk.png",
        "saliency_efficiency.png",
        "saliency_asr_heatmap.png",
        "saliency_importance_heatmap.png",
        "saliency_method_correlation.png",
    ]:
        print(f"  {args.plots_dir}/{fname}")
    print("=" * 65)


if __name__ == "__main__":
    main()
