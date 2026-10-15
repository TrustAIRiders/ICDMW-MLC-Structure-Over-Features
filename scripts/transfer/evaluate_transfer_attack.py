import argparse
import json
import os
import sys
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
    "gcn":       "GCN (surrogate)",
}

ARCH_ORDER = ["gatv2", "sage", "sage_edge", "gin"]

MODEL_COLORS = {
    "gatv2":     "#2196F3",
    "sage":      "#4CAF50",
    "sage_edge": "#FF9800",
    "gin":       "#E91E63",
    "gcn":       "#9C27B0",
}


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_transfer_heatmap(transfer_results, outdir):
    epsilons = [str(e) for e in transfer_results["epsilons"]]
    targets  = ARCH_ORDER

    gcn_data       = transfer_results.get("gcn_to_targets", {})
    cross_data     = transfer_results.get("cross_arch_matrix", {})
    has_cross_arch = bool(cross_data)

    surrogates = ["gcn"] + [ct for ct in ARCH_ORDER if ct in cross_data]
    surr_labels = [ARCH_NAMES.get(s, s) for s in surrogates]
    tgt_labels  = [ARCH_NAMES.get(t, t) for t in targets]

    n_eps = len(epsilons)
    fig, axes = plt.subplots(1, n_eps, figsize=(5.5 * n_eps, max(3.5, 0.7 * len(surrogates) + 2.5)))
    if n_eps == 1:
        axes = [axes]

    for ax, eps_str in zip(axes, epsilons):
        matrix = np.full((len(surrogates), len(targets)), np.nan)

        for i, surr in enumerate(surrogates):
            for j, tgt in enumerate(targets):
                if surr == "gcn":
                    val = gcn_data.get(eps_str, {}).get(tgt, {}).get("transfer_asr", np.nan)
                elif surr == tgt:
                    val = np.nan
                else:
                    val = (cross_data.get(surr, {})
                                     .get(eps_str, {})
                                     .get(tgt, {})
                                     .get("transfer_asr", np.nan))
                matrix[i, j] = val if val is not None else np.nan

        mask = np.isnan(matrix)

        sns.heatmap(
            matrix * 100,
            ax=ax,
            mask=mask,
            xticklabels=tgt_labels,
            yticklabels=surr_labels,
            annot=True, fmt=".1f",
            cmap="YlOrRd",
            vmin=0, vmax=100,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Transfer ASR (%)", "shrink": 0.8},
        )

        for i in range(len(surrogates)):
            for j in range(len(targets)):
                if mask[i, j]:
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1, fill=True, color="#DDDDDD", zorder=3
                    ))
                    ax.text(j + 0.5, i + 0.5, "—",
                            ha="center", va="center", fontsize=10,
                            color="#999999", zorder=4)

        ax.set_title(f"ε = {eps_str}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Target Model", fontsize=10)
        ax.set_ylabel("Surrogate Model" if eps_str == epsilons[0] else "", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Transfer Attack — ASR Matrix (surrogate × target)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    outpath = os.path.join(outdir, "transfer_heatmap.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_transfer_vs_whitebox(transfer_results, whitebox_results, outdir):

    epsilons = transfer_results["epsilons"]
    wb_epsilons = set()
    for mdata in whitebox_results.get("models", {}).values():
        wb_epsilons |= {float(e) for e in mdata.get("epsilon_results", {}).keys()}
    common_eps = sorted(set(epsilons) & wb_epsilons)

    if not common_eps:
        print("  [!] No overlapping epsilons between transfer and white-box results — skipping plot 2.")
        return

    targets = [ct for ct in ARCH_ORDER if ct in whitebox_results.get("models", {})]
    gcn_data = transfer_results.get("gcn_to_targets", {})

    n_targets = len(targets)
    n_eps     = len(common_eps)
    bar_w     = 0.35
    group_gap = 0.15

    fig, axes = plt.subplots(1, n_eps, figsize=(max(6, 2.8 * n_targets) * n_eps, 4.8),
                             sharey=True)
    if n_eps == 1:
        axes = [axes]

    for ax, eps in zip(axes, common_eps):
        eps_str = str(eps)
        x = np.arange(n_targets)

        wb_asrs  = []
        tr_asrs  = []
        for ct in targets:
            wb  = (whitebox_results["models"]
                   .get(ct, {})
                   .get("epsilon_results", {})
                   .get(eps_str, {})
                   .get("asr", np.nan))
            tr  = (gcn_data
                   .get(eps_str, {})
                   .get(ct, {})
                   .get("transfer_asr", np.nan))
            wb_asrs.append(wb * 100 if not np.isnan(wb) else np.nan)
            tr_asrs.append(tr * 100 if not np.isnan(tr) else np.nan)

        bars_wb = ax.bar(x - bar_w / 2, wb_asrs, bar_w,
                         label="White-box (PGD direct)",
                         color=[MODEL_COLORS.get(ct, "#777") for ct in targets],
                         alpha=0.9, edgecolor="white")
        bars_tr = ax.bar(x + bar_w / 2, tr_asrs, bar_w,
                         label="Gray-box (GCN surrogate)",
                         color=[MODEL_COLORS.get(ct, "#777") for ct in targets],
                         alpha=0.40, hatch="//", edgecolor="grey")

        for bar, v in zip(list(bars_wb) + list(bars_tr),
                          wb_asrs + tr_asrs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_NAMES.get(ct, ct) for ct in targets],
                           fontsize=10, rotation=15, ha="right")
        ax.set_title(f"ε = {eps}", fontsize=12, fontweight="bold")
        ax.set_ylabel("ASR (%)" if eps == common_eps[0] else "", fontsize=11)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.35)

    axes[0].legend(framealpha=0.9, fontsize=9, loc="upper left")
    fig.suptitle("White-box vs Gray-box Transfer ASR  (per target model)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    outpath = os.path.join(outdir, "transfer_vs_whitebox.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_arch_pair_ranking(transfer_results, outdir):
    cross_data = transfer_results.get("cross_arch_matrix", {})
    if not cross_data:
        print("  [!] No cross_arch_matrix in results — skipping plot 3.")
        return

    epsilons = [str(e) for e in transfer_results["epsilons"]]

    pair_asrs = {}
    for surr_conv, eps_dict in cross_data.items():
        for tgt_conv in ARCH_ORDER:
            if tgt_conv == surr_conv:
                continue
            asrs = []
            for eps_str in epsilons:
                val = (eps_dict.get(eps_str, {})
                               .get(tgt_conv, {})
                               .get("transfer_asr", None))
                if val is not None:
                    asrs.append(val)
            if asrs:
                pair_asrs[(surr_conv, tgt_conv)] = np.mean(asrs)

    if not pair_asrs:
        print("  [!] No valid cross-arch pairs found — skipping plot 3.")
        return

    sorted_pairs = sorted(pair_asrs.items(), key=lambda kv: kv[1], reverse=True)
    pair_labels = [f"{ARCH_NAMES.get(s,'?')} → {ARCH_NAMES.get(t,'?')}"
                   for (s, t), _ in sorted_pairs]
    asr_vals    = [v * 100 for _, v in sorted_pairs]
    bar_colors  = [MODEL_COLORS.get(s, "#999") for (s, _), _ in sorted_pairs]

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(pair_labels)), 4.8))

    bars = ax.bar(range(len(pair_labels)), asr_vals,
                  color=bar_colors, alpha=0.85, edgecolor="white", width=0.6)

    for bar, v in zip(bars, asr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    if len(asr_vals) >= 2:
        bars[0].set_edgecolor("#B71C1C")
        bars[0].set_linewidth(2.5)
        bars[-1].set_edgecolor("#1B5E20")
        bars[-1].set_linewidth(2.5)

    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Transfer ASR (%) across ε", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.35)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS.get(ct, "#999"), alpha=0.85)
        for ct in ARCH_ORDER if ct in cross_data
    ]
    legend_labels = [ARCH_NAMES.get(ct, ct) for ct in ARCH_ORDER if ct in cross_data]
    ax.legend(legend_handles, legend_labels,
              title="Surrogate arch", framealpha=0.9, fontsize=9, loc="upper right")

    ax.set_title("Cross-Architecture Transfer — Pair Ranking\n"
                 "(mean ASR across ε, red outline = highest, green = lowest)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    outpath = os.path.join(outdir, "transfer_arch_pairs.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def print_transfer_analysis(transfer_results, whitebox_results):
    epsilons = [str(e) for e in transfer_results["epsilons"]]
    gcn_data = transfer_results.get("gcn_to_targets", {})
    cross_data = transfer_results.get("cross_arch_matrix", {})
    wb_models = whitebox_results.get("models", {}) if whitebox_results else {}

    print("\n" + "=" * 65)
    print("  TRANSFER ATTACK ANALYSIS")
    print("=" * 65)

    print(f"\n[1] GCN Surrogate → All Targets")
    print(f"    {'Target':<14}", end="")
    for eps_str in epsilons:
        print(f"  ε={float(eps_str):<7}", end="")
    print(f"  {'Mean ASR':>9}")
    print("    " + "-" * (14 + 11 * len(epsilons) + 11))

    for ct in ARCH_ORDER:
        asrs = []
        row = f"    {ARCH_NAMES.get(ct, ct):<14}"
        for eps_str in epsilons:
            asr = gcn_data.get(eps_str, {}).get(ct, {}).get("transfer_asr", np.nan)
            asrs.append(asr if asr is not None else np.nan)
            row += f"  {asr*100:>7.2f}%" if not np.isnan(asr) else f"  {'N/A':>7}"
        mean_asr = np.nanmean(asrs)
        row += f"  {mean_asr*100:>8.2f}%"
        print(row)

    if wb_models:
        print(f"\n[2] Transferability Gap  (white-box ASR − transfer ASR)")

        wb_eps_set = set()
        for mdata in wb_models.values():
            wb_eps_set |= {str(e) for e in mdata.get("epsilon_results", {}).keys()}
        common = [e for e in epsilons if e in wb_eps_set]

        if common:
            print(f"    {'Target':<14}  {'WB ASR':>8}  {'TR ASR':>8}  {'Gap':>8}  {'TR/WB':>8}")
            print("    " + "-" * 55)
            for ct in ARCH_ORDER:
                if ct not in wb_models:
                    continue
                wb_vals, tr_vals = [], []
                for eps_str in common:
                    wb = wb_models[ct]["epsilon_results"].get(eps_str, {}).get("asr", np.nan)
                    tr = gcn_data.get(eps_str, {}).get(ct, {}).get("transfer_asr", np.nan)
                    if wb is not None: wb_vals.append(wb)
                    if tr is not None and tr is not np.nan: tr_vals.append(tr)
                mean_wb = np.nanmean(wb_vals)
                mean_tr = np.nanmean(tr_vals)
                gap     = mean_wb - mean_tr
                ratio   = mean_tr / mean_wb if mean_wb > 0 else np.nan
                print(f"    {ARCH_NAMES.get(ct, ct):<14}  {mean_wb*100:>7.2f}%  "
                      f"{mean_tr*100:>7.2f}%  {gap*100:>+7.2f}%  {ratio:>7.2f}x")

    if cross_data:
        print(f"\n[3] Cross-Architecture Transfer — Best & Worst Pairs")
        pair_asrs = {}
        for surr, eps_dict in cross_data.items():
            for tgt in ARCH_ORDER:
                if tgt == surr:
                    continue
                asrs = []
                for eps_str in epsilons:
                    v = eps_dict.get(eps_str, {}).get(tgt, {}).get("transfer_asr", None)
                    if v is not None:
                        asrs.append(v)
                if asrs:
                    pair_asrs[(surr, tgt)] = np.mean(asrs)

        sorted_pairs = sorted(pair_asrs.items(), key=lambda kv: kv[1], reverse=True)

        print(f"\n    {'Pair (surrogate → target)':<30}  {'Mean ASR':>9}")
        print("    " + "-" * 42)
        for (surr, tgt), asr in sorted_pairs:
            tag = ""
            if (surr, tgt) == sorted_pairs[0][0]:
                tag = "  ← highest transfer"
            elif (surr, tgt) == sorted_pairs[-1][0]:
                tag = "  ← lowest transfer"
            print(f"    {ARCH_NAMES.get(surr,'?'):<14} → {ARCH_NAMES.get(tgt,'?'):<14}"
                  f"  {asr*100:>8.2f}%{tag}")

        print(f"\n[4] Architectural Influence on Transferability")
        print()

        surr_mean = {}
        for surr in ARCH_ORDER:
            pairs = [(s, t) for (s, t) in pair_asrs if s == surr]
            if pairs:
                surr_mean[surr] = np.mean([pair_asrs[p] for p in pairs])
        if surr_mean:
            best_surr  = max(surr_mean, key=surr_mean.get)
            worst_surr = min(surr_mean, key=surr_mean.get)
            print(f"  Best surrogate  (highest avg transfer out): "
                  f"{ARCH_NAMES[best_surr]} ({surr_mean[best_surr]*100:.2f}%)")
            print(f"  Worst surrogate (lowest  avg transfer out): "
                  f"{ARCH_NAMES[worst_surr]} ({surr_mean[worst_surr]*100:.2f}%)")

        tgt_mean = {}
        for tgt in ARCH_ORDER:
            pairs = [(s, t) for (s, t) in pair_asrs if t == tgt]
            if pairs:
                tgt_mean[tgt] = np.mean([pair_asrs[p] for p in pairs])
        if tgt_mean:
            easiest = max(tgt_mean, key=tgt_mean.get)
            hardest = min(tgt_mean, key=tgt_mean.get)
            print(f"  Easiest target  (highest avg transfer in):  "
                  f"{ARCH_NAMES[easiest]} ({tgt_mean[easiest]*100:.2f}%)")
            print(f"  Hardest target  (lowest  avg transfer in):  "
                  f"{ARCH_NAMES[hardest]} ({tgt_mean[hardest]*100:.2f}%)")

        if wb_models:
            gcn_mean_asrs = []
            for eps_str in epsilons:
                for ct in ARCH_ORDER:
                    v = gcn_data.get(eps_str, {}).get(ct, {}).get("transfer_asr", None)
                    if v is not None:
                        gcn_mean_asrs.append(v)
            gcn_mean = np.nanmean(gcn_mean_asrs) if gcn_mean_asrs else np.nan
            best_cross_mean = max(surr_mean.values()) if surr_mean else np.nan
            print()
            print(f"  GCN surrogate avg transfer ASR:  {gcn_mean*100:.2f}%")
            print(f"  Best arch surrogate avg ASR:     {best_cross_mean*100:.2f}%")
            if not np.isnan(gcn_mean) and not np.isnan(best_cross_mean):
                if best_cross_mean > gcn_mean:
                    print(f"  → Intra-family transfer ({ARCH_NAMES[best_surr]}) "
                          f"outperforms cross-family (GCN) by "
                          f"{(best_cross_mean - gcn_mean)*100:+.2f}pp")
                else:
                    print(f"  → GCN surrogate outperforms all intra-family "
                          f"surrogates by {(gcn_mean - best_cross_mean)*100:+.2f}pp — "
                          f"suggests decision boundaries are surprisingly similar across families.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise transfer attack results"
    )
    parser.add_argument("--results",
                        default="data/results/transfer_attack_results.json",
                        help="Path to transfer_attack_results.json")
    parser.add_argument("--whitebox_results",
                        default="data/results/pgd_attack_results.json",
                        help="Path to pgd_attack_results.json for comparison")
    parser.add_argument("--plots_dir",
                        default="data/results/plots",
                        help="Output directory for PNG plots")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: {args.results} not found.")
        print("Run scripts/run_transfer_attack.py first.")
        sys.exit(1)

    os.makedirs(args.plots_dir, exist_ok=True)
    setup_style()

    print(f"\n=== Transfer Attack Evaluation ===")
    transfer_results = load_json(args.results)

    whitebox_results = None
    if os.path.exists(args.whitebox_results):
        whitebox_results = load_json(args.whitebox_results)
        print(f"  White-box results: {args.whitebox_results}")
    else:
        print(f"  [!] White-box results not found at {args.whitebox_results} "
              f"— skipping comparison plot.")

    epsilons   = transfer_results["epsilons"]
    surrogate  = transfer_results.get("surrogate", "?")
    has_cross  = bool(transfer_results.get("cross_arch_matrix"))

    print(f"  Surrogate:    {surrogate}")
    print(f"  Epsilons:     {epsilons}")
    print(f"  Cross-arch:   {has_cross}")

    print("\n--- [1/3] Plotting transfer ASR heatmap ---")
    plot_transfer_heatmap(transfer_results, args.plots_dir)

    print("\n--- [2/3] Plotting white-box vs transfer comparison ---")
    if whitebox_results:
        plot_transfer_vs_whitebox(transfer_results, whitebox_results, args.plots_dir)
    else:
        print("  Skipped (no white-box results).")

    print("\n--- [3/3] Plotting cross-arch pair ranking ---")
    plot_arch_pair_ranking(transfer_results, args.plots_dir)

    print_transfer_analysis(transfer_results, whitebox_results)

    print("=" * 65)
    print("Done. Outputs:")
    print(f"  {args.plots_dir}/transfer_heatmap.png")
    print(f"  {args.plots_dir}/transfer_vs_whitebox.png")
    print(f"  {args.plots_dir}/transfer_arch_pairs.png")
    print("=" * 65)


if __name__ == "__main__":
    main()