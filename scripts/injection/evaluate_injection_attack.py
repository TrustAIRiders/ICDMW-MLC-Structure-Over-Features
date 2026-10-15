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

STRATEGY_COLORS = {
    "random":  "#1565C0",
    "degree":  "#E65100",
    "mimicry": "#2E7D32",
}
STRATEGY_MARKERS = {
    "random":  "o",
    "degree":  "s",
    "mimicry": "^",
}
STRATEGY_LABELS = {
    "random":  "Random",
    "degree":  "Degree-based",
    "mimicry": "Mimicry (centroid)",
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


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_combo(results: dict, conv_type: str, strategy: str, k: int, conn: int) -> dict:
    key = f"{strategy}_k{k}_c{conn}"
    return (results["models"]
                   .get(conv_type, {})
                   .get("combinations", {})
                   .get(key, {}))


def models_present(results: dict) -> list[str]:
    return [ct for ct in ARCH_ORDER if ct in results.get("models", {})]


def plot_asr_vs_k(results: dict, outdir: str) -> None:
    conv_types  = models_present(results)
    k_nodes     = results["k_nodes"]
    strategies  = results["strategies"]
    max_conn    = max(results["connections_per_node"])
    n_models    = len(conv_types)

    ncols = min(2, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 4.5 * nrows),
                             sharey=False)
    axes_flat = np.array(axes).flatten() if n_models > 1 else [axes]

    for ax, ct in zip(axes_flat, conv_types):
        for strategy in strategies:
            asr_vals = []
            for k in k_nodes:
                combo = get_combo(results, ct, strategy, k, max_conn)
                asr_vals.append(combo.get("asr", float("nan")) * 100)

            ax.plot(
                k_nodes, asr_vals,
                marker=STRATEGY_MARKERS.get(strategy, "o"),
                linewidth=2, markersize=8,
                color=STRATEGY_COLORS.get(strategy),
                label=STRATEGY_LABELS.get(strategy, strategy),
            )
            if not np.isnan(asr_vals[-1]):
                ax.annotate(
                    f"{asr_vals[-1]:.1f}%",
                    (k_nodes[-1], asr_vals[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=8.5, color=STRATEGY_COLORS.get(strategy),
                )

        ax.set_title(ARCH_NAMES.get(ct, ct), fontsize=12, fontweight="bold")
        ax.set_xlabel("Injected nodes (k)", fontsize=11)
        ax.set_ylabel("ASR (%)", fontsize=11)
        ax.set_xticks(k_nodes)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(bottom=0, top=min(105, ax.get_ylim()[1] * 1.15))
        ax.legend(title="Strategy", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.35)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Node Injection — ASR vs injected nodes k  [conn={max_conn}]",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    outpath = os.path.join(outdir, "injection_asr_vs_k.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_strategy_heatmap(results: dict, outdir: str) -> None:
    conv_types  = models_present(results)
    strategies  = results["strategies"]
    k_nodes     = results["k_nodes"]
    max_conn    = max(results["connections_per_node"])
    k_mid       = k_nodes[len(k_nodes) // 2]

    row_labels  = [ARCH_NAMES.get(ct, ct) for ct in conv_types]
    col_labels  = [STRATEGY_LABELS.get(s, s) for s in strategies]

    asr_matrix  = np.full((len(conv_types), len(strategies)), float("nan"))
    conf_matrix = np.full_like(asr_matrix, float("nan"))

    for i, ct in enumerate(conv_types):
        for j, strategy in enumerate(strategies):
            combo = get_combo(results, ct, strategy, k_mid, max_conn)
            asr_matrix[i, j]  = combo.get("asr", float("nan")) * 100
            conf_matrix[i, j] = combo.get("mean_confidence_drop", float("nan"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3.5, 0.85 * len(conv_types) + 2)))

    def _heatmap(ax, matrix, title, fmt, cmap, vmin, vmax, cbar_label):
        sns.heatmap(
            matrix, ax=ax,
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=True, fmt=fmt,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            linewidths=0.5,
            cbar_kws={"label": cbar_label, "shrink": 0.8},
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Strategy", fontsize=10)
        ax.tick_params(axis="y", rotation=0)
        ax.tick_params(axis="x", rotation=20)

    _heatmap(ax1, asr_matrix,  f"ASR (%)  [k={k_mid}, conn={max_conn}]",
             ".1f", "YlOrRd", 0, 100, "ASR (%)")
    _heatmap(ax2, conf_matrix, f"Confidence drop  [k={k_mid}, conn={max_conn}]",
             ".3f", "YlOrRd", 0, conf_matrix[~np.isnan(conf_matrix)].max() * 1.05,
             "Mean P(susp) drop")

    fig.suptitle("Node Injection — Strategy Comparison  (model × strategy)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    outpath = os.path.join(outdir, "injection_strategy_heatmap.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_k_scaling(results: dict, outdir: str) -> None:
    conv_types = models_present(results)
    k_nodes    = results["k_nodes"]
    max_conn   = max(results["connections_per_node"])
    strategy   = "random"

    n_models = len(conv_types)
    bar_w    = 0.18
    x        = np.arange(len(k_nodes))

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for i, ct in enumerate(conv_types):
        asr_vals = []
        for k in k_nodes:
            combo = get_combo(results, ct, strategy, k, max_conn)
            asr_vals.append(combo.get("asr", float("nan")) * 100)

        offset = (i - (n_models - 1) / 2) * bar_w
        bars   = ax.bar(
            x + offset, asr_vals,
            width=bar_w,
            color=MODEL_COLORS.get(ct),
            alpha=0.88,
            label=ARCH_NAMES.get(ct, ct),
            edgecolor="white",
        )
        for bar, v in zip(bars, asr_vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.7,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"k = {k}" for k in k_nodes], fontsize=11)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(bottom=0, top=min(105, ax.get_ylim()[1] * 1.14))
    ax.set_title(
        f"Node Injection — ASR Scaling with k  [strategy=random, conn={max_conn}]",
        fontsize=12, fontweight="bold"
    )
    ax.legend(title="Model", framealpha=0.9, fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    outpath = os.path.join(outdir, "injection_k_scaling.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_conn_sensitivity(results: dict, outdir: str) -> None:
    conv_types  = models_present(results)
    conn_values = sorted(results["connections_per_node"])
    k_nodes     = results["k_nodes"]
    k_mid       = k_nodes[len(k_nodes) // 2]
    strategy    = "random"

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for ct in conv_types:
        asr_vals = []
        for conn in conn_values:
            combo = get_combo(results, ct, strategy, k_mid, conn)
            asr_vals.append(combo.get("asr", float("nan")) * 100)

        ax.plot(
            conn_values, asr_vals,
            marker=MODEL_MARKERS.get(ct, "o"),
            linewidth=2, markersize=8,
            color=MODEL_COLORS.get(ct),
            label=ARCH_NAMES.get(ct, ct),
        )
        for x_val, y_val in zip(conn_values, asr_vals):
            if not np.isnan(y_val):
                ax.annotate(
                    f"{y_val:.1f}%", (x_val, y_val),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8,
                    color=MODEL_COLORS.get(ct),
                )

    ax.set_xticks(conn_values)
    ax.set_xlabel("Connections per injected node", fontsize=11)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(bottom=0, top=min(105, ax.get_ylim()[1] * 1.15))
    ax.set_title(
        f"Node Injection — Connectivity Sensitivity  [strategy=random, k={k_mid}]",
        fontsize=12, fontweight="bold"
    )
    ax.legend(title="Model", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    outpath = os.path.join(outdir, "injection_conn_sensitivity.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_robustness_comparison(results: dict, pgd_results: dict, outdir: str) -> None:
    conv_types = models_present(results)
    k_max      = max(results["k_nodes"])
    max_conn   = max(results["connections_per_node"])

    pgd_epsilons = sorted({
        float(e)
        for mdata in pgd_results.get("models", {}).values()
        for e in mdata.get("epsilon_results", {}).keys()
    })
    if not pgd_epsilons:
        print("  [!] No PGD epsilon data — skipping robustness comparison plot.")
        return
    mid_eps = pgd_epsilons[len(pgd_epsilons) // 2]

    inj_asrs = []
    pgd_asrs = []
    labels   = []

    for ct in conv_types:
        inj = get_combo(results, ct, "random", k_max, max_conn)
        pgd = (pgd_results.get("models", {})
                          .get(ct, {})
                          .get("epsilon_results", {})
                          .get(str(mid_eps), {}))
        inj_asrs.append(inj.get("asr", float("nan")) * 100)
        pgd_asrs.append(pgd.get("asr", float("nan")) * 100)
        labels.append(ARCH_NAMES.get(ct, ct))

    x     = np.arange(len(conv_types))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    bars_pgd = ax.bar(x - bar_w / 2, pgd_asrs, bar_w,
                      label=f"White-box PGD  (ε={mid_eps})",
                      color=[MODEL_COLORS.get(ct) for ct in conv_types],
                      alpha=0.9, edgecolor="white")
    bars_inj = ax.bar(x + bar_w / 2, inj_asrs, bar_w,
                      label=f"Node Injection  (random, k={k_max}, conn={max_conn})",
                      color=[MODEL_COLORS.get(ct) for ct in conv_types],
                      alpha=0.42, hatch="//", edgecolor="grey")

    for bar, v in zip(list(bars_pgd) + list(bars_inj),
                      pgd_asrs + inj_asrs):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    if "sage_edge" in conv_types:
        se_idx = conv_types.index("sage_edge")
        pgd_rank = sorted(range(len(conv_types)),
                          key=lambda i: pgd_asrs[i]).index(se_idx) + 1
        inj_rank = sorted(range(len(conv_types)),
                          key=lambda i: inj_asrs[i]).index(se_idx) + 1
        if pgd_rank <= 2 and inj_rank >= len(conv_types) - 1:
            ax.annotate(
                "Rank inversion",
                xy=(x[se_idx] + bar_w / 2, inj_asrs[se_idx]),
                xytext=(x[se_idx] + bar_w / 2 + 0.35, inj_asrs[se_idx] + 5),
                fontsize=9, color="#B71C1C",
                arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.5),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(bottom=0, top=min(105, max(pgd_asrs + inj_asrs) * 1.22))
    ax.set_title("Robustness Comparison — PGD (feature) vs Node Injection (structural)\n"
                 "Robustness under feature attacks does not imply robustness under structural attacks",
                 fontsize=11, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    outpath = os.path.join(outdir, "injection_robustness_comparison.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def print_robustness_analysis(results: dict) -> None:
    conv_types  = models_present(results)
    k_nodes     = results["k_nodes"]
    strategies  = results["strategies"]
    max_conn    = max(results["connections_per_node"])
    k_max       = max(k_nodes)

    print("\n" + "=" * 65)
    print("  NODE INJECTION — ROBUSTNESS ANALYSIS")
    print("=" * 65)

    for strategy in strategies:
        print(f"\n  Strategy: {STRATEGY_LABELS.get(strategy, strategy)}")
        model_means: dict[str, float] = {}
        for ct in conv_types:
            asrs = [
                get_combo(results, ct, strategy, k, max_conn).get("asr", float("nan"))
                for k in k_nodes
            ]
            model_means[ct] = float(np.nanmean(asrs))
        ranked = sorted(model_means.items(), key=lambda kv: kv[1])

        print(f"  {'Rank':<5} {'Model':<14} {'Mean ASR':>9} {'k=1':>7} {'k=5':>7} {'k=10':>7}")
        print("  " + "-" * 50)
        for rank, (ct, mean_asr) in enumerate(ranked, 1):
            tag = "← most robust"    if rank == 1              else \
                  "← most vulnerable" if rank == len(ranked) else ""
            k1  = get_combo(results, ct, strategy, k_nodes[0], max_conn).get("asr", float("nan"))
            k5  = get_combo(results, ct, strategy, 5 if 5 in k_nodes else k_nodes[-1], max_conn).get("asr", float("nan"))
            k10 = get_combo(results, ct, strategy, k_max, max_conn).get("asr", float("nan"))
            print(f"  {rank:<5} {ARCH_NAMES.get(ct, ct):<14} "
                  f"{mean_asr:>8.1%} {k1:>6.1%} {k5:>6.1%} {k10:>6.1%}  {tag}")


def print_strategy_analysis(results: dict) -> None:
    conv_types  = models_present(results)
    k_nodes     = results["k_nodes"]
    strategies  = results["strategies"]
    max_conn    = max(results["connections_per_node"])
    k_mid       = k_nodes[len(k_nodes) // 2]

    print("\n" + "=" * 65)
    print(f"  STRATEGY COMPARISON  [k={k_mid}, conn={max_conn}]")
    print("=" * 65)

    header = f"  {'Model':<14}" + "".join(
        f"  {STRATEGY_LABELS.get(s, s)[:10]:>12}" for s in strategies
    ) + f"  {'Best':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ct in conv_types:
        asrs = {
            s: get_combo(results, ct, s, k_mid, max_conn).get("asr", float("nan"))
            for s in strategies
        }
        best_s = max(asrs, key=lambda s: asrs[s])
        row = f"  {ARCH_NAMES.get(ct, ct):<14}"
        row += "".join(f"  {asrs[s]*100:>11.2f}%" for s in strategies)
        row += f"  {STRATEGY_LABELS.get(best_s, best_s):>12}"
        print(row)

    print()
    print("  Key observations:")

    if "gin" in conv_types and "mimicry" in strategies and "random" in strategies:
        gin_rand   = get_combo(results, "gin", "random",  k_mid, max_conn).get("asr", float("nan"))
        gin_mimicry = get_combo(results, "gin", "mimicry", k_mid, max_conn).get("asr", float("nan"))
        ratio = gin_rand / gin_mimicry if gin_mimicry > 1e-6 else float("inf")
        print(f"\n  GIN mimicry resistance:")
        print(f"    random={gin_rand:.1%}  vs  mimicry={gin_mimicry:.1%}  "
              f"(ratio {ratio:.1f}×)")
        print(f"    → GIN's structural sensitivity (WL-isomorphism) renders feature")
        print(f"      camouflage nearly ineffective — topology dominates over features.")

    if "sage_edge" in conv_types and "gin" in conv_types and "random" in strategies:
        se_asr  = get_combo(results, "sage_edge", "random", max(k_nodes), max_conn).get("asr", float("nan"))
        gin_asr = get_combo(results, "gin",       "random", max(k_nodes), max_conn).get("asr", float("nan"))
        if se_asr > gin_asr:
            print(f"\n  SAGE+Edge rank inversion (k={max(k_nodes)}, random):")
            print(f"    SAGE+Edge={se_asr:.1%}  GIN={gin_asr:.1%}")
            print(f"    → SAGE+Edge aggregates edge features via scatter_mean.")
            print(f"      Injected edges from licit distribution directly corrupt")
            print(f"      that aggregation — structural injection exploits what PGD cannot.")

    if "random" in strategies and "degree" in strategies:
        r_asrs = [
            get_combo(results, ct, "random", max(k_nodes), max_conn).get("asr", float("nan"))
            for ct in conv_types
        ]
        d_asrs = [
            get_combo(results, ct, "degree", max(k_nodes), max_conn).get("asr", float("nan"))
            for ct in conv_types
        ]
        r_mean = float(np.nanmean(r_asrs))
        d_mean = float(np.nanmean(d_asrs))
        if r_mean > d_mean:
            print(f"\n  Random > Degree (k={max(k_nodes)}, avg: {r_mean:.1%} vs {d_mean:.1%}):")
            print(f"    → Elliptic2 subgraphs are small (~3-4 nodes). Degree-based")
            print(f"      connects all k injected nodes to the SAME hubs, creating")
            print(f"      a detectable star pattern. Random yields varied topology.")


def print_scaling_analysis(results: dict) -> None:
    conv_types = models_present(results)
    k_nodes    = results["k_nodes"]
    max_conn   = max(results["connections_per_node"])
    strategy   = "random"

    print("\n" + "=" * 65)
    print(f"  INJECTION SCALING  [strategy=random, conn={max_conn}]")
    print("=" * 65)
    print(f"\n  {'Model':<14}", end="")
    for k in k_nodes:
        print(f"  k={k:<5}", end="")
    print(f"  {'Δ(k1→k10)':>10}")
    print("  " + "-" * 55)

    for ct in conv_types:
        asrs = [
            get_combo(results, ct, strategy, k, max_conn).get("asr", float("nan"))
            for k in k_nodes
        ]
        delta = asrs[-1] - asrs[0] if not (np.isnan(asrs[0]) or np.isnan(asrs[-1])) else float("nan")
        print(f"  {ARCH_NAMES.get(ct, ct):<14}", end="")
        for a in asrs:
            print(f"  {a*100:>5.1f}%", end="")
        print(f"  {delta*100:>+9.1f}pp")

    print()
    print("  Marginal gain per additional injected node:")
    for ct in conv_types:
        asrs = [
            get_combo(results, ct, strategy, k, max_conn).get("asr", float("nan"))
            for k in k_nodes
        ]
        gains = []
        for i in range(1, len(k_nodes)):
            dk = k_nodes[i] - k_nodes[i - 1]
            gains.append((asrs[i] - asrs[i - 1]) / dk * 100)
        gain_str = "  ".join(
            f"k{k_nodes[i-1]}→k{k_nodes[i]}: {gains[i-1]:+.1f}pp/node"
            for i in range(1, len(k_nodes))
        )
        print(f"    {ARCH_NAMES.get(ct, ct):<14}: {gain_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise Node Injection Attack results"
    )
    parser.add_argument("--results",
                        default="data/results/node_injection_results.json",
                        help="Path to node_injection_results.json")
    parser.add_argument("--pgd_results",
                        default="data/results/pgd_attack_results.json",
                        help="Path to pgd_attack_results.json (optional, for robustness comparison)")
    parser.add_argument("--plots_dir",
                        default="data/results/plots",
                        help="Output directory for PNG plots")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        print("Run scripts/run_injection_attack.py first.")
        sys.exit(1)

    os.makedirs(args.plots_dir, exist_ok=True)
    setup_style()

    print(f"\n=== Node Injection Attack Evaluation ===")
    print(f"  Reading: {args.results}")
    results = load_json(args.results)

    pgd_results = None
    if os.path.exists(args.pgd_results):
        pgd_results = load_json(args.pgd_results)
        print(f"  PGD results: {args.pgd_results}")
    else:
        print(f"  [!] PGD results not found at {args.pgd_results} — skipping comparison plot.")

    conv_types  = models_present(results)
    strategies  = results.get("strategies", [])
    k_nodes     = results.get("k_nodes", [])
    conn_values = results.get("connections_per_node", [])
    print(f"  Models:      {[ARCH_NAMES.get(ct, ct) for ct in conv_types]}")
    print(f"  Strategies:  {strategies}")
    print(f"  k_nodes:     {k_nodes}")
    print(f"  conn values: {conn_values}")

    print("\n--- [1/5] Plotting ASR vs k (per model, per strategy) ---")
    plot_asr_vs_k(results, args.plots_dir)

    print("\n--- [2/5] Plotting strategy comparison heatmap ---")
    plot_strategy_heatmap(results, args.plots_dir)

    print("\n--- [3/5] Plotting k-scaling bars (random strategy) ---")
    plot_k_scaling(results, args.plots_dir)

    print("\n--- [4/5] Plotting connectivity sensitivity ---")
    plot_conn_sensitivity(results, args.plots_dir)

    print("\n--- [5/5] Plotting robustness comparison (injection vs PGD) ---")
    if pgd_results:
        plot_robustness_comparison(results, pgd_results, args.plots_dir)
    else:
        print("  Skipped (no PGD results).")

    print_robustness_analysis(results)
    print_strategy_analysis(results)
    print_scaling_analysis(results)

    print("\n" + "=" * 65)
    print("Done. Outputs:")
    print(f"  {args.summary}")
    for name in [
        "injection_asr_vs_k.png",
        "injection_strategy_heatmap.png",
        "injection_k_scaling.png",
        "injection_conn_sensitivity.png",
        "injection_robustness_comparison.png",
    ]:
        print(f"  {args.plots_dir}/{name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
