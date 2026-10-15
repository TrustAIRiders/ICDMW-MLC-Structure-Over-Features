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

TECHNIQUE_COLORS = {
    "chain_injection": "#1976D2",
    "star":            "#F57C00",
    "parallel":        "#388E3C",
}
TECHNIQUE_LABELS = {
    "chain_injection": "Chain Injection",
    "star":            "Star (hub)",
    "parallel":        "Parallel",
}
TECHNIQUE_ORDER = ["chain_injection", "star", "parallel"]


def setup_style() -> None:
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
        arch_idx = ARCH_ORDER.index(conv_type) if conv_type in ARCH_ORDER else 99
        for combo_key, stats in mdata.get("combinations", {}).items():
            row = {
                "conv_type":  conv_type,
                "model":      ARCH_NAMES.get(conv_type, conv_type),
                "arch_order": arch_idx,
                "technique":  stats.get("technique",       combo_key.split("_k")[0]),
                "k":          stats.get("k_intermediates", int(combo_key.split("_k")[1])),
            }
            row.update(stats)
            rows.append(row)

    rows.sort(key=lambda r: (
        r["arch_order"],
        TECHNIQUE_ORDER.index(r["technique"]) if r["technique"] in TECHNIQUE_ORDER else 99,
        r["k"],
    ))
    return rows


def models_in_order(rows: list[dict]) -> list[str]:
    seen, out = set(), []
    for r in rows:
        if r["conv_type"] not in seen:
            seen.add(r["conv_type"])
            out.append(r["conv_type"])
    return out


def techniques_in_order(rows: list[dict]) -> list[str]:
    seen, out = set(), []
    for tech in TECHNIQUE_ORDER:
        if any(r["technique"] == tech for r in rows):
            out.append(tech)
    return out


def k_values(rows: list[dict]) -> list[int]:
    return sorted(set(r["k"] for r in rows))


def plot_asr_vs_k(rows: list[dict], outdir: str) -> None:
    conv_types = models_in_order(rows)
    techniques = techniques_in_order(rows)
    ks         = k_values(rows)
    n_models   = len(conv_types)

    ncols = 2
    nrows = (n_models + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.2 * nrows),
                             sharey=False)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, ct in zip(axes_flat, conv_types):
        ct_rows = [r for r in rows if r["conv_type"] == ct]

        for tech in techniques:
            tech_rows = sorted(
                [r for r in ct_rows if r["technique"] == tech],
                key=lambda r: r["k"],
            )
            if not tech_rows:
                continue
            xs   = [r["k"]           for r in tech_rows]
            asrs = [r.get("asr", 0.0) * 100 for r in tech_rows]

            ax.plot(
                xs, asrs,
                color=TECHNIQUE_COLORS.get(tech, "#888"),
                marker="o", linewidth=2, markersize=7,
                label=TECHNIQUE_LABELS.get(tech, tech),
            )
            for x, y in zip(xs, asrs):
                ax.annotate(
                    f"{y:.1f}%", (x, y),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5,
                    color=TECHNIQUE_COLORS.get(tech, "#888"),
                )

        ax.set_title(ARCH_NAMES.get(ct, ct), fontsize=12, fontweight="bold")
        ax.set_xlabel("k (attack budget)", fontsize=10)
        ax.set_ylabel("ASR (%)", fontsize=10)
        ax.set_xticks(ks)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.35)

    for ax in list(axes_flat)[len(conv_types):]:
        ax.set_visible(False)

    fig.suptitle(
        "Topology Rewiring Attack — ASR vs Budget k\n"
        "(per model, per technique)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    outpath = os.path.join(outdir, "topo_asr_vs_k.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_technique_heatmap(rows: list[dict], outdir: str) -> None:
    conv_types = models_in_order(rows)
    techniques = techniques_in_order(rows)
    max_k      = max(k_values(rows))

    def _matrix(metric_key: str, scale: float = 1.0) -> np.ndarray:
        mat = np.full((len(conv_types), len(techniques)), np.nan)
        for i, ct in enumerate(conv_types):
            for j, tech in enumerate(techniques):
                match = [
                    r for r in rows
                    if r["conv_type"] == ct and r["technique"] == tech and r["k"] == max_k
                ]
                if match:
                    val = match[0].get(metric_key, np.nan)
                    mat[i, j] = val * scale if not np.isnan(val) else np.nan
        return mat

    asr_mat  = _matrix("asr",                   scale=100)
    conf_mat = _matrix("mean_confidence_drop",   scale=1)

    y_labels = [ARCH_NAMES.get(ct, ct) for ct in conv_types]
    x_labels = [TECHNIQUE_LABELS.get(t, t)  for t in techniques]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(3.5, 0.9 * len(conv_types) + 2)))

    sns.heatmap(
        asr_mat, ax=ax1,
        xticklabels=x_labels, yticklabels=y_labels,
        annot=True, fmt=".1f",
        cmap="YlOrRd", vmin=0, vmax=100,
        linewidths=0.5, cbar_kws={"label": "ASR (%)", "shrink": 0.75},
    )
    ax1.set_title(f"ASR (%)  —  k={max_k}", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", rotation=20)
    ax1.tick_params(axis="y", rotation=0)

    sns.heatmap(
        conf_mat, ax=ax2,
        xticklabels=x_labels, yticklabels=y_labels,
        annot=True, fmt="+.3f",
        cmap="YlOrRd", vmin=0,
        linewidths=0.5, cbar_kws={"label": "Δconfidence", "shrink": 0.75},
    )
    ax2.set_title(f"Mean Confidence Drop  —  k={max_k}", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=20)
    ax2.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Topology Rewiring — Cross-Architecture x Cross-Technique",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    outpath = os.path.join(outdir, "topo_technique_heatmap.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_disruption_profiles(rows: list[dict], outdir: str) -> None:
    techniques = techniques_in_order(rows)
    ks         = k_values(rows)

    PANELS = [
        ("mean_delta_diameter",   "Mean Δdiameter",           None),
        ("mean_delta_clustering", "Mean Δclustering",         None),
        ("mean_delta_avg_degree", "Mean Δavg degree",         None),
        ("mean_delta_edges",      "Mean Δedges",              None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes_flat = axes.flat

    bar_w  = 0.25
    x      = np.arange(len(ks))

    for ax, (metric_key, panel_title, _) in zip(axes_flat, PANELS):
        for i, tech in enumerate(techniques):
            vals = []
            for k in ks:
                tech_k_rows = [
                    r for r in rows if r["technique"] == tech and r["k"] == k
                ]
                if tech_k_rows:
                    v = np.nanmean([r.get(metric_key, np.nan) for r in tech_k_rows])
                else:
                    v = np.nan
                vals.append(v)

            offset = (i - len(techniques) / 2 + 0.5) * bar_w
            bars = ax.bar(
                x + offset, vals,
                bar_w,
                label=TECHNIQUE_LABELS.get(tech, tech),
                color=TECHNIQUE_COLORS.get(tech, "#888"),
                alpha=0.85, edgecolor="white",
            )
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.02 if v >= 0 else -0.05),
                        f"{v:+.2f}",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=7.5,
                        color=TECHNIQUE_COLORS.get(tech, "#888"),
                    )

        ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in ks], fontsize=9)
        ax.set_xlabel("Attack budget k", fontsize=9)
        ax.legend(fontsize=8.5, framealpha=0.85)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Topology Disruption Profiles per Technique\n"
        "(mean over all 4 target models)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    outpath = os.path.join(outdir, "topo_disruption_profiles.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_topo_scatter(rows: list[dict], outdir: str) -> None:
    techniques = techniques_in_order(rows)
    conv_types = models_in_order(rows)
    ks         = k_values(rows)

    k_min, k_max = min(ks), max(ks)
    SIZE_RANGE   = (60, 200)

    def _marker_size(k: int) -> float:
        if k_min == k_max:
            return 130
        t = (k - k_min) / (k_max - k_min)
        return SIZE_RANGE[0] + t * (SIZE_RANGE[1] - SIZE_RANGE[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = [
        ("mean_delta_nodes",    "Mean Δnodes (added)",  "Structural budget vs effectiveness"),
        ("mean_delta_diameter", "Mean Δdiameter",       "Topological disruption vs effectiveness"),
    ]

    tech_handles  = {}
    model_handles = {}

    for ax, (x_key, x_label, subtitle) in zip(axes, panels):
        for tech in techniques:
            for ct in conv_types:
                for k in ks:
                    match = [
                        r for r in rows
                        if r["technique"] == tech and r["conv_type"] == ct and r["k"] == k
                    ]
                    if not match:
                        continue
                    r     = match[0]
                    x_val = r.get(x_key, 0.0)
                    y_val = r.get("asr",  0.0) * 100

                    color  = TECHNIQUE_COLORS.get(tech,  "#888")
                    marker = MODEL_MARKERS.get(ct, "x")
                    size   = _marker_size(k)

                    ax.scatter(
                        x_val, y_val,
                        c=color, marker=marker, s=size,
                        alpha=0.85, linewidths=0.6, edgecolors="white",
                        zorder=3,
                    )
                    ax.annotate(
                        f"k={k}",
                        (x_val, y_val),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color=color, alpha=0.8,
                    )

                    if tech not in tech_handles:
                        tech_handles[tech] = matplotlib.lines.Line2D(
                            [], [], color=color, marker="o",
                            linestyle="None", markersize=9,
                            label=TECHNIQUE_LABELS.get(tech, tech),
                        )
                    if ct not in model_handles:
                        model_handles[ct] = matplotlib.lines.Line2D(
                            [], [], color="#555", marker=marker,
                            linestyle="None", markersize=9,
                            label=ARCH_NAMES.get(ct, ct),
                        )

        all_x = [r.get(x_key, np.nan) for r in rows]
        all_y = [r.get("asr", np.nan) * 100 for r in rows]
        valid  = [(x, y) for x, y in zip(all_x, all_y)
                  if not np.isnan(x) and not np.isnan(y)]
        if len(valid) >= 3:
            xs_v, ys_v = zip(*valid)
            coef = np.polyfit(xs_v, ys_v, 1)
            xs_fit = np.linspace(min(xs_v), max(xs_v), 100)
            ax.plot(xs_fit, np.polyval(coef, xs_fit),
                    color="#757575", linewidth=1.2, linestyle="--",
                    alpha=0.6, label=f"Trend (slope={coef[0]:+.2f})")
            r2 = np.corrcoef(xs_v, ys_v)[0, 1] ** 2
            ax.text(
                0.03, 0.97, f"R²={r2:.3f}",
                transform=ax.transAxes,
                va="top", fontsize=9, color="#555",
                bbox=dict(fc="white", alpha=0.7, ec="none"),
            )

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("ASR (%)", fontsize=10)
        ax.set_title(subtitle, fontsize=10, style="italic")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.35)

    import matplotlib.patches as mpatches
    spacer = mpatches.Patch(color="none", label="")
    size_note = matplotlib.lines.Line2D(
        [], [], color="#aaa", marker="o", linestyle="None",
        markersize=6, label="Size ∝ k",
    )
    handles = (
        list(tech_handles.values()) + [spacer] +
        list(model_handles.values()) + [spacer, size_note]
    )
    labels = [h.get_label() for h in handles]
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(handles),
        frameon=True, fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle(
        "Topology Change vs Attack Success Rate\n"
        "(colour = technique, shape = architecture, size ∝ k)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    outpath = os.path.join(outdir, "topo_scatter.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def print_robustness_analysis(rows: list[dict]) -> None:
    conv_types = models_in_order(rows)
    techniques = techniques_in_order(rows)
    max_k      = max(k_values(rows))
    ks         = k_values(rows)

    print("\n" + "=" * 68)
    print("  ROBUSTNESS ANALYSIS — TOPOLOGY REWIRING")
    print("=" * 68)

    model_asrs = defaultdict(list)
    for r in rows:
        model_asrs[r["conv_type"]].append(r.get("asr", np.nan))

    ranked = sorted(model_asrs.items(), key=lambda kv: np.nanmean(kv[1]))
    print(f"\n{'Rank':<5} {'Model':<14} {'Mean ASR':>9} {'Max ASR':>9}")
    print("  " + "-" * 40)
    for rank, (ct, asrs) in enumerate(ranked, 1):
        tag = "  ← most robust"  if rank == 1           else \
              "  ← least robust" if rank == len(ranked) else ""
        print(f"  {rank:<3} {ARCH_NAMES.get(ct, ct):<14} "
              f"{np.nanmean(asrs):>8.1%}  {np.nanmax(asrs):>8.1%}{tag}")

    print(f"\n  Best technique at k={max_k}:")
    print(f"  {'Model':<14}  {'Best technique':<14}  {'ASR':>7}  vs worst  {'ΔConf':>8}")
    print("  " + "-" * 58)
    for ct in conv_types:
        tech_rows = {
            tech: next(
                (r for r in rows
                 if r["conv_type"] == ct and r["technique"] == tech and r["k"] == max_k),
                None,
            )
            for tech in techniques
        }
        valid = {t: r for t, r in tech_rows.items() if r is not None}
        if not valid:
            continue
        best  = max(valid.items(), key=lambda kv: kv[1].get("asr", 0))
        worst = min(valid.items(), key=lambda kv: kv[1].get("asr", 0))
        gap   = (best[1].get("asr", 0) - worst[1].get("asr", 0)) * 100
        conf  = best[1].get("mean_confidence_drop", np.nan)
        print(f"  {ARCH_NAMES.get(ct, ct):<14}  "
              f"{TECHNIQUE_LABELS.get(best[0], best[0]):<14}  "
              f"{best[1].get('asr', 0):>6.1%}  "
              f"(+{gap:.1f}pp vs {TECHNIQUE_LABELS.get(worst[0], worst[0])})  "
              f"{conf:>+8.4f}")

    print(f"\n  Budget efficiency  (ASR / mean_delta_nodes, at k={max_k}):")
    print(f"  {'Technique':<14}  {'Avg new nodes':>14}  {'Avg ASR':>8}  {'ASR/node':>10}")
    print("  " + "-" * 52)
    for tech in techniques:
        tech_k_rows = [r for r in rows if r["technique"] == tech and r["k"] == max_k]
        if not tech_k_rows:
            continue
        avg_nodes = np.nanmean([r.get("mean_delta_nodes", np.nan) for r in tech_k_rows])
        avg_asr   = np.nanmean([r.get("asr",              np.nan) for r in tech_k_rows])
        efficiency = avg_asr / avg_nodes if avg_nodes > 0 else np.nan
        print(f"  {TECHNIQUE_LABELS.get(tech, tech):<14}  "
              f"{avg_nodes:>14.1f}  "
              f"{avg_asr:>7.1%}  "
              f"{efficiency:>9.4f}")

    print(f"\n  Topology disruption summary  (mean over all models, k={max_k}):")
    print(f"  {'Technique':<14}  {'Δdiameter':>10}  {'Δclustering':>12}  "
          f"{'Δavg_deg':>10}  {'Δedges':>8}")
    print("  " + "-" * 62)
    for tech in techniques:
        tech_k_rows = [r for r in rows if r["technique"] == tech and r["k"] == max_k]
        if not tech_k_rows:
            continue
        dd = np.nanmean([r.get("mean_delta_diameter",   np.nan) for r in tech_k_rows])
        dc = np.nanmean([r.get("mean_delta_clustering", np.nan) for r in tech_k_rows])
        dg = np.nanmean([r.get("mean_delta_avg_degree", np.nan) for r in tech_k_rows])
        de = np.nanmean([r.get("mean_delta_edges",      np.nan) for r in tech_k_rows])
        print(f"  {TECHNIQUE_LABELS.get(tech, tech):<14}  "
              f"{dd:>+10.2f}  {dc:>+12.4f}  {dg:>+10.4f}  {de:>+8.1f}")

    print(f"\n  ASR scaling with k  (averaged over all models & techniques):")
    for k in ks:
        k_rows   = [r for r in rows if r["k"] == k]
        mean_asr = np.nanmean([r.get("asr", np.nan) for r in k_rows])
        bar      = "█" * int(mean_asr * 50)
        print(f"    k={k:<3}  {mean_asr:>6.1%}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise Topology Rewiring Attack results"
    )
    parser.add_argument(
        "--results",   default="data/results/topology_rewiring_results.json",
        help="Path to topology_rewiring_results.json",
    )
    parser.add_argument(
        "--plots_dir", default="data/results/plots",
        help="Output directory for PNG plots",
    )

    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        print("Run scripts/run_topology_attack.py first.")
        sys.exit(1)

    os.makedirs(args.plots_dir, exist_ok=True)
    setup_style()

    print(f"\n=== Topology Rewiring Attack Evaluation ===")
    print(f"  Reading: {args.results}")

    results = load_results(args.results)
    rows    = build_rows(results)

    if not rows:
        print("ERROR: No combinations found in results JSON.")
        sys.exit(1)

    conv_types = models_in_order(rows)
    techniques = techniques_in_order(rows)
    ks         = k_values(rows)
    print(f"  Models:     {[ARCH_NAMES.get(ct, ct) for ct in conv_types]}")
    print(f"  Techniques: {[TECHNIQUE_LABELS.get(t, t) for t in techniques]}")
    print(f"  k values:   {ks}")
    print(f"  Total combinations: {len(rows)}")

    print("\n--- [1/4] Plotting ASR vs k curves ---")
    plot_asr_vs_k(rows, args.plots_dir)

    print("\n--- [2/4] Plotting technique heatmap ---")
    plot_technique_heatmap(rows, args.plots_dir)

    print("\n--- [3/4] Plotting topology disruption profiles ---")
    plot_disruption_profiles(rows, args.plots_dir)

    print("\n--- [4/4] Plotting topology change vs ASR scatter ---")
    plot_topo_scatter(rows, args.plots_dir)

    print_robustness_analysis(rows)

    print("=" * 68)
    print("Done. Outputs:")
    print(f"  {args.plots_dir}/topo_asr_vs_k.png")
    print(f"  {args.plots_dir}/topo_technique_heatmap.png")
    print(f"  {args.plots_dir}/topo_disruption_profiles.png")
    print(f"  {args.plots_dir}/topo_scatter.png")
    print(f"  {args.summary}")
    print("=" * 68)


if __name__ == "__main__":
    main()
