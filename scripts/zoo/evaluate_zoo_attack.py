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
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_results(path):
    with open(path) as f:
        return json.load(f)


def build_rows(results):
    ARCH_ORDER = ["gatv2", "sage", "sage_edge", "gin"]
    rows = []
    for conv_type, mdata in results["models"].items():
        for eps_str, stats in mdata["epsilon_results"].items():
            row = {
                "conv_type":  conv_type,
                "model":      ARCH_NAMES.get(conv_type, conv_type),
                "epsilon":    float(eps_str),
                "arch_order": ARCH_ORDER.index(conv_type) if conv_type in ARCH_ORDER else 99,
            }
            row.update(stats)
            row["f1_drop"]     = (row.get("post_f1_macro",           float("nan"))
                                  - row.get("clean_f1_macro",        float("nan")))
            row["recall_drop"] = (row.get("post_recall_suspicious",  float("nan"))
                                  - row.get("clean_recall_suspicious", float("nan")))
            rows.append(row)
    rows.sort(key=lambda r: (r["arch_order"], r["epsilon"]))
    return rows


def models_in_order(rows):
    seen, out = set(), []
    for r in rows:
        if r["conv_type"] not in seen:
            seen.add(r["conv_type"])
            out.append(r["conv_type"])
    return out


def plot_asr_curve(rows, outdir):
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ct in conv_types:
        model_rows = sorted([r for r in rows if r["conv_type"] == ct],
                            key=lambda r: r["epsilon"])
        if not model_rows:
            continue
        eps_vals = [r["epsilon"] for r in model_rows]
        asr_vals = [r.get("asr", float("nan")) * 100 for r in model_rows]

        ax.plot(
            eps_vals, asr_vals,
            marker=MODEL_MARKERS.get(ct, "o"),
            linewidth=2, markersize=8,
            color=MODEL_COLORS.get(ct),
            label=ARCH_NAMES.get(ct, ct),
        )
        for x, y in zip(eps_vals, asr_vals):
            if not np.isnan(y):
                ax.annotate(f"{y:.1f}%", (x, y),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=MODEL_COLORS.get(ct))

    ax.set_xlabel("Perturbation Budget ε  (L∞)", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("ZOO Feature Attack — ASR vs ε\n(full query budget; evasion of suspicious subgraph detection)",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(bottom=0, top=min(105, ax.get_ylim()[1] * 1.15))
    if len(epsilons) > 1:
        ax.set_xlim(left=0)
    ax.legend(title="Model", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.35)

    outpath = os.path.join(outdir, "zoo_asr_curve.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_query_efficiency(rows, results, outdir):
    query_budgets = results.get("query_budgets", [])
    if not query_budgets:
        print("  [!] No query_budgets found — skipping query efficiency plot.")
        return

    epsilons   = sorted(set(r["epsilon"] for r in rows))
    conv_types = models_in_order(rows)
    n_eps      = len(epsilons)

    fig, axes = plt.subplots(
        1, n_eps,
        figsize=(6 * n_eps, 4.5),
        sharey=True,
    )
    if n_eps == 1:
        axes = [axes]

    for ax, eps in zip(axes, epsilons):
        for ct in conv_types:
            match = [r for r in rows if r["conv_type"] == ct and r["epsilon"] == eps]
            if not match:
                continue
            aqb = match[0].get("asr_at_queries", {})
            budgets = [b for b in query_budgets if str(b) in aqb]
            asr_vals = [aqb[str(b)] * 100 for b in budgets]
            if not budgets:
                continue

            ax.plot(
                budgets, asr_vals,
                marker=MODEL_MARKERS.get(ct, "o"),
                linewidth=2, markersize=6,
                color=MODEL_COLORS.get(ct),
                label=ARCH_NAMES.get(ct, ct),
            )

        ax.set_xlabel("Query Budget", fontsize=11)
        ax.set_ylabel("ASR (%)", fontsize=11)
        ax.set_title(f"ε = {eps:.3f}", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(bottom=0, top=105)
        ax.legend(title="Model", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.35)

    fig.suptitle("ZOO Attack — Query Efficiency  (ASR vs Query Budget)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    outpath = os.path.join(outdir, "zoo_query_efficiency.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_query_distribution(rows, results, outdir):
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))
    mid_eps    = epsilons[len(epsilons) // 2]

    data_per_model = {}
    for ct in conv_types:
        eps_res = results["models"].get(ct, {}).get("epsilon_results", {}).get(str(mid_eps), {})
        pg = eps_res.get("per_graph", {})
        qs = pg.get("queries_used", [])
        ev = pg.get("evaded", [])
        if qs:
            data_per_model[ct] = {"queries": qs, "evaded": ev}

    if not data_per_model:
        print("  [!] No per-graph query data found — skipping query distribution plot.")
        return

    n_models = len(data_per_model)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.2), sharey=False)
    if n_models == 1:
        axes = [axes]

    max_queries = results.get("max_queries", None)

    for ax, (ct, d) in zip(axes, data_per_model.items()):
        queries  = np.array(d["queries"])
        evaded   = np.array(d["evaded"])
        color    = MODEL_COLORS.get(ct, "#999999")

        successful_q = queries[evaded == 1]
        failed_q     = queries[evaded == 0]

        bins = np.linspace(0, max(queries) + 1, 30)
        if len(successful_q):
            ax.hist(successful_q, bins=bins, alpha=0.7, color=color,
                    label=f"Evaded  (n={len(successful_q)})")
        if len(failed_q):
            ax.hist(failed_q, bins=bins, alpha=0.35, color=color, hatch="//",
                    label=f"Failed  (n={len(failed_q)})")

        if max_queries:
            ax.axvline(max_queries, color="red", linestyle="--", linewidth=1.5,
                       label=f"Budget={max_queries}")

        if len(successful_q):
            ax.axvline(np.mean(successful_q), color=color, linestyle=":", linewidth=2,
                       label=f"μ_success={np.mean(successful_q):.0f}")

        ax.set_xlabel("Queries Used", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{ARCH_NAMES.get(ct, ct)}  (ε={mid_eps:.3f})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.85)

    fig.suptitle("ZOO Attack — Query Count Distribution per Subgraph",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    outpath = os.path.join(outdir, "zoo_query_distribution.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_confidence_distribution(rows, results, outdir):
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))

    pg_data = {}
    for ct, mdata in results["models"].items():
        for eps_str, stats in mdata["epsilon_results"].items():
            pg = stats.get("per_graph", {})
            pg_data[(ct, float(eps_str))] = {
                "pre":       pg.get("confidence_pre",  []),
                "post":      pg.get("confidence_post", []),
                "mean_pre":  stats.get("mean_confidence_pre",  None),
                "mean_post": stats.get("mean_confidence_post", None),
            }

    has_per_graph = any(len(v["pre"]) > 1 for v in pg_data.values())

    if has_per_graph:
        _plot_confidence_kde(conv_types, epsilons, pg_data, outdir)
    else:
        _plot_confidence_bars(rows, outdir)


def _plot_confidence_kde(conv_types, epsilons, pg_data, outdir):
    n_models = len(conv_types)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.2), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, ct in zip(axes, conv_types):
        eps_list = sorted(
            [eps for (c, eps), v in pg_data.items() if c == ct and len(v["pre"]) > 1]
        )
        if not eps_list:
            ax.set_visible(False)
            continue
        target_eps = eps_list[len(eps_list) // 2]
        v    = pg_data[(ct, target_eps)]
        pre  = np.array(v["pre"])
        post = np.array(v["post"])

        bins = np.linspace(0, 1, 26)
        ax.hist(pre,  bins=bins, alpha=0.55, color="#2196F3", label="Pre-attack",  density=True)
        ax.hist(post, bins=bins, alpha=0.55, color="#E91E63", label="Post-attack", density=True)

        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(0, 1, 200)
            if len(pre) > 2:
                ax.plot(xs, gaussian_kde(pre)(xs),  color="#1565C0", linewidth=2)
            if len(post) > 2:
                ax.plot(xs, gaussian_kde(post)(xs), color="#B71C1C", linewidth=2)
        except ImportError:
            pass

        ax.axvline(pre.mean(),  color="#1565C0", linestyle="--", linewidth=1.5,
                   label=f"μ_pre={pre.mean():.3f}")
        ax.axvline(post.mean(), color="#B71C1C", linestyle="--", linewidth=1.5,
                   label=f"μ_post={post.mean():.3f}")

        ax.set_xlim(0, 1)
        ax.set_xlabel("P(suspicious)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{ARCH_NAMES.get(ct, ct)}  (ε={target_eps:.3f})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.85)

    fig.suptitle("ZOO Attack — Confidence Score Distribution: Pre vs Post",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    outpath = os.path.join(outdir, "zoo_confidence_distribution.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def _plot_confidence_bars(rows, outdir):
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))

    n_models = len(conv_types)
    n_eps    = len(epsilons)
    bar_w    = 0.35
    group_w  = bar_w * 2 + 0.15
    spacing  = 0.4

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * n_eps * n_models), 4.5))

    for i, ct in enumerate(conv_types):
        model_rows = {r["epsilon"]: r for r in rows if r["conv_type"] == ct}
        color = MODEL_COLORS.get(ct, "#999999")
        label = ARCH_NAMES.get(ct, ct)

        for j, eps in enumerate(epsilons):
            row  = model_rows.get(eps, {})
            pre  = row.get("mean_confidence_pre",  0.0)
            post = row.get("mean_confidence_post", 0.0)
            x_base = j * (n_models * group_w + spacing) + i * group_w

            ax.bar(x_base,         pre,  bar_w, color=color, alpha=0.85,
                   label=label if j == 0 else "_nolegend_")
            ax.bar(x_base + bar_w, post, bar_w, color=color, alpha=0.40, hatch="//",
                   label=f"{label} (post)" if j == 0 else "_nolegend_")

    ticks, tick_labels = [], []
    for j, eps in enumerate(epsilons):
        center = j * (n_models * group_w + spacing) + (n_models * group_w) / 2 - group_w / 2
        ticks.append(center)
        tick_labels.append(f"ε={eps:.3f}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean P(suspicious)", fontsize=11)
    ax.set_title("ZOO Attack — Confidence: Pre (solid) vs Post (hatched)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Model", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.35)

    outpath = os.path.join(outdir, "zoo_confidence_distribution.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def plot_metrics_overview(rows, outdir):
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))
    n_m, n_e   = len(conv_types), len(epsilons)

    METRICS = [
        ("asr",                  "ASR (%)",             lambda r: r.get("asr", float("nan")) * 100,        "RdYlGn_r"),
        ("f1_drop",              "F1-macro drop",        lambda r: r.get("f1_drop", float("nan")),          "RdYlGn"),
        ("recall_drop",          "Recall drop (susp.)",  lambda r: r.get("recall_drop", float("nan")),      "RdYlGn"),
        ("mean_confidence_drop", "Confidence drop",      lambda r: r.get("mean_confidence_drop", float("nan")), "YlOrRd"),
    ]
    n_metrics = len(METRICS)

    fig, axes = plt.subplots(1, n_metrics,
                              figsize=(4.5 * n_metrics, max(3.5, 0.85 * n_m + 2)))

    if n_m >= 2 and n_e >= 2:
        y_labels = [ARCH_NAMES.get(ct, ct) for ct in conv_types]
        x_labels = [f"{e:.3f}" for e in epsilons]

        for ax, (key, title, extractor, cmap) in zip(axes, METRICS):
            matrix = np.full((n_m, n_e), float("nan"))
            for i, ct in enumerate(conv_types):
                for j, eps in enumerate(epsilons):
                    match = [r for r in rows if r["conv_type"] == ct and r["epsilon"] == eps]
                    if match:
                        matrix[i, j] = extractor(match[0])

            sns.heatmap(
                matrix, ax=ax,
                xticklabels=x_labels,
                yticklabels=y_labels,
                annot=True, fmt=".2f",
                cmap=cmap,
                linewidths=0.5,
                cbar_kws={"shrink": 0.75},
            )
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel("ε", fontsize=10)
            ax.set_ylabel("")
            ax.tick_params(axis="y", rotation=0)

    else:
        for ax, (key, title, extractor, _) in zip(axes, METRICS):
            vals   = [extractor(r) for r in rows]
            labels = [f"{r['model']}\nε={r['epsilon']:.3f}" for r in rows]
            colors = [MODEL_COLORS.get(r["conv_type"], "#999999") for r in rows]

            bars = ax.bar(range(len(rows)), vals, color=colors, alpha=0.85, width=0.6)
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.35)

            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.002,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("ZOO Attack — Metrics Overview  (model x ε)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    outpath = os.path.join(outdir, "zoo_metrics_overview.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [✓] Saved: {outpath}")


def print_robustness_analysis(rows):
    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))

    print("\n" + "=" * 65)
    print("  ROBUSTNESS ANALYSIS")
    print("=" * 65)

    model_stats = {}
    for ct in conv_types:
        m_rows = [r for r in rows if r["conv_type"] == ct]
        asrs   = [r.get("asr", float("nan")) for r in m_rows]
        drops  = [r.get("f1_drop", float("nan")) for r in m_rows]
        mqs    = [r.get("mean_queries_used", float("nan")) for r in m_rows]
        model_stats[ct] = {
            "mean_asr":     np.nanmean(asrs),
            "max_asr":      np.nanmax(asrs),
            "mean_f1_drop": np.nanmean(drops),
            "mean_queries": np.nanmean(mqs),
            "rows":         m_rows,
        }

    ranked = sorted(model_stats.items(), key=lambda kv: kv[1]["mean_asr"])

    print(f"\n{'Rank':<5} {'Model':<14} {'Mean ASR':>9} {'Max ASR':>9} "
          f"{'Mean ΔF1':>10} {'Mean Q':>8}")
    print("-" * 58)
    for rank, (ct, st) in enumerate(ranked, 1):
        tag = "  ← most robust" if rank == 1 else ("  ← least robust" if rank == len(ranked) else "")
        print(f"  {rank:<3} {ARCH_NAMES.get(ct, ct):<14} "
              f"{st['mean_asr']:>8.1%} {st['max_asr']:>8.1%} "
              f"{st['mean_f1_drop']:>+9.4f} {st['mean_queries']:>7.0f}{tag}")

    if len(ranked) >= 2:
        most  = ARCH_NAMES.get(ranked[0][0],  ranked[0][0])
        least = ARCH_NAMES.get(ranked[-1][0], ranked[-1][0])
        asr_gap = ranked[-1][1]["mean_asr"] - ranked[0][1]["mean_asr"]
        print(f"\n  → {most} is the most robust (lowest mean ASR).")
        print(f"  → {least} is the most vulnerable (highest mean ASR).")
        print(f"  → ASR gap between best and worst: {asr_gap:.1%}")

    if len(epsilons) > 1:
        print(f"\n  ASR progression across ε (averaged over models):")
        for eps in epsilons:
            e_rows   = [r for r in rows if r["epsilon"] == eps]
            mean_asr = np.nanmean([r.get("asr", float("nan")) for r in e_rows])
            bar      = "█" * int(mean_asr * 40)
            print(f"    ε={eps:.3f}  {mean_asr:>6.1%}  {bar}")


def print_query_analysis(rows, results):
    query_budgets = results.get("query_budgets", [])
    conv_types    = models_in_order(rows)
    epsilons      = sorted(set(r["epsilon"] for r in rows))

    print("\n" + "=" * 65)
    print("  QUERY EFFICIENCY ANALYSIS")
    print("=" * 65)

    print(f"\n  {'Model':<14} {'ε':>6} {'ASR':>7}  "
          f"{'Mean Q':>7}  {'Med Q':>6}  {'Q/success':>9}  {'Med Q/suc':>10}")
    print("  " + "-" * 65)
    for row in rows:
        print(f"  {row['model']:<14} {row['epsilon']:>6.3f} "
              f"{row.get('asr', float('nan')):>6.1%}  "
              f"{row.get('mean_queries_used', float('nan')):>7.0f}  "
              f"{row.get('median_queries_used', float('nan')):>6.0f}  "
              f"{row.get('mean_queries_to_success', float('nan')):>9.0f}  "
              f"{row.get('median_queries_to_success', float('nan')):>10.0f}")

    if query_budgets:
        mid_eps = str(epsilons[len(epsilons) // 2])
        print(f"\n  ASR@budget (averaged over models, ε={mid_eps}):")
        for b in query_budgets:
            vals = []
            for ct in conv_types:
                eps_res = results["models"].get(ct, {}).get("epsilon_results", {}).get(mid_eps, {})
                aqb = eps_res.get("asr_at_queries", {})
                if str(b) in aqb:
                    vals.append(aqb[str(b)])
            if vals:
                mean_asr = np.mean(vals)
                bar      = "█" * int(mean_asr * 40)
                print(f"    @{b:<5}  {mean_asr:>6.1%}  {bar}")


def print_sanity_check(rows):
    print("\n" + "=" * 65)
    print("  SANITY CHECK — PERTURBATION REALISM")
    print("=" * 65)

    print("""
  For an L∞ attack with budget ε on F=43 features, the theoretical
  L2 upper bound per node is ε · √43 ≈ ε · 6.56.
  ZOO estimates gradients via finite differences, so perturbations
  are typically smaller than white-box PGD at the same ε.
""")

    ok = True
    print(f"  {'Model':<14} {'ε':>6} {'Mean L2':>10} {'L2/(ε·√43)':>12} "
          f"{'ConfDrop':>10} {'ASR':>7}  Check")
    print("  " + "-" * 70)

    for row in rows:
        eps = row["epsilon"]
        l2  = row.get("mean_l2_perturbation", float("nan"))
        cd  = row.get("mean_confidence_drop",  float("nan"))
        asr = row.get("asr",                   float("nan"))
        post_recall  = row.get("post_recall_suspicious",  float("nan"))
        clean_recall = row.get("clean_recall_suspicious", float("nan"))

        norm_l2 = l2 / (eps * np.sqrt(43)) if eps > 0 else float("nan")

        flags = []
        if not np.isnan(norm_l2) and norm_l2 < 0.05:
            flags.append("⚠ L2 suspiciously low")
        if not np.isnan(asr) and not np.isnan(cd) and asr > 0.05 and cd < 0.01:
            flags.append("⚠ high ASR but tiny conf drop")
        if not np.isnan(post_recall) and not np.isnan(clean_recall):
            if post_recall > clean_recall + 1e-6:
                flags.append("⚠ recall INCREASED after attack (impossible)")
        status = " ".join(flags) if flags else "✓ OK"

        print(f"  {row['model']:<14} {eps:>6.3f} {l2:>10.4f} {norm_l2:>12.3f} "
              f"{cd:>10.4f} {asr:>6.1%}  {status}")
        if flags:
            ok = False

    print()
    if ok:
        print("  All checks passed — perturbation magnitudes look realistic.\n")
    else:
        print("  Some checks flagged — inspect ⚠ rows above.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise ZOO black-box feature attack results"
    )
    parser.add_argument(
        "--results",   default="data/results/zoo_attack_results.json",
        help="Path to zoo_attack_results.json"
    )
    parser.add_argument(
        "--plots_dir", default="data/results/plots",
        help="Output directory for PNG plots"
    )

    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        print("Run scripts/zoo/run_zoo_attack.py first.")
        sys.exit(1)

    os.makedirs(args.plots_dir, exist_ok=True)
    setup_style()

    print(f"\n=== ZOO Attack Evaluation ===")
    print(f"  Reading: {args.results}")

    results = load_results(args.results)
    rows    = build_rows(results)

    if not rows:
        print("ERROR: No results found in JSON. Check that the attack ran successfully.")
        sys.exit(1)

    conv_types = models_in_order(rows)
    epsilons   = sorted(set(r["epsilon"] for r in rows))
    print(f"  Models:        {[ARCH_NAMES.get(ct, ct) for ct in conv_types]}")
    print(f"  Epsilons:      {epsilons}")
    print(f"  Query budgets: {results.get('query_budgets', [])}")
    print(f"  Total (model x ε) pairs: {len(rows)}")

    print("\n--- [1/4] Plotting ASR vs ε ---")
    plot_asr_curve(rows, args.plots_dir)

    print("\n--- [2/4] Plotting query efficiency curves ---")
    plot_query_efficiency(rows, results, args.plots_dir)

    print("\n--- [3/4] Plotting query count distribution ---")
    plot_query_distribution(rows, results, args.plots_dir)

    print("\n--- [4/4] Plotting metrics overview ---")
    plot_metrics_overview(rows, args.plots_dir)

    print_robustness_analysis(rows)
    print_query_analysis(rows, results)
    print_sanity_check(rows)

    print("=" * 65)
    print("Done. Outputs:")
    print(f"  {args.summary}")
    print(f"  {args.plots_dir}/zoo_asr_curve.png")
    print(f"  {args.plots_dir}/zoo_query_efficiency.png")
    print(f"  {args.plots_dir}/zoo_query_distribution.png")
    print(f"  {args.plots_dir}/zoo_metrics_overview.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
