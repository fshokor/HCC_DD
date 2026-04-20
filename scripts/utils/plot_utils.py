"""
plot_utils.py
=============
Shared plotting helpers used across PPI, survival, DGI, and GNN scripts.
All functions return a (fig, ax) or fig object — nothing is saved internally.
Call plt.savefig() or fig.savefig() in the calling script.

Functions
---------
plot_ppi_network      — spring/kamada-kawai network with coloured nodes
plot_km_grid          — Kaplan-Meier survival grid (n x 4)
plot_cox_forest       — hazard ratio forest plot
plot_dgi_dashboard    — 5-panel drug-gene interaction summary
plot_drug_ranking     — horizontal bar chart of top drug candidates
plot_training_curves  — GNN loss curves per model
plot_model_comparison — R² / MSE / MAE grouped bar chart
plot_scatter          — predicted vs actual scatter with residual lines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx

# ── Colour palette ─────────────────────────────────────────────────────────
COLOR_UP      = "#D85A30"   # upregulated / risk gene
COLOR_DOWN    = "#1D9E75"   # downregulated / protective gene
COLOR_APPROVED = "#1D9E75"
COLOR_NONAPPR  = "#B4B2A9"
MODEL_COLORS  = {"GCN": "#534AB7", "GAT": "#D85A30", "GraphSAGE": "#1D9E75"}
SOURCE_COLORS = {"DGIdb": "#534AB7", "ChEMBL": "#1D9E75",
                 "OpenTargets": "#D85A30", "Curated": "#888780"}
PHASE_COLORS  = {0: "#D3D1C7", 1: "#B5D4F4", 2: "#378ADD",
                 3: "#185FA5", 4: "#1D9E75"}


# ─────────────────────────────────────────────────────────────────────────────
# PPI
# ─────────────────────────────────────────────────────────────────────────────

def plot_ppi_network(
    G: "nx.Graph",
    hub_df: pd.DataFrame,
    top_nodes: int = 15,
    top_labels: int = 15,
    string_score: int = 400,
    log2fc_thresh: float = 1.0,
    padj_thresh: float = 0.05,
) -> tuple:
    """
    Render the PPI network using kamada-kawai layout.

    Returns (fig, ax).
    """
    top_set = set(hub_df.head(top_nodes)["gene"])
    H = G.subgraph(top_set).copy()

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#fafafa")

    try:
        pos = nx.kamada_kawai_layout(H, weight="weight")
    except Exception:
        pos = nx.spring_layout(H, seed=42, k=2.5 / np.sqrt(H.number_of_nodes()))

    reg_map   = {n: G.nodes[n].get("regulation", "up") for n in H.nodes()}
    node_col  = [COLOR_UP if reg_map[n] == "up" else COLOR_DOWN
                 for n in H.nodes()]
    hub_map   = hub_df.set_index("gene")["hub_score"].to_dict()
    node_size = [400 + 3500 * hub_map.get(n, 0) for n in H.nodes()]

    edges_list = list(H.edges(data=True))
    raw_scores = [d.get("weight", 400) for _, _, d in edges_list]
    s_mn, s_mx = min(raw_scores), max(raw_scores)

    def _nw(s):
        return 0.5 + 3.5 * (s - s_mn) / (s_mx - s_mn + 1e-9)

    def _na(s):
        return 0.30 + 0.50 * (s - s_mn) / (s_mx - s_mn + 1e-9)

    for u, v, d in edges_list:
        w = d.get("weight", 400)
        nx.draw_networkx_edges(H, pos, edgelist=[(u, v)], width=_nw(w),
                               alpha=_na(w), edge_color="#555555", ax=ax)

    nx.draw_networkx_nodes(H, pos, node_color=node_col, node_size=node_size,
                           alpha=0.92, linewidths=0.8, edgecolors="white", ax=ax)

    label_set = set(hub_df.head(top_labels)["gene"])
    nx.draw_networkx_labels(H, pos,
                            labels={n: n for n in H.nodes() if n in label_set},
                            font_size=8.5, font_weight="bold",
                            font_color="#111111", ax=ax)

    legend = [
        mpatches.Patch(facecolor=COLOR_UP,   label="Upregulated"),
        mpatches.Patch(facecolor=COLOR_DOWN,  label="Downregulated"),
        Line2D([0], [0], color="#555555", lw=0.8, alpha=0.4, label="Low confidence"),
        Line2D([0], [0], color="#555555", lw=3.5, alpha=0.85, label="High confidence"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=10, framealpha=0.85)
    ax.set_title(
        f"HCC PPI Network — top {top_nodes} hub genes\n"
        f"(STRING ≥ {string_score}  |log2FC| ≥ {log2fc_thresh}  padj < {padj_thresh})\n"
        "Node size = hub score  ·  Edge width = STRING confidence",
        fontsize=12, pad=14,
    )
    ax.axis("off")
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Survival
# ─────────────────────────────────────────────────────────────────────────────

def plot_km_grid(
    top_genes: list,
    surv_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    n_cols: int = 4,
    is_simulated: bool = False,
) -> plt.Figure:
    """
    Kaplan-Meier survival grid. Requires lifelines.
    Returns fig.
    """
    from lifelines import KaplanMeierFitter

    n_rows = int(np.ceil(len(top_genes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.5, n_rows * 4),
                             facecolor="white")
    axes = axes.flatten()

    for idx, gene in enumerate(top_genes):
        ax = axes[idx]
        gd = merged_df[["OS_time", "OS_event", gene]].dropna().copy()
        gd.columns = ["T", "E", "expr"]
        gd["group"] = np.where(gd["expr"] >= gd["expr"].median(), "High", "Low")
        hi, lo = gd[gd.group == "High"], gd[gd.group == "Low"]

        kmf = KaplanMeierFitter()
        kmf.fit(hi["T"], event_observed=hi["E"], label="High")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLOR_UP,
                                   linewidth=2, ci_alpha=0.12)
        kmf.fit(lo["T"], event_observed=lo["E"], label="Low")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLOR_DOWN,
                                   linewidth=2, ci_alpha=0.12)

        row = surv_df[surv_df.gene == gene]
        if len(row):
            km_p  = row["logrank_p"].values[0]
            hr    = row["HR"].values[0]
            ci_l  = row["HR_CI_low"].values[0]
            ci_h  = row["HR_CI_high"].values[0]
            prog  = "↓ protective" if pd.notna(hr) and hr < 1 else "↑ risk"
            p_str = "p < 0.001" if km_p < 0.001 else f"p = {km_p:.3f}"
            ax.set_title(
                f"{gene}  {prog}\n{p_str}   HR = {hr:.2f} [{ci_l:.2f}–{ci_h:.2f}]",
                fontsize=9, pad=4,
                color=COLOR_DOWN if pd.notna(hr) and hr < 1 else COLOR_UP,
            )
        ax.set_xlabel("Days", fontsize=8)
        ax.set_ylabel("Survival probability", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.get_legend().remove()

    for j in range(len(top_genes), len(axes)):
        axes[j].set_visible(False)

    legend_h = [
        Line2D([0], [0], color=COLOR_UP,   lw=2, label="High expression"),
        Line2D([0], [0], color=COLOR_DOWN, lw=2, label="Low expression"),
    ]
    fig.legend(handles=legend_h, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    note = "SIMULATED data" if is_simulated else "TCGA-LIHC"
    fig.suptitle(
        f"Kaplan–Meier — top DEGs  ({note}, n = {len(merged_df)} patients)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    return fig


def plot_cox_forest(
    surv_df: pd.DataFrame,
    top_n: int = 20,
    cox_p_thresh: float = 0.05,
) -> tuple:
    """Cox proportional-hazards forest plot. Returns (fig, ax)."""
    forest = (surv_df.dropna(subset=["HR", "HR_CI_low", "HR_CI_high"])
              .sort_values("cox_p").head(top_n).iloc[::-1])

    fig, ax = plt.subplots(figsize=(10, max(6, len(forest) * 0.42)),
                           facecolor="white")
    y = np.arange(len(forest))

    for i, (_, row) in enumerate(forest.iterrows()):
        col  = COLOR_DOWN if row["HR"] < 1 else COLOR_UP
        sig  = row["cox_p"] < cox_p_thresh
        ax.plot([row["HR_CI_low"], row["HR_CI_high"]], [y[i], y[i]],
                color=col, lw=1.6 if sig else 0.8, alpha=0.9 if sig else 0.45)
        ax.scatter(row["HR"], y[i], color=col, s=70 if sig else 35,
                   marker="D" if sig else "o", zorder=5)
        star = ("***" if row["cox_p"] < 0.001 else "**" if row["cox_p"] < 0.01
                else "*" if row["cox_p"] < 0.05 else "")
        if star:
            ax.text(row["HR_CI_high"] + 0.02, y[i], star,
                    va="center", fontsize=9,
                    color=COLOR_UP if row["HR"] > 1 else COLOR_DOWN)

    ax.axvline(1.0, color="#888", lw=1, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(forest["gene"], fontsize=9)
    ax.set_xlabel("Hazard ratio (HR)  95% CI", fontsize=10)
    ax.set_title("Cox proportional hazards — top DEGs\n"
                 "◆ = significant  Left of 1.0 = protective",
                 fontsize=10, pad=10)
    ax.legend(handles=[mpatches.Patch(facecolor=COLOR_DOWN, label="Protective (HR<1)"),
                       mpatches.Patch(facecolor=COLOR_UP,   label="Risk (HR>1)")],
              fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Drug-gene interaction
# ─────────────────────────────────────────────────────────────────────────────

def plot_drug_ranking(
    ranking_df: pd.DataFrame,
    best_name: str = "GNN",
    top_n: int = 25,
) -> tuple:
    """
    Horizontal bar chart of top drug candidates ranked by GNN score.
    Returns (fig, ax).
    """
    top_df = ranking_df.head(top_n).iloc[::-1].reset_index(drop=True)
    bar_colors = [PHASE_COLORS.get(int(p), "#888780")
                  for p in top_df["clinical_phase"]]

    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.38)), facecolor="white")
    ax.barh(range(len(top_df)), top_df["gnn_score"],
            color=bar_colors, alpha=0.88, height=0.72)

    for i, row in top_df.iterrows():
        marker = "\u2605" if row["approved"] else "\u25CB"
        ax.text(row["gnn_score"] + 0.005, i, marker, va="center", fontsize=11,
                color=COLOR_APPROVED if row["approved"] else "#888780")
        if "original_score" in row:
            ax.scatter(row["original_score"], i, marker="|", s=60,
                       color="#444441", zorder=5, linewidths=1.5)

    labels = [row["drug"] + "   \u2192   " + row["gene"]
              for _, row in top_df.iterrows()]
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(f"Predicted interaction score ({best_name})", fontsize=11)
    ax.set_title(
        f"Top {top_n} drug candidates\n"
        "\u2605 = approved   \u25CB = not approved   | = original score",
        fontsize=11, pad=12,
    )
    legend_p = [mpatches.Patch(color=c,
                               label=f"Phase {k}" if k > 0 else "Preclinical",
                               alpha=0.88)
                for k, c in sorted(PHASE_COLORS.items())]
    ax.legend(handles=legend_p, loc="lower right",
              fontsize=8.5, framealpha=0.85, title="Clinical phase")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1.10)
    ax.axvline(0.5, color="#ccc", lw=0.8, linestyle="--")
    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# GNN
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    all_results: dict,
    best_name: str,
) -> plt.Figure:
    """Loss curves for GCN / GAT / GraphSAGE. Returns fig."""
    model_names = list(all_results.keys())
    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(6 * len(model_names), 5),
                             facecolor="white")
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        h   = all_results[name]["history"]
        ep  = range(1, len(h["train_loss"]) + 1)
        col = MODEL_COLORS.get(name, "#888780")
        m   = all_results[name]["test"]
        ax.plot(ep, h["train_loss"], color=col, lw=2, label="Train")
        ax.plot(ep, h["val_loss"],   color=col, lw=1.5,
                linestyle="--", alpha=0.65, label="Val")
        ax.fill_between(ep, h["train_loss"], h["val_loss"],
                        color=col, alpha=0.06)
        best_ep = int(np.argmin(h["val_loss"])) + 1
        ax.axvline(best_ep, color=col, lw=0.8, linestyle=":", alpha=0.6)
        ax.set_title(
            f"{name}\nTest R²={m['r2']:.4f}  MSE={m['mse']:.4f}",
            fontsize=11,
            color=col if name == best_name else "black",
            fontweight="bold" if name == best_name else "normal",
        )
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8, framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if name == best_name:
            ax.set_facecolor("#f8f6ff")

    fig.suptitle(f"GNN training curves  —  best: {best_name}",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_model_comparison(all_results: dict) -> plt.Figure:
    """R² / MSE / MAE bar chart comparing all models. Returns fig."""
    model_names = list(all_results.keys())
    metric_defs = [("r2",  "R² (↑ better)", True),
                   ("mse", "MSE (↓ better)", False),
                   ("mae", "MAE (↓ better)", False)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor="white")
    for ax, (metric, label, higher) in zip(axes, metric_defs):
        vals   = [all_results[n]["test"][metric] for n in model_names]
        colors = [MODEL_COLORS.get(n, "#888780") for n in model_names]
        best_v = max(vals) if higher else min(vals)
        bars   = ax.bar(model_names, vals, color=colors, width=0.5)
        for bar, val, col in zip(bars, vals, colors):
            bar.set_alpha(1.0 if val == best_v else 0.55)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.015,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Model performance comparison — test set", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_scatter(all_results: dict, best_name: str) -> plt.Figure:
    """Predicted vs actual scatter with residual lines. Returns fig."""
    model_names = list(all_results.keys())
    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(5 * len(model_names), 5),
                             facecolor="white")
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        m   = all_results[name]["test"]
        col = MODEL_COLORS.get(name, "#888780")
        ax.scatter(m["true"], m["pred"], color=col, s=90, alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)
        lo = min(m["true"].min(), m["pred"].min()) - 0.05
        hi = max(m["true"].max(), m["pred"].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.45, zorder=2)
        for t, p in zip(m["true"], m["pred"]):
            ax.plot([t, t], [t, p], color=col, lw=0.6, alpha=0.30, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("True score"); ax.set_ylabel("Predicted score")
        ax.set_title(f"{name}  R²={m['r2']:.4f}", fontsize=10,
                     color=col if name == best_name else "black")
        ax.spines[["top", "right"]].set_visible(False)
        if name == best_name:
            ax.set_facecolor("#f8f6ff")

    fig.suptitle("Predicted vs actual — test set", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig
