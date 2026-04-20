"""
plot_dgi_dashboard  — drop-in replacement for dgi_functions.py
===============================================================
Copy this function (and _as_path helper) into dgi_functions.py,
replacing the existing ones.

Fixes in this version
---------------------
B  Donut legend overlapping the chart
   → Rare interaction types (< 3 %) are merged into an "other" bucket.
     The legend is placed OUTSIDE the axes so it never overlaps the donut.

C  100% approval bars make the chart meaningless
   → Redesigned as horizontal stacked-percentage bars per source.
     Both segments always visible; % labels on each segment.
     Total count shown to the right.

D  Annotation text clipped at top of axes
   → Annotation y-position capped at 88% of axis upper limit.

E  X-axis gene labels unreadable (too many, overlapping)
   → Gene columns capped at max_heatmap_genes (default 12).
     Labels rotated 90° so they never overlap.

New: per-panel saving
   → After saving the combined dashboard, each panel is saved individually:
        dgi_panel_A_interactions.png
        dgi_panel_B_interaction_types.png
        dgi_panel_C_approval.png
        dgi_panel_D_clinical_phase.png
        dgi_panel_E_score_heatmap.png
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SRC_COL = {
    "DGIdb"      : "#534AB7",
    "ChEMBL"     : "#1D9E75",
    "OpenTargets": "#D85A30",
    "Curated"    : "#888780",
}
PHASE_COL = {
    0: "#D3D1C7",
    1: "#B5D4F4",
    2: "#378ADD",
    3: "#185FA5",
    4: "#1D9E75",
}
_TYPE_COLS = [
    "#534AB7","#1D9E75","#D85A30","#BA7517","#888780","#B5D4F4",
    "#E07B54","#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B",
]


def _as_path(p):
    return Path(p)


def _save_panel(ax, figures_dir: Path, filename: str, dpi: int = 200):
    """Save a single axes as an individual PNG via tight bbox extraction."""
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    extent = ax.get_tightbbox(renderer)
    if extent is None:
        return
    bbox_inches = extent.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(figures_dir / filename, dpi=dpi,
                bbox_inches=bbox_inches.expanded(1.06, 1.06))


def plot_dgi_dashboard(dgi_df: pd.DataFrame, figures_dir,
                       top_genes: int = 30,
                       top_heatmap_drugs: int = 20,
                       max_heatmap_genes: int = 12):
    figures_dir = _as_path(figures_dir)

    # ── Pre-compute gene counts ───────────────────────────────────────────────
    gc_full = dgi_df.groupby(["gene", "source"]).size().unstack(fill_value=0)
    gc_full["_total"] = gc_full.sum(axis=1)
    gc_full = gc_full.sort_values("_total", ascending=False)
    gc = gc_full.head(top_genes).drop(columns="_total")
    gc = gc.loc[gc.sum(axis=1).sort_values(ascending=True).index]
    n_genes = len(gc)

    # ── Layout ────────────────────────────────────────────────────────────────
    panel_a_height = max(4.5, min(11.0, n_genes * 0.30))
    fig = plt.figure(figsize=(18, panel_a_height + 6.0), facecolor="white")
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[panel_a_height, 6.0],
        hspace=0.60, wspace=0.45,
    )

    # ── Panel A ───────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    bot = np.zeros(n_genes)
    for src in list(SRC_COL):
        if src in gc.columns:
            v = gc[src].values
            ax1.barh(gc.index, v, left=bot, color=SRC_COL[src],
                     label=src, alpha=0.88, height=0.72)
            bot += v
    totals = gc.sum(axis=1)
    for i, (_, tot) in enumerate(totals.items()):
        ax1.text(tot + totals.max() * 0.01, i, f"{int(tot):,}",
                 va="center", fontsize=7.5, color="#333")
    ax1.set_xlabel("Number of drug interactions", fontsize=10)
    ax1.set_title(f"A  Interactions per gene  (top {top_genes} of {gc_full.shape[0]})",
                  fontsize=11, fontweight="bold", loc="left")
    ax1.tick_params(axis="y", labelsize=8)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax1.margins(y=0.01)

    # ── Panel B — donut with external legend ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    tc_raw  = dgi_df["interaction_type"].str.lower().fillna("unknown").value_counts()
    pct_raw = tc_raw / tc_raw.sum() * 100
    keep    = pct_raw >= 3.0
    tc_kept = tc_raw[keep].copy()
    other   = tc_raw[~keep].sum()
    if other > 0:
        tc_kept["other"] = other
    n_t = len(tc_kept)
    tcols = _TYPE_COLS[:n_t]
    pct_k = tc_kept / tc_kept.sum() * 100

    ax2.pie(
        tc_kept.values,
        labels=None,
        colors=tcols,
        autopct=lambda p: f"{p:.0f}%" if p >= 6.0 else "",
        pctdistance=0.75,
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 8, "fontweight": "bold"},
    )
    legend_handles = [
        mpatches.Patch(color=tcols[i], label=f"{lbl}  ({pct_k.iloc[i]:.0f}%)")
        for i, lbl in enumerate(tc_kept.index)
    ]
    ax2.legend(handles=legend_handles,
               loc="center left", bbox_to_anchor=(-0.60, 0.50),
               fontsize=7.5, framealpha=0.9,
               title="Interaction type", title_fontsize=8)
    ax2.set_title("B  Interaction types", fontsize=11, fontweight="bold", loc="left")

    # ── Panel C — stacked 100% horizontal bars ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    appr = (dgi_df.groupby(["source","approved"]).size().unstack(fill_value=0)
            .rename(columns={True:"Approved", False:"Not approved",
                              1:"Approved",   0:"Not approved"}))
    for col in ["Approved","Not approved"]:
        if col not in appr.columns:
            appr[col] = 0
    appr = appr[appr.sum(axis=1) > 0].copy()
    tot_src  = appr.sum(axis=1)
    pct_a    = appr["Approved"]     / tot_src * 100
    pct_na   = appr["Not approved"] / tot_src * 100
    y_pos    = np.arange(len(appr))
    bh       = 0.45
    ax3.barh(y_pos, pct_a.values,  height=bh, color="#1D9E75", alpha=0.88, label="Approved")
    ax3.barh(y_pos, pct_na.values, height=bh, left=pct_a.values,
             color="#D3D1C7", alpha=0.88, label="Not approved")
    for i in range(len(appr)):
        pa, pn = pct_a.iloc[i], pct_na.iloc[i]
        if pa > 6:
            ax3.text(pa/2, i, f"{pa:.0f}%",
                     ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
        if pn > 6:
            ax3.text(pa + pn/2, i, f"{pn:.0f}%",
                     ha="center", va="center", fontsize=8,
                     color="#444", fontweight="bold")
        ax3.text(102, i, f"n={int(tot_src.iloc[i]):,}",
                 va="center", fontsize=7.5, color="#333")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(appr.index, fontsize=9)
    ax3.set_xlabel("Percentage (%)", fontsize=9)
    ax3.set_xlim(0, 118)
    ax3.axvline(100, color="#ccc", lw=0.7, ls="--")
    ax3.set_title("C  Approval by source", fontsize=11, fontweight="bold", loc="left")
    ax3.spines[["top","right"]].set_visible(False)
    ax3.legend(fontsize=8, framealpha=0.85, loc="lower right")

    # ── Panel D — clinical phase (log scale, capped annotations) ─────────────
    ax4 = fig.add_subplot(gs[1, 1])
    pm  = {0:"Preclinical",1:"Phase 1",2:"Phase 2",3:"Phase 3",4:"Approved"}
    po  = list(pm.values())
    pv  = [dgi_df["clinical_phase"].map(pm).value_counts().get(p,0) for p in po]
    bars = ax4.bar(po, pv, color=[PHASE_COL[k] for k in range(5)],
                   alpha=0.88, edgecolor="white", zorder=3)
    ax4.set_yscale("symlog", linthresh=10)
    ax4.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v,_: f"{int(v):,}" if v >= 1 else "0"))
    fig.canvas.draw()
    y_top = ax4.get_ylim()[1]
    for b, v in zip(bars, pv):
        if v > 0:
            y_ann = min(v * 1.12, y_top * 0.88)
            ax4.text(b.get_x() + b.get_width()/2, y_ann, f"{v:,}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.axvline(0.5, color="#aaa", lw=1.0, ls="--", zorder=2)
    ax4.set_ylabel("Count (log scale)", fontsize=9)
    ax4.set_title("D  Clinical phase", fontsize=11, fontweight="bold", loc="left")
    ax4.tick_params(axis="x", labelsize=8, rotation=15)
    ax4.spines[["top","right"]].set_visible(False)
    ax4.grid(axis="y", ls=":", alpha=0.4, zorder=0)

    # ── Panel E — heatmap (capped columns, 90° x labels) ─────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    d_gc  = dgi_df.groupby("drug")["gene"].nunique()
    multi = d_gc[d_gc >= 2].index
    top_m = (dgi_df[dgi_df["drug"].isin(multi)].drop_duplicates("drug")
             .nlargest(top_heatmap_drugs,"composite_score")["drug"].tolist())
    if len(top_m) < top_heatmap_drugs:
        rem = (dgi_df[~dgi_df["drug"].isin(top_m)].drop_duplicates("drug")
               .nlargest(top_heatmap_drugs - len(top_m),"composite_score")
               ["drug"].tolist())
        sel = top_m + rem
    else:
        sel = top_m
    hdf = (dgi_df[dgi_df["drug"].isin(sel)]
           .pivot_table(index="drug",columns="gene",
                        values="composite_score",aggfunc="max",fill_value=0))
    hdf = hdf.loc[:, (hdf > 0).any()]
    hdf = hdf.loc[hdf.max(axis=1).sort_values(ascending=False).index]
    if hdf.shape[1] > max_heatmap_genes:
        col_fill = (hdf > 0).sum().sort_values(ascending=False)
        hdf = hdf[col_fill.head(max_heatmap_genes).index]
    im = ax5.imshow(hdf.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax5.set_xticks(range(len(hdf.columns)))
    ax5.set_xticklabels(hdf.columns, rotation=90, ha="center", fontsize=7.5)
    ax5.set_yticks(range(len(hdf.index)))
    ax5.set_yticklabels(hdf.index, fontsize=7.5)
    plt.colorbar(im, ax=ax5, shrink=0.75, pad=0.02, label="Composite score")
    ax5.set_title("E  Score heatmap — top drugs",
                  fontsize=11, fontweight="bold", loc="left")

    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Drug–Gene Interaction Analysis — HCC Hub Genes\n"
        f"{gc_full.shape[0]} genes · {dgi_df['drug'].nunique():,} unique drugs "
        f"· {int(dgi_df['approved'].sum()):,} approved",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Save combined ─────────────────────────────────────────────────────────
    combined_out = figures_dir / "dgi_summary_dashboard.png"
    fig.savefig(combined_out, dpi=200, bbox_inches="tight")
    print(f"Saved (combined): {combined_out}")

    # ── Save each panel individually ──────────────────────────────────────────
    fig.canvas.draw()
    for ax, fname in [
        (ax1, "dgi_panel_A_interactions.png"),
        (ax2, "dgi_panel_B_interaction_types.png"),
        (ax3, "dgi_panel_C_approval.png"),
        (ax4, "dgi_panel_D_clinical_phase.png"),
        (ax5, "dgi_panel_E_score_heatmap.png"),
    ]:
        _save_panel(ax, figures_dir, fname, dpi=200)
        print(f"Saved (panel)  : {figures_dir / fname}")

    plt.show()
    return fig
