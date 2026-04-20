"""
dgi_functions.py
================
All logic for notebook P3 · Drug–Gene Interaction Collection.

Functions
---------
load_dgi_inputs     — load hub genes + survival targets
collect_interactions — query selected databases + curated fallback
build_dgi_dataframe — clean, deduplicate, and compute composite score
build_gnn_edge_list — add GNN feature columns and export
plot_dgi_dashboard  — 5-panel summary figure
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

# Colours per source (used in dashboard)
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


# ─────────────────────────────────────────────────────────────────────────────
def load_dgi_inputs(tables_dir):
    """
    Load hub gene list and survival target set from previous steps.

    Parameters
    ----------
    tables_dir : Path

    Returns
    -------
    gene_list : list
        Hub gene symbols (input to database queries).
    hub_score_map : dict
        gene → hub_score (used in composite scoring).
    surv_genes : set
        Genes with significant survival association (get score bonus).
    """
    hub_df    = pd.read_csv(tables_dir / "hub_genes.csv")
    gene_list = hub_df.gene.dropna().unique().tolist()
    hub_score_map = (hub_df.set_index("gene")["hub_score"].to_dict()
                     if "hub_score" in hub_df.columns else {})

    surv_file = tables_dir / "survival_filtered_genes.csv"
    surv_genes = (set(pd.read_csv(surv_file)["gene"].dropna())
                  if surv_file.exists() else set())

    print(f"Hub genes        : {len(gene_list)}")
    print(f"Survival targets : {len(surv_genes)}")
    return gene_list, hub_score_map, surv_genes


# ─────────────────────────────────────────────────────────────────────────────
def collect_interactions(gene_list,
                         use_dgidb=True,
                         use_chembl=True,
                         use_opentargets=True,
                         use_curated=True):
    """
    Query selected databases and merge results.

    Each flag independently enables or disables a data source.
    The curated fallback automatically fills genes not covered by live APIs.

    Parameters
    ----------
    gene_list : list
        Gene symbols to query.
    use_dgidb, use_chembl, use_opentargets : bool
        Enable/disable each live API.
    use_curated : bool
        Enable/disable the built-in curated fallback dataset.

    Returns
    -------
    all_edges : list of dict
        Combined raw interaction records from all enabled sources.
    apis_ok : list of str
        Names of sources that returned at least one result.
    """
    # Import here so the function works in Colab without the full repo
    import sys
    from pathlib import Path

    # Try to find and add scripts/ to path if not already there
    for candidate in [Path(__file__).resolve().parent,
                      Path.cwd() / "scripts",
                      Path.cwd()]:
        if (candidate / "utils" / "__init__.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break

    from utils.api_clients import (query_dgidb, query_chembl,
                                   query_opentargets, get_curated_fallback)

    all_edges, apis_ok = [], []

    if use_dgidb:
        print("Querying DGIdb...")
        edges = query_dgidb(gene_list)
        if edges:
            all_edges.extend(edges)
            apis_ok.append("DGIdb")
            print(f"  → {len(edges)} interactions")
        else:
            print("  → No results (API unreachable or no matches)")

    if use_chembl:
        print("Querying ChEMBL...")
        edges = query_chembl(gene_list)
        if edges:
            all_edges.extend(edges)
            apis_ok.append("ChEMBL")
            print(f"  → {len(edges)} interactions")
        else:
            print("  → No results (API unreachable or no matches)")

    if use_opentargets:
        print("Querying OpenTargets...")
        edges = query_opentargets(gene_list)
        if edges:
            all_edges.extend(edges)
            apis_ok.append("OpenTargets")
            print(f"  → {len(edges)} interactions")
        else:
            print("  → No results (API unreachable or no matches)")

    if use_curated:
        if not all_edges:
            print("No live API results — loading full curated fallback...")
            all_edges = get_curated_fallback(gene_list)
            print(f"  → {len(all_edges)} curated interactions")
        else:
            covered = {e["gene"] for e in all_edges}
            missing = [g for g in gene_list if g not in covered]
            if missing:
                curated = get_curated_fallback(missing)
                all_edges.extend(curated)
                print(f"Curated: {len(curated)} interactions added "
                      f"for {len(missing)} uncovered genes")
    elif not all_edges:
        print("\nWarning: no databases selected and curated fallback is disabled.")

    print(f"\nSources used    : {apis_ok or ['curated fallback']}")
    print(f"Raw interactions: {len(all_edges)}")
    return all_edges, apis_ok


# ─────────────────────────────────────────────────────────────────────────────

DRUG_FEAT_COLS = [
    "approved", "immunotherapy", "anti_neoplastic", "clinical_phase",
    "interaction_score", "n_publications", "source_DGIdb", "source_ChEMBL",
    "source_OpenTargets", "type_inhibitor", "type_agonist", "type_antagonist",
    "type_antibody", "type_binder", "type_activator",
]


def build_dgi_dataframe(all_edges, hub_score_map, surv_genes, W):
    """
    Convert raw interaction records into a clean, scored dataframe.

    Steps:
      1. Ensure all feature columns exist (fill missing with 0 / False)
      2. Standardise types (bool, int, float)
      3. Deduplicate gene-drug pairs (keep highest interaction_score)
      4. Compute composite score as weighted sum

    Parameters
    ----------
    all_edges : list of dict
        Raw records from collect_interactions().
    hub_score_map : dict
        gene → hub_score.
    surv_genes : set
        Genes with survival association (get +0.10 bonus).
    W : dict
        Scoring weights: interaction, publications, phase, approved, hub.

    Returns
    -------
    dgi_df : pd.DataFrame
        Clean edge dataframe sorted by composite_score descending.
    """
    dgi_df = pd.DataFrame(all_edges)

    for col in DRUG_FEAT_COLS:
        if col not in dgi_df.columns:
            dgi_df[col] = 0
    for col in ["approved", "immunotherapy", "anti_neoplastic"]:
        dgi_df[col] = dgi_df[col].fillna(False).astype(bool)
    dgi_df["interaction_score"] = pd.to_numeric(
        dgi_df.interaction_score, errors="coerce").fillna(0)
    dgi_df["n_publications"] = pd.to_numeric(
        dgi_df.n_publications, errors="coerce").fillna(0).astype(int)
    dgi_df["clinical_phase"] = pd.to_numeric(
        dgi_df.clinical_phase, errors="coerce").fillna(0).astype(int)
    dgi_df["drug"] = dgi_df.drug.str.strip().str.title()
    dgi_df["gene"] = dgi_df.gene.str.strip().str.upper()

    dgi_df = (dgi_df
              .sort_values("interaction_score", ascending=False)
              .drop_duplicates(["gene", "drug"], keep="first")
              .reset_index(drop=True))

    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    dgi_df["composite_score"] = (
        W["interaction"]  * norm(dgi_df.interaction_score) +
        W["publications"] * norm(dgi_df.n_publications.clip(0, 30)) +
        W["phase"]        * (dgi_df.clinical_phase / 4) +
        W["approved"]     * dgi_df.approved.astype(float) +
        W["hub"]          * norm(dgi_df.gene.map(hub_score_map).fillna(0)) +
        dgi_df.gene.isin(surv_genes).astype(float) * 0.10
    ).clip(0, 1).round(4)

    dgi_df = dgi_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    print(f"Edges (deduplicated): {len(dgi_df)}")
    print(f"Unique genes        : {dgi_df.gene.nunique()}")
    print(f"Unique drugs        : {dgi_df.drug.nunique()}")
    print(f"Approved drugs      : {dgi_df.approved.sum()}")
    return dgi_df


# ─────────────────────────────────────────────────────────────────────────────
def build_gnn_edge_list(dgi_df, hub_score_map, surv_genes, tables_dir):
    """
    Add one-hot GNN feature columns and export dgi_edges_gnn.csv.

    Adds:
      - hub_score, survival_target
      - source_DGIdb, source_ChEMBL, source_OpenTargets  (one-hot)
      - type_inhibitor, type_agonist, … (one-hot by interaction_type)

    Parameters
    ----------
    dgi_df : pd.DataFrame
        Output of build_dgi_dataframe().
    hub_score_map : dict
    surv_genes : set
    tables_dir : Path

    Returns
    -------
    gnn_df : pd.DataFrame
        GNN-ready edge dataframe (also saved to disk).
    """
    gnn_df = dgi_df.copy()
    gnn_df["hub_score"]       = gnn_df.gene.map(hub_score_map).fillna(0)
    gnn_df["survival_target"] = gnn_df.gene.isin(surv_genes).astype(int)

    for src in ["DGIdb", "ChEMBL", "OpenTargets"]:
        gnn_df[f"source_{src}"] = (gnn_df.source == src).astype(int)
    for it in ["inhibitor", "agonist", "antagonist", "antibody", "binder", "activator"]:
        gnn_df[f"type_{it}"] = (gnn_df.interaction_type.str.lower() == it).astype(int)

    gnn_cols = (
        ["gene", "drug", "composite_score", "approved", "immunotherapy",
         "anti_neoplastic", "clinical_phase", "interaction_score", "n_publications"] +
        [f"source_{s}" for s in ["DGIdb", "ChEMBL", "OpenTargets"]] +
        [f"type_{t}" for t in ["inhibitor", "agonist", "antagonist",
                                "antibody", "binder", "activator"]] +
        ["hub_score", "survival_target", "interaction_type", "directionality", "source"]
    )
    gnn_df[[c for c in gnn_cols if c in gnn_df.columns]].to_csv(
        tables_dir / "dgi_edges_gnn.csv", index=False)

    print(f"Saved: dgi_edges_gnn.csv")
    print(f"  Edges  : {len(gnn_df)}")
    print(f"  Genes  : {gnn_df.gene.nunique()}")
    print(f"  Drugs  : {gnn_df.drug.nunique()}")
    print(f"  → Ready for notebook P4 (GNN)")
    return gnn_df


# ─────────────────────────────────────────────────────────────────────────────
_TYPE_COLS = [
    "#534AB7","#1D9E75","#D85A30","#BA7517","#888780","#B5D4F4",
    "#E07B54","#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B",
]
 
 
def _as_path(p):
    return Path(p)
def _save_panel(ax, figures_dir: Path, filename: str, dpi: int = 200):
    """Save a single axes as an individual PNG via tight bbox extraction."""
    fig      = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    extent   = ax.get_tightbbox(renderer)
    if extent is None:
        return
    bbox_in = extent.transformed(fig.dpi_scale_trans.inverted())
    # Use a very small pad (1.02) to avoid picking up content from
    # adjacent axes while still capturing axis labels and titles
    fig.savefig(figures_dir / filename, dpi=dpi,
                bbox_inches=bbox_in.expanded(1.02, 1.02))
 
 
def plot_dgi_dashboard(dgi_df: pd.DataFrame, figures_dir,
                       top_genes: int = 30,
                       top_heatmap_drugs: int = 20,
                       max_heatmap_genes: int = 12):
    """
    5-panel summary dashboard.  Saves combined + 5 individual panels.
 
    clinical_phase 0=Preclinical, 1=Phase1, 2=Phase2, 3=Phase3, 4=Approved.
    """
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
    # Extra height for bottom row to give room for Panel B legend below donut
    fig = plt.figure(figsize=(18, panel_a_height + 7.0), facecolor="white")
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[panel_a_height, 7.0],
        hspace=0.65, wspace=0.45,
    )
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel A — interactions per gene
    # ══════════════════════════════════════════════════════════════════════════
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
    ax1.set_title(
        f"A  Interactions per gene  (top {top_genes} of {gc_full.shape[0]})",
        fontsize=11, fontweight="bold", loc="left")
    ax1.tick_params(axis="y", labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax1.margins(y=0.01)
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel B — interaction type donut
    # FIX: legend placed BELOW the donut (not beside it touching Panel A)
    # ══════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 2])
 
    tc_raw  = dgi_df["interaction_type"].str.lower().fillna("unknown").value_counts()
    pct_raw = tc_raw / tc_raw.sum() * 100
    keep    = pct_raw >= 3.0
    tc_kept = tc_raw[keep].copy()
    other   = tc_raw[~keep].sum()
    if other > 0:
        tc_kept["other"] = other
    n_t   = len(tc_kept)
    tcols = _TYPE_COLS[:n_t]
    pct_k = tc_kept / tc_kept.sum() * 100
 
    ax2.pie(
        tc_kept.values,
        labels=None,                    # all labels go in the legend below
        colors=tcols,
        autopct=lambda p: f"{p:.0f}%" if p >= 6.0 else "",
        pctdistance=0.72,
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 8, "fontweight": "bold"},
    )
 
    # Legend BELOW the donut — bbox_to_anchor y < 0 puts it under the axes
    legend_handles = [
        mpatches.Patch(color=tcols[i],
                       label=f"{lbl}  ({pct_k.iloc[i]:.0f}%)")
        for i, lbl in enumerate(tc_kept.index)
    ]
    ax2.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.05),   # centred, below the axes
        ncol=2,                          # two columns so it doesn't get too tall
        fontsize=8, framealpha=0.9,
        title="Interaction type", title_fontsize=8.5,
        borderpad=0.8, handlelength=1.2,
    )
    ax2.set_title("B  Interaction types",
                  fontsize=11, fontweight="bold", loc="left")
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel C — stacked 100% horizontal bars
    # FIX: legend placed ABOVE the bars (outside the data area)
    # ══════════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 0])
 
    appr = (dgi_df.groupby(["source", "approved"]).size().unstack(fill_value=0)
            .rename(columns={True: "Approved", False: "Not approved",
                              1:   "Approved", 0:    "Not approved"}))
    for col in ["Approved", "Not approved"]:
        if col not in appr.columns:
            appr[col] = 0
    appr     = appr[appr.sum(axis=1) > 0].copy()
    tot_src  = appr.sum(axis=1)
    pct_a    = appr["Approved"]     / tot_src * 100
    pct_na   = appr["Not approved"] / tot_src * 100
    y_pos    = np.arange(len(appr))
    bh       = 0.45
 
    bars_a  = ax3.barh(y_pos, pct_a.values, height=bh,
                       color="#1D9E75", alpha=0.88, label="Approved")
    bars_na = ax3.barh(y_pos, pct_na.values, height=bh, left=pct_a.values,
                       color="#D3D1C7", alpha=0.88, label="Not approved")
 
    # Segment labels
    for i in range(len(appr)):
        pa, pn = pct_a.iloc[i], pct_na.iloc[i]
        if pa > 6:
            ax3.text(pa / 2, i, f"{pa:.0f}%",
                     ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
        if pn > 6:
            ax3.text(pa + pn / 2, i, f"{pn:.0f}%",
                     ha="center", va="center", fontsize=8,
                     color="#444", fontweight="bold")
        # Total count to the right of the 100% mark
        ax3.text(103, i, f"n={int(tot_src.iloc[i]):,}",
                 va="center", fontsize=7.5, color="#333")
 
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(appr.index, fontsize=9)
    ax3.set_xlabel("Percentage (%)", fontsize=9)
    ax3.set_xlim(0, 120)
    ax3.axvline(100, color="#ccc", lw=0.7, ls="--")
    ax3.set_title("C  Approval by source",
                  fontsize=11, fontweight="bold", loc="left")
    ax3.spines[["top", "right"]].set_visible(False)
 
    # Legend ABOVE the bars — completely outside the data area
    ax3.legend(
        fontsize=8, framealpha=0.90,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),     # above the axes
        ncol=2, borderpad=0.7,
    )
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel D — clinical phase (log scale)
    # FIX: phase 4 labelled as "Approved\n(phase 4)" so encoding is clear
    # ══════════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, 1])
 
    pm = {
        0: "Preclinical\n(phase 0)",
        1: "Phase 1",
        2: "Phase 2",
        3: "Phase 3",
        4: "Approved\n(phase 4)",
    }
    po = list(pm.values())
    pv = [dgi_df["clinical_phase"].map(pm).value_counts().get(p, 0) for p in po]
    bars = ax4.bar(po, pv, color=[PHASE_COL[k] for k in range(5)],
                   alpha=0.88, edgecolor="white", zorder=3)
 
    ax4.set_yscale("symlog", linthresh=10)
    ax4.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else "0"))
 
    fig.canvas.draw()
    y_top = ax4.get_ylim()[1]
    for b, v in zip(bars, pv):
        if v > 0:
            y_ann = min(v * 1.12, y_top * 0.88)
            ax4.text(b.get_x() + b.get_width() / 2, y_ann, f"{v:,}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
 
    ax4.axvline(0.5, color="#aaa", lw=1.0, ls="--", zorder=2)
    ax4.set_ylabel("Count (log scale)", fontsize=9)
    ax4.set_title("D  Clinical phase",
                  fontsize=11, fontweight="bold", loc="left")
    # Two-line tick labels need slightly more space — reduce font
    ax4.tick_params(axis="x", labelsize=7.5, rotation=0)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(axis="y", ls=":", alpha=0.4, zorder=0)
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel E — score heatmap
    # ══════════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[1, 2])
 
    d_gc  = dgi_df.groupby("drug")["gene"].nunique()
    multi = d_gc[d_gc >= 2].index
    top_m = (dgi_df[dgi_df["drug"].isin(multi)].drop_duplicates("drug")
             .nlargest(top_heatmap_drugs, "composite_score")["drug"].tolist())
    if len(top_m) < top_heatmap_drugs:
        rem = (dgi_df[~dgi_df["drug"].isin(top_m)].drop_duplicates("drug")
               .nlargest(top_heatmap_drugs - len(top_m), "composite_score")
               ["drug"].tolist())
        sel = top_m + rem
    else:
        sel = top_m
 
    hdf = (dgi_df[dgi_df["drug"].isin(sel)]
           .pivot_table(index="drug", columns="gene",
                        values="composite_score", aggfunc="max", fill_value=0))
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
