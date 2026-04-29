"""
report_functions.py
===================
HTML report generators for the three main notebooks.

Functions
---------
generate_scrna_report        — notebook 01: scRNA-seq analysis
generate_target_report       — notebook 02: target prioritisation
generate_gnn_report          — notebook 03: GNN drug ranking

Each function takes the variables already in the notebook namespace,
encodes figures as base64, and writes a self-contained HTML file.

Changelog (fixes applied 2026-04-19)
-------------------------------------
generate_scrna_report:
  [1] Section 1 — raw cell counts now require `n_raw_hcc1` / `n_raw_hcc2`
      passed explicitly; post-filter counts shown separately as "after QC".
  [2] Section 2 — removed the spurious "Rationale" column from the QC table;
      the table is now a clean 2-column (Filter | Value) layout.
  [3] Section 3 — UMAP figure key corrected to match the filename produced by
      scrna_functions.run_umap() → "umap_leiden.png" (was missing entirely).
  [4] Section 4 — added QC violin plot, annotated UMAP, and sample UMAP.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Shared CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 2rem; color: #1a1a2e;
         background: #f8f9fa; }
  h1   { font-size: 1.8rem; border-bottom: 3px solid #4361ee; padding-bottom: .5rem; }
  h2   { font-size: 1.2rem; color: #3a0ca3; margin-top: 2rem; }
  h3   { font-size: 1rem; color: #555; margin: 1rem 0 .4rem; }
  .meta { font-size: .85rem; color: #666; margin-bottom: 1.5rem; }
  .section { background: #fff; border-radius: 10px; padding: 1.5rem;
             margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
  .box  { background: #f4f6ff; border-radius: 8px; padding: 1rem; margin-top: .5rem; }
  .grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
  .stat { background: #fff; border-radius: 8px; padding: 1rem 1.5rem;
          border-left: 4px solid #4361ee; min-width: 130px; }
  .stat .value { font-size: 1.6rem; font-weight: 700; color: #4361ee; }
  .stat .label { font-size: .8rem; color: #555; margin-top: 2px; }
  .good  { background: #d4edda; border-radius: 6px; padding: .7rem; margin: .5rem 0; }
  .warn  { background: #fff3cd; border-radius: 6px; padding: .7rem; margin: .5rem 0; }
  table  { border-collapse: collapse; width: 100%; margin: .5rem 0; }
  th, td { border: 1px solid #dee2e6; padding: .45rem .8rem; font-size: .9rem; text-align: left; }
  th     { background: #e9ecef; font-weight: 600; }
  tr:nth-child(even) td { background: #f8f9fa; }
  .param { background: #e7f3ff; border-radius: 4px; padding: 1px 6px;
           font-family: monospace; font-size: .9rem; }
  img    { max-width: 100%; border-radius: 8px; margin: .5rem 0; }
  .fig-missing { color: #aaa; font-style: italic; padding: .5rem 0; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .summary { background: #eef2ff; border-left: 4px solid #4361ee; border-radius: 8px;
             padding: 1rem 1.25rem; margin-bottom: 1.5rem; font-size: .95rem;
             line-height: 1.6; color: #1a1a2e; }
  .summary strong { color: #3a0ca3; }
  @media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _png_to_b64(path: Path) -> str:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""


def _img_tag(b64: str, caption: str = "", width: str = "100%") -> str:
    if not b64:
        return f'<p class="fig-missing">Figure not found — {caption}</p>'
    cap = f'<p style="font-size:.85em;color:#666;margin:3px 0 14px">{caption}</p>'
    return f'<img src="data:image/png;base64,{b64}" alt="{caption}" style="width:{width}">{cap}'


def _stat(value, label):
    return (f'<div class="stat">'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label}</div>'
            f'</div>')


def _param_row(label, value):
    return f"<tr><td>{label}</td><td><span class='param'>{value}</span></td></tr>"


def _table(df, max_rows=20):
    return df.head(max_rows).to_html(
        index=True, border=0, classes="",
        float_format=lambda x: f"{x:.4f}"
    )


def _write(html: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"  Report saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 01 — scRNA-seq Analysis
# ─────────────────────────────────────────────────────────────────────────────

def generate_scrna_report(
    adata,              # final annotated AnnData (post-QC, post-filter)
    sig,                # DEA significant genes DataFrame
    vote_df,            # cluster → cell type vote table
    score_df,           # marker score DataFrame
    ranked,             # GSEA ranked gene list (Series)
    # FIX [1]: raw counts must be passed explicitly (before any filtering)
    n_raw_hcc1: int,    # cell count in HCC1 BEFORE quality filtering
    n_raw_hcc2: int,    # cell count in HCC2 BEFORE quality filtering
    # configuration
    min_genes, max_genes, max_mt_pct, n_top_genes,
    n_neighbors, n_pcs, resolutions, leiden_col,
    padj_thresh, log2fc_thresh, group,
    # paths
    figures_dir, tables_dir, reports_dir,
):
    """
    Generate an HTML summary report for notebook 01 (scRNA-seq analysis).

    Parameters
    ----------
    n_raw_hcc1, n_raw_hcc2 : int
        Cell counts **before** any QC filtering.  Pass these from the
        notebook immediately after loading the raw data, e.g.::

            n_raw_hcc1 = int((adata_raw.obs["sample"] == "HCC1").sum())
            n_raw_hcc2 = int((adata_raw.obs["sample"] == "HCC2").sum())
    """
    figures_dir = Path(figures_dir)
    tables_dir  = Path(tables_dir)
    reports_dir = Path(reports_dir)

    # ── Collect numbers (all from post-QC adata) ──────────────────────────────
    sample_counts = adata.obs["sample"].value_counts()
    n_qc_hcc1    = int(sample_counts.get("HCC1", sample_counts.get("normal (HCC1)", 0)))
    n_qc_hcc2    = int(sample_counts.get("HCC2", sample_counts.get("tumor (HCC2)",  0)))
    n_total_raw  = n_raw_hcc1 + n_raw_hcc2
    n_total_qc   = adata.n_obs
    n_genes_raw  = adata.n_vars   # genes after HVG step; raw gene # stored in uns if available
    n_clusters   = adata.obs[leiden_col].nunique()
    n_hvg        = (int(adata.var["highly_variable"].sum())
                    if "highly_variable" in adata.var.columns else n_top_genes)
    n_degs  = len(sig)
    n_up    = int((sig["regulation"] == "up").sum())   if "regulation" in sig.columns else "-"
    n_down  = int((sig["regulation"] == "down").sum()) if "regulation" in sig.columns else "-"

    # Cell-type composition
    ct_col     = "manual_celltype" if "manual_celltype" in adata.obs.columns else leiden_col
    ct_counts  = adata.obs[ct_col].value_counts()

    # GSEA tables
    gsea_loaded = {}
    for ont, fname in [("GO-BP","gsea_go_bp.csv"),("GO-MF","gsea_go_mf.csv"),
                       ("KEGG","gsea_kegg.csv")]:
        fpath = tables_dir / fname
        if fpath.exists():
            gsea_loaded[ont] = pd.read_csv(fpath)

    # ── Figures ───────────────────────────────────────────────────────────────
    # FIX [3]: corrected key names to match filenames produced by scrna_functions
    # FIX [4]: added qc_violin, umap_annot, umap_sample
    figs = {
        # QC
        "qc_violin"     : _png_to_b64(figures_dir / "qc_violin.png"),
        # Clustering
        "umap_leiden"   : _png_to_b64(figures_dir / "umap_leiden.png"),
        # Annotation
        "umap_annot"    : _png_to_b64(figures_dir / "umap_annotation.png"),
        "umap_sample"   : _png_to_b64(figures_dir / "umap_samplewise.png"),
        # DEA
        "volcano"       : _png_to_b64(figures_dir / "volcano_plot.png"),
        # GSEA
        "gsea_bp"       : _png_to_b64(figures_dir / "gsea_go_biological_process.png"),
        "gsea_kegg"     : _png_to_b64(figures_dir / "gsea_kegg_pathways.png"),
    }

    # ── Annotation vote table ─────────────────────────────────────────────────
    if "n_cells" in vote_df.columns and "final_label" in vote_df.columns:
        annot_table = vote_df[["final_label","n_cells"]].to_html(
            index=True, border=0, classes=""
        )
    else:
        annot_table = vote_df.to_html(index=True, border=0, classes="")

    # ── Cell-type bar chart figure ─────────────────────────────────────────────
    fig_ct, ax = plt.subplots(figsize=(7, max(3, len(ct_counts) * 0.38)),
                              facecolor="white")
    colors = plt.cm.Set2(np.linspace(0, 1, len(ct_counts)))
    ax.barh(ct_counts.index[::-1], ct_counts.values[::-1],
            color=colors[::-1], edgecolor="white", height=0.7)
    for i, (ct, n) in enumerate(zip(ct_counts.index[::-1], ct_counts.values[::-1])):
        pct = n / n_total_qc * 100
        ax.text(n + n_total_qc * 0.005, i, f"{n:,}  ({pct:.1f}%)",
                va="center", fontsize=8.5)
    ax.set_xlabel("Number of cells")
    ax.set_title("Cell-type composition")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    ct_bar_b64 = _fig_to_b64(fig_ct)
    plt.close(fig_ct)
 
    # ── Top DEG table ─────────────────────────────────────────────────────────
    top_up   = (sig[sig.regulation == "up"]
                .nlargest(10, "log2FC")[["gene","log2FC","adj_pvalue"]])
    top_down = (sig[sig.regulation == "down"]
                .nsmallest(10, "log2FC")[["gene","log2FC","adj_pvalue"]])
 
    def _deg_table(df):
        rows = ""
        for _, r in df.iterrows():
            rows += (f"<tr><td><b>{r.gene}</b></td>"
                     f"<td>{r.log2FC:+.3f}</td>"
                     f"<td>{r.adj_pvalue:.2e}</td></tr>")
        return (f"<table><tr><th>Gene</th><th>log2FC</th>"
                f"<th>adj p-value</th></tr>{rows}</table>")
    # ── GSEA table ────────────────────────────────────────────────────────────
    gsea_rows = ""
    for ont, df in gsea_loaded.items():
        if len(df) > 0 and "Description" in df.columns and "NES" in df.columns:
            for _, r in df.nlargest(5, "NES").iterrows():
                col = "#1D9E75" if r.NES > 0 else "#D85A30"
                gsea_rows += (
                    f"<tr><td><b>{ont}</b></td><td>{r.Description}</td>"
                    f"<td style='color:{col};font-weight:600'>{r.NES:.2f}</td>"
                    f"<td>{r['p.adjust']:.3e}</td></tr>"
                )
    gsea_table = (
        f"<table><tr><th>Ontology</th><th>Pathway</th><th>NES</th><th>adj p</th></tr>"
        f"{gsea_rows}</table>"
        if gsea_rows
        else "<p class='fig-missing'>GSEA tables not found — run the GSEA step first.</p>"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # HTML
    # ─────────────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>scRNA-seq Analysis Report — HCC</title>
  {_CSS}
</head>
<body>

<h1>scRNA-seq Analysis Report</h1>
<div class="meta">
  <b>Dataset:</b> GEO GSE166635 &nbsp;|&nbsp;
  <b>Generated:</b> {_now()} &nbsp;|&nbsp;
  <b>Notebook:</b> 01_scrna_analysis.ipynb
</div>

<!-- ── SUMMARY ──────────────────────────────────────────────────────── -->
<div class="summary">
  <strong>What this report covers:</strong> Single-cell RNA sequencing analysis of
  hepatocellular carcinoma (HCC) using dataset GSE166635, which contains cells from
  tumour-adjacent normal tissue (HCC1) and tumour tissue (HCC2). The pipeline covers
  quality control, dimensionality reduction, unsupervised clustering, multi-source
  cell-type annotation, differential expression analysis (DEA), and gene set
  enrichment analysis (GSEA) — building a high-resolution map of the HCC tumour
  microenvironment and identifying candidate biomarkers and transcriptional drivers
  of disease progression.
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     SECTION 1 · DATASET
     ══════════════════════════════════════════════════════════════════════ -->
<div class="section">
<h2>1 · Dataset</h2>
<div class="box">
  <p>Single-cell RNA sequencing data (10x Genomics Chromium) from two conditions
  obtained from GEO accession <b>GSE166635</b>.</p>
  <h3>Raw data — before quality filtering</h3>
  <div class="grid">
    {_stat(f"{n_raw_hcc1:,}", "raw cells — normal-adjacent (HCC1)")}
    {_stat(f"{n_raw_hcc2:,}", "raw cells — tumor (HCC2)")}
    {_stat(f"{n_total_raw:,}", "total raw cells")}
    {_stat(f"{n_genes_raw:,}", "total raw genes")}
  </div>
</div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     SECTION 2 · PREPROCESSING & QC
     ══════════════════════════════════════════════════════════════════════ -->
<div class="section">
<h2>2 · Preprocessing &amp; Quality Control</h2>
<div class="box">
  <h3>Quality control filters applied</h3>
  <table>
    <tr><th>Filter</th><th>Value</th></tr>
    {_param_row("Minimum genes per cell",   str(min_genes))}
    {_param_row("Maximum genes per cell",   str(max_genes))}
    {_param_row("Max mitochondrial %",      f"{max_mt_pct} %")}
  </table>

  <h3>QC metrics per sample</h3>
  {_img_tag(figs["qc_violin"], "Violin plots — genes detected, total RNA counts, and % mitochondrial reads per sample before filtering")}

  <h3>Cells retained after QC</h3>
  <div class="grid">
    {_stat(f"{n_qc_hcc1:,}", "normal-adjacent (HCC1)")}
    {_stat(f"{n_qc_hcc2:,}", "tumor (HCC2)")}
    {_stat(f"{n_total_qc:,}", "total cells retained")}
    {_stat(f"{100*n_total_qc/n_total_raw:.1f} %" if n_total_raw else "N/A", "cells passing QC")}
  </div>

  <h3>Normalisation &amp; feature selection</h3>
  <table>
    <tr><th>Step</th><th>Value</th></tr>
    {_param_row("Normalisation target",     "10,000 counts / cell")}
    {_param_row("Transformation",           "log1p")}
    {_param_row("Highly variable genes",    f"{n_hvg:,} selected")}
  </table>
</div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     SECTION 3 · CLUSTERING
     ══════════════════════════════════════════════════════════════════════ -->
<div class="section">
<h2>3 · Dimensionality Reduction &amp; Clustering</h2>
<div class="box">
  <table>
    <tr><th>Step</th><th>Value</th></tr>
    {_param_row("PCA components used",       str(n_pcs))}
    {_param_row("kNN neighbours (UMAP)",     str(n_neighbors))}
    {_param_row("Leiden resolutions tested", ", ".join(str(r) for r in resolutions))}
    {_param_row("Resolution selected",       leiden_col.split("_")[-1])}
    {_param_row("Clusters identified",       str(n_clusters))}
  </table>
  <br>
  {_img_tag(figs["umap_leiden"], "UMAP coloured by Leiden cluster (left) and by sample origin — HCC1 normal-adjacent vs. HCC2 tumour (right)")}
</div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     SECTION 4 · CELL-TYPE ANNOTATION
     ══════════════════════════════════════════════════════════════════════ -->
<div class="section">
<h2>4 · Cell-type Annotation</h2>
<div class="box">
  <p>Four evidence sources combined by <b>majority vote</b> per cluster:
  CellTypist, ScType (liver-specific, double-weighted for parenchymal types),
  SingleR (HPCA reference), and a curated marker-score method.</p>

  <h3>UMAP — annotated cell types</h3>
  {_img_tag(figs["umap_annot"], "UMAP coloured by majority-vote cell-type annotation")}

  <h3>Cluster annotation summary</h3>
  {annot_table}
  <h3>Cell-type composition</h3>
  {_img_tag(ct_bar_b64, "Cell count and percentage per annotated cell type")}
</div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     SECTION 5 · DIFFERENTIAL EXPRESSION
     ══════════════════════════════════════════════════════════════════════ -->
<div class="section">
<h2>5 · Differential Expression Analysis</h2>
<div class="box">
  <div class="grid">
    {_stat(str(n_degs), "significant DEGs")}
    {_stat(str(n_up),   "up-regulated in tumour")}
    {_stat(str(n_down), "down-regulated in tumour")}
  </div>
  <h3>Filters applied</h3>
  <table>
    <tr><th>Filter</th><th>Value</th></tr>
    {_param_row("Adjusted p-value threshold", str(padj_thresh))}
    {_param_row("log₂ fold-change threshold", str(log2fc_thresh))}
    {_param_row("Comparison group",           str(group))}
  </table>
  
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
    <div>
      <h3>Top 10 upregulated genes</h3>
      {_deg_table(top_up)}
    </div>
    <div>
      <h3>Top 10 downregulated genes</h3>
      {_deg_table(top_down)}
    </div>
  </div>
</div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     SECTION 6 · PATHWAY ENRICHMENT (GSEA)
     ══════════════════════════════════════════════════════════════════════ -->
<div class="section">
<h2>6 · Gene Set Enrichment Analysis (GSEA)</h2>
<div class="box">
  {gsea_table}
  <br>
  {_img_tag(figs["gsea_bp"], "Top enriched GO Biological Process pathways")}
</div>
</div>

<!-- ── PARAMETERS ───────────────────────────────────────────────────── -->
<div class="section">
<h2>7 · Full Parameter Reference</h2>
<div class="box">
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Step</th></tr>
    <tr><td>min_genes</td><td><span class='param'>{min_genes}</span></td><td>QC filter</td></tr>
    <tr><td>max_genes</td><td><span class='param'>{max_genes}</span></td><td>QC filter</td></tr>
    <tr><td>max_mt_pct</td><td><span class='param'>{max_mt_pct}%</span></td><td>QC filter</td></tr>
    <tr><td>n_top_genes</td><td><span class='param'>{n_top_genes}</span></td><td>HVG selection</td></tr>
    <tr><td>n_pcs</td><td><span class='param'>{n_pcs}</span></td><td>PCA</td></tr>
    <tr><td>n_neighbors</td><td><span class='param'>{n_neighbors}</span></td><td>UMAP / kNN</td></tr>
    <tr><td>leiden_resolution</td><td><span class='param'>{leiden_col.split("_")[-1]}</span></td><td>Clustering</td></tr>
    <tr><td>padj_thresh</td><td><span class='param'>{padj_thresh}</span></td><td>DEA</td></tr>
    <tr><td>log2fc_thresh</td><td><span class='param'>{log2fc_thresh}</span></td><td>DEA</td></tr>
  </table>
</div>
</div>

</body>
</html>"""

    out = reports_dir / "01_scrna_analysis_report.html"
    _write(html, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02 — Target Prioritisation  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def generate_target_report(
    sig, gene_list, G, hub_df, edges_df,
    string_score, log2fc_thresh, padj_thresh,
    surv_df, surv_filtered, is_sim,
    km_p_thresh, cox_p_thresh, hr_min, hr_max,
    dgi_df, apis_ok,
    use_dgidb, use_chembl, use_opentargets, use_curated, W,
    figures_dir, tables_dir, reports_dir,
):
    figures_dir = Path(figures_dir)
    tables_dir  = Path(tables_dir)
    reports_dir = Path(reports_dir)

    figs = {
        "ppi"       : _png_to_b64(figures_dir / "ppi_network.png"),
        "km"        : _png_to_b64(figures_dir / "km_plots.png"),
        "cox"       : _png_to_b64(figures_dir / "cox_forest_plot.png"),
        "dgi_bar"   : _png_to_b64(figures_dir / "dgi_summary_dashboard.png"),
    }

    n_hub    = len(hub_df)
    n_surv   = len(surv_filtered)
    n_dgi    = len(dgi_df)

    surv_note = ("⚠ Simulated survival data used (TCGA-LIHC unavailable)"
                 if is_sim else "✓ TCGA-LIHC survival data")

    # P3 table: drop internal one-hot / GNN feature columns — show only meaningful ones
    _dgi_display_cols = [c for c in
        ["gene", "drug", "source", "interaction_type", "directionality",
         "approved", "clinical_phase", "n_publications", "composite_score"]
        if c in dgi_df.columns]
    dgi_display = dgi_df[_dgi_display_cols].head(20)

    # Survival table: drop redundant columns
    _surv_display_cols = [c for c in
        ["gene", "HR", "HR_CI_low", "HR_CI_high", "logrank_p", "cox_p",
         "log2FC", "regulation", "prognosis"]
        if c in surv_filtered.columns]
    surv_display = surv_filtered[_surv_display_cols].head(20)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Target Prioritisation Report — HCC</title>
{_CSS}
</head><body>

<h1>Target Prioritisation Report</h1>
<div class="meta"><b>Dataset:</b> GEO GSE166635
<b>Generated:</b> {_now()} &nbsp;|&nbsp; <b>Notebook:</b> 02_target_prioritisation.ipynb</div>

<div class="summary">
  <strong>What this report covers:</strong> A three-stage target prioritisation pipeline
  applied to the differentially expressed genes identified in notebook 01. Stage P1 builds
  a protein–protein interaction (PPI) network using STRING and ranks hub genes by network
  centrality. Stage P2 queries drug databases
  (DGIdb, ChEMBL, OpenTargets) to identify approved or clinical-stage compounds that
  interact with the prioritised genes — generating a ranked list of repurposing candidates.
</div>

<div class="section">
<h2>P1 · Protein-Protein Interaction Network</h2>
<div class="box">
  <table><tr><th>Parameter</th><th>Value</th></tr>
  {_param_row("STRING confidence score", str(string_score))}
  {_param_row("log₂FC threshold", str(log2fc_thresh))}
  {_param_row("adj p threshold", str(padj_thresh))}
  {_param_row("DEGs queried", str(len(gene_list)))}
  {_param_row("Top hub genes shown", str(min(n_hub, 20)))}
  </table><br>
  {_img_tag(figs["ppi"], "PPI network — top hub genes highlighted by degree and hub score")}
  <h3>Top 20 hub genes</h3>
  {hub_df[["gene","degree","hub_score","regulation"]].head(20).to_html(index=False, border=0, classes="")}
</div></div>

# <div class="section">
# <h2>P2 · Survival Filter</h2>
# <div class="box">
#   <div class="{'warn' if is_sim else 'good'}">{surv_note}</div>
#   <table><tr><th>Parameter</th><th>Value</th></tr>
#   {_param_row("KM log-rank p threshold", str(km_p_thresh))}
#   {_param_row("Cox p threshold", str(cox_p_thresh))}
#   {_param_row("HR range filter", f"{hr_min} – {hr_max}")}
#   {_param_row("Genes passing survival filter", str(n_surv))}
#   </table><br>
#   <div class="two-col">
#     {_img_tag(figs["km"],  "Kaplan-Meier survival curves for top candidates")}
#     {_img_tag(figs["cox"], "Cox proportional-hazards forest plot")}
#   </div>
#   {surv_display.to_html(index=False, border=0, classes="", float_format=lambda x: f"{x:.4f}")}
# </div></div>

<div class="section">
<h2>P3 · Drug-Gene Interactions</h2>
<div class="box">
  <table><tr><th>Source</th><th>Enabled</th></tr>
  <tr><td>DGIdb</td><td>{"✓" if use_dgidb else "✗"}</td></tr>
  <tr><td>ChEMBL</td><td>{"✓" if use_chembl else "✗"}</td></tr>
  <tr><td>OpenTargets</td><td>{"✓" if use_opentargets else "✗"}</td></tr>
  <tr><td>Curated (manual)</td><td>{"✓" if use_curated else "✗"}</td></tr>
  </table>
  <p><b>{n_dgi:,}</b> drug-gene interactions found across {len(apis_ok)} active source(s).
  Table below shows the top 20 by composite score.</p>
  {_img_tag(figs["dgi_bar"], "Drug-gene interaction summary — counts by interaction type and approval status")}
  {dgi_display.to_html(index=False, border=0, classes="", float_format=lambda x: f"{x:.4f}")}
</div></div>

</body></html>"""

    out = reports_dir / "02_target_prioritisation_report.html"
    _write(html, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03 — GNN Drug Ranking  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def generate_gnn_report(
    all_results, best_name, ranking,
    edges_df, gene_set, drug_set,
    feat_dim,
    hidden_dim, embed_dim, dropout, lr, weight_decay, n_epochs, patience,
    figures_dir, tables_dir, reports_dir,
):
    figures_dir = Path(figures_dir)
    tables_dir  = Path(tables_dir)
    reports_dir = Path(reports_dir)

    best    = all_results[best_name]
    metrics = best.get("test", best.get("metrics", {}))

    # ── Figures — filenames match what gnn_functions.py actually saves ────────
    figs = {
        "training"   : _png_to_b64(figures_dir / "gnn_training_curves.png"),
        "comparison" : _png_to_b64(figures_dir / "gnn_model_comparison.png"),
        "scatter"    : _png_to_b64(figures_dir / "gnn_predicted_vs_actual.png"),
        "ranking"    : _png_to_b64(figures_dir / "gnn_drug_ranking.png"),
        "network"    : _png_to_b64(figures_dir / "drug_gene_network.png"),
    }

    def _fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)

    # ── Model comparison table rows ───────────────────────────────────────────
    rows = ""
    for name, res in all_results.items():
        m    = res.get("test", res.get("metrics", {}))
        star = " ★" if name == best_name else ""
        rows += (f"<tr><td><b>{name}{star}</b></td>"
                 f"<td>{_fmt(m.get('r2', '-'))}</td>"
                 f"<td>{_fmt(m.get('mse', '-'))}</td>"
                 f"<td>{_fmt(m.get('mae', '-'))}</td></tr>")

    # ── Ranking table — drop one-hot / internal GNN feature columns ──────────
    _rank_display_cols = [c for c in
        ["rank", "drug", "gene", "gnn_score", "original_score", "score_delta",
         "approved", "clinical_phase", "interaction_type", "source"]
        if ranking is not None and c in ranking.columns]

    top_drugs_html = (
        ranking[_rank_display_cols].head(20).to_html(
            index=False, border=0, classes="",
            float_format=lambda x: f"{x:.4f}")
        if ranking is not None
        else "<p class='fig-missing'>No ranking available</p>"
    )

    # ── Approved count in top 20 ──────────────────────────────────────────────
    n_approved_top20 = (
        int(ranking.head(20)["approved"].sum())
        if ranking is not None and "approved" in ranking.columns else "-"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>GNN Drug Ranking Report — HCC</title>
{_CSS}
</head><body>

<h1>GNN Drug Ranking Report</h1>
<div class="meta"><b>Dataset:</b> GEO GSE166635 &nbsp;|&nbsp;
<b>Generated:</b> {_now()} &nbsp;|&nbsp; <b>Notebook:</b> 03_gnn_drug_ranking.ipynb</div>

<div class="summary">
  <strong>What this report covers:</strong> A Graph Neural Network (GNN) pipeline that
  learns drug–gene interaction scores from the prioritised hub gene network built in
  notebook 02. Three GNN architectures are trained and compared (GCN, GAT, GraphSAGE)
  on a bipartite graph of {len(gene_set)} target genes and {len(drug_set):,} drug
  candidates connected by {len(edges_df):,} known interactions. The best-performing
  model re-scores every drug–gene pair and produces a final ranked list of repurposing
  candidates. The GNN score refines the composite score from P3 by incorporating
  structural information from the interaction graph — drugs that are well-connected to
  highly-scored hub genes receive a higher ranking even if their direct interaction
  evidence is sparse.
</div>

<div class="section">
<h2>1 · Graph Construction</h2>
<div class="box">
  <div class="grid">
    {_stat(str(len(gene_set)), "target genes (nodes)")}
    {_stat(f"{len(drug_set):,}", "drug candidates (nodes)")}
    {_stat(f"{len(edges_df):,}", "edges (known interactions)")}
    {_stat(str(feat_dim), "input feature dimensions")}
  </div>
</div></div>

<div class="section">
<h2>2 · Model Configuration</h2>
<div class="box">
  <table><tr><th>Hyperparameter</th><th>Value</th></tr>
  {_param_row("Hidden dimension",          str(hidden_dim))}
  {_param_row("Embedding dimension",       str(embed_dim))}
  {_param_row("Dropout rate",              str(dropout))}
  {_param_row("Learning rate",             str(lr))}
  {_param_row("Weight decay",              str(weight_decay))}
  {_param_row("Max epochs",                str(n_epochs))}
  {_param_row("Early stopping patience",   str(patience))}
  </table>
</div></div>

<div class="section">
<h2>3 · Model Comparison</h2>
<div class="box">
  <table><tr><th>Model</th><th>R²</th><th>MSE</th><th>MAE</th></tr>
  {rows}
  </table>
  <p class="good"><b>Best model: {best_name}</b> — R² {_fmt(metrics.get('r2', '-'))},
  MSE {_fmt(metrics.get('mse', '-'))},
  MAE {_fmt(metrics.get('mae', '-'))}</p>
  <h3>Training loss curves</h3>
  {_img_tag(figs["training"], "Training and validation MSE loss per epoch for all three models. Highlighted panel = best model.")}
  <h3>Metric comparison</h3>
  {_img_tag(figs["comparison"], "R², MSE, and MAE grouped bar chart across all models on the held-out test set.")}
  <h3>Predicted vs. actual scores</h3>
  {_img_tag(figs["scatter"], "Scatter plot of GNN-predicted vs. true composite scores on the test set. Dashed line = perfect prediction.")}
</div></div>

<div class="section">
<h2>4 · Drug Ranking</h2>
<div class="box">
  <p>Drugs are ranked by their GNN-predicted interaction score with the target genes.
  A higher score indicates stronger predicted therapeutic relevance based on the graph
  structure. The <b>score delta</b> (GNN score − original composite score) highlights
  drugs the GNN promotes or demotes relative to the rule-based ranking.</p>
  <div class="grid">
    {_stat(str(n_approved_top20), "approved drugs in top 20")}
    {_stat(str(ranking["drug"].nunique()) if ranking is not None else "-", "unique drugs ranked")}
    {_stat(str(ranking["gene"].nunique()) if ranking is not None else "-", "target genes covered")}
  </div>
  <br>
  {_img_tag(figs["ranking"], f"Top 25 drug candidates ranked by {best_name} GNN score. ★ = approved. | = original composite score. Colour = clinical phase.")}
  <h3>Drug–gene interaction network</h3>
  {_img_tag(figs["network"], f"Bipartite network of top 20 GNN-predicted drug–gene pairs. Gene size = number of top drug connections. Drug colour = clinical phase. Gold border = FDA approved. Edge width & opacity = GNN score.")}
  <h3>Top 20 ranked drug–gene pairs</h3>
  {top_drugs_html}
</div></div>

</body></html>"""

    out = reports_dir / "03_gnn_drug_ranking_report.html"
    _write(html, out)
    return out
