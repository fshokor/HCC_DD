"""
test_pipeline.py
================
Smoke test for the HCC Drug Discovery pipeline.

Runs preprocessing → clustering → annotation using Python-only tools.
No R packages required.

Annotation strategy
-------------------
  Primary   : CellTypist  (Immune_All_High + Immune_All_Low, majority voting)
  Secondary : Marker gene scoring  (11 liver cell-type signatures)

  Final label per cluster = majority vote between CellTypist (×2 weight,
  if available) and marker score (×1 weight). CellTypist is skipped
  gracefully if not installed, in which case marker scoring is used alone.

  R-based tools (ScType, SingleR) are the supplementary annotation options
  available in the full pipeline (notebook 01_scrna_analysis.ipynb).

Usage
-----
    conda activate hcc_drug_discovery
    python scripts/data_download.py      # only needed once
    python scripts/test_pipeline.py

Outputs
-------
    results/figures/test_umap_clusters.png    — UMAP coloured by Leiden cluster
    results/figures/test_umap_annotation.png  — UMAP coloured by cell type
    results/tables/test_cluster_summary.csv   — cell counts per type × sample
    data/processed/adata_annotated_test.h5ad  — full annotated AnnData object
"""

import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive, safe for scripts
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Locate repo root ──────────────────────────────────────────────────────────
def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "paths.py").exists():
            return p
    raise FileNotFoundError(
        "paths.py not found.\n"
        "Run: python scripts/data_download.py"
    )

REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paths import RAW_DIR, PROC_DIR, FIGURES_DIR, TABLES_DIR

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_GENES   = 200
MAX_GENES   = 2500
MAX_MT_PCT  = 5
N_TOP_GENES = 2000
N_NEIGHBORS = 15
N_PCS       = 10
LEIDEN_RES  = 0.5
LEIDEN_COL  = f"leiden_res_{LEIDEN_RES:.2f}"

# CellTypist label → MARKER_SETS key  (substring match, case-insensitive)
_CT_MAP = {
    "macrophage"      : "Macrophage",
    "monocyte"        : "Monocyte",
    "cd8"             : "CD8_T_cell",
    "t cell"          : "T_cell",
    "nk"              : "NK_ILC",
    "b cell"          : "B_cell",
    "plasma"          : "Plasma_cell",
    "dendritic"       : "DC",
    "hepatocyte"      : "Hepatocyte",
    "fibroblast"      : "Fibroblast",
    "endothelial"     : "Endothelial",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
_START = time.time()

def step(label: str) -> None:
    elapsed = time.time() - _START
    pad = "─" * max(0, 50 - len(label))
    print(f"\n── {label} ({elapsed:.0f}s) {pad}")

def banner(text: str, char: str = "=") -> None:
    print(f"\n{char * 60}")
    print(f"  {text}")
    print(f"{char * 60}")

def _ct_to_key(label: str) -> str:
    """Map a raw CellTypist label to a MARKER_SETS key, or '' if no match."""
    low = label.lower()
    for fragment, key in _CT_MAP.items():
        if fragment in low:
            return key
    return ""

def _majority(series):
    return series.value_counts().index[0]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Check environment and data
# ─────────────────────────────────────────────────────────────────────────────
banner("HCC Drug Discovery — Pipeline Test")

step("Checking environment")
try:
    import scanpy as sc
    import numpy as np
    import pandas as pd
    print(f"  scanpy  : {sc.__version__}")
    print(f"  numpy   : {np.__version__}")
    print(f"  pandas  : {pd.__version__}")
except ImportError as e:
    print(f"  ✗ {e}")
    print("  Make sure the conda environment is active:")
    print("    conda activate hcc_drug_discovery")
    sys.exit(1)

from utils.scrna_functions import (
    load_samples, qc_metrics, filter_cells, normalize, select_hvg,
    run_pca, run_umap, marker_score_clusters, MARKER_SETS,
)

step("Checking data files")
missing = []
for sample in ["HCC1", "HCC2"]:
    for fname in ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]:
        p = RAW_DIR / sample / fname
        if not p.exists():
            missing.append(str(p))

if missing:
    print("  ✗ Missing files:")
    for m in missing:
        print(f"      {m}")
    print("  Run: python scripts/data_download.py")
    sys.exit(1)

print("  ✓ HCC1 and HCC2 files present")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
step("Preprocessing")

adata = load_samples(RAW_DIR)
adata = qc_metrics(adata)
adata = filter_cells(adata, MIN_GENES, MAX_GENES, MAX_MT_PCT)
adata = normalize(adata)
adata = select_hvg(adata, n_top_genes=N_TOP_GENES, batch_key="sample")

print(f"  ✓ {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Dimensionality reduction & clustering
# ─────────────────────────────────────────────────────────────────────────────
step("Dimensionality reduction & clustering")

adata = run_pca(adata)
adata = run_umap(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
sc.tl.leiden(adata, key_added=LEIDEN_COL,
             resolution=LEIDEN_RES, flavor="igraph")

n_clusters = adata.obs[LEIDEN_COL].nunique()
print(f"  ✓ {n_clusters} Leiden clusters  (resolution={LEIDEN_RES})")

# Cluster + sample UMAP
sc.pl.umap(adata, color=[LEIDEN_COL, "sample"],
           wspace=0.4, size=6, show=False)
plt.savefig(FIGURES_DIR / "test_umap_clusters.png",
            dpi=150, bbox_inches="tight")
plt.close("all")
print(f"  ✓ {FIGURES_DIR}/test_umap_clusters.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4a. CellTypist annotation (primary, Python-only)
# ─────────────────────────────────────────────────────────────────────────────
step("CellTypist annotation  (primary)")

celltypist_ok = False
try:
    import celltypist
    from celltypist import models as ct_models

    # CellTypist requires normalised, dense, log1p expression
    adata_ct = adata.copy()
    adata_ct.X = adata_ct.layers["counts"]
    sc.pp.normalize_total(adata_ct, target_sum=1e4)
    sc.pp.log1p(adata_ct)
    if hasattr(adata_ct.X, "toarray"):
        adata_ct.X = adata_ct.X.toarray()

    ct_models.download_models(
        force_update=False,
        model=["Immune_All_High.pkl", "Immune_All_Low.pkl"],
    )
    model_high = ct_models.Model.load(model="Immune_All_High.pkl")
    model_low  = ct_models.Model.load(model="Immune_All_Low.pkl")

    pred_high = celltypist.annotate(
        adata_ct, model=model_high,
        majority_voting=True, over_clustering=LEIDEN_COL,
    )
    pred_low = celltypist.annotate(
        adata_ct, model=model_low,
        majority_voting=True, over_clustering=LEIDEN_COL,
    )

    adata.obs["ct_coarse"] = pred_high.to_adata().obs.loc[
        adata.obs.index, "majority_voting"]
    adata.obs["ct_fine"] = pred_low.to_adata().obs.loc[
        adata.obs.index, "majority_voting"]

    # Majority CellTypist label per cluster (for the vote step)
    adata.obs["ct_cluster_label"] = (
        adata.obs.groupby(LEIDEN_COL)["ct_fine"]
        .transform(_majority)
    )

    print(f"  ✓ {adata.obs.ct_coarse.nunique()} coarse types")
    print(f"  ✓ {adata.obs.ct_fine.nunique()} fine types")
    celltypist_ok = True

except ImportError:
    print("  ⚠ celltypist not installed — will use marker scoring only")
    print("    pip install celltypist")
    adata.obs["ct_coarse"] = "Unknown"
    adata.obs["ct_fine"]   = "Unknown"
    adata.obs["ct_cluster_label"] = "Unknown"

except Exception as e:
    print(f"  ⚠ CellTypist failed ({e}) — will use marker scoring only")
    adata.obs["ct_coarse"] = "Unknown"
    adata.obs["ct_fine"]   = "Unknown"
    adata.obs["ct_cluster_label"] = "Unknown"

# ─────────────────────────────────────────────────────────────────────────────
# 4b. Marker gene scoring (secondary, always runs)
# ─────────────────────────────────────────────────────────────────────────────
step("Marker gene scoring  (secondary)")

score_df = marker_score_clusters(
    adata, leiden_col=LEIDEN_COL, marker_sets=MARKER_SETS)

# ─────────────────────────────────────────────────────────────────────────────
# 4c. Assign final cell-type label per cluster
# ─────────────────────────────────────────────────────────────────────────────
step("Majority vote  →  final cell-type label")

cluster_labels = {}
for cl in sorted(adata.obs[LEIDEN_COL].unique(), key=int):

    marker_label = score_df.loc[cl, "best_by_score"]

    ct_raw   = adata.obs.loc[
        adata.obs[LEIDEN_COL] == cl, "ct_cluster_label"
    ].iloc[0]
    ct_label = _ct_to_key(ct_raw)     # "" if no match or CellTypist unavailable

    # Vote: CellTypist gets 2 votes (more curated model),
    #       marker score gets 1 vote (always available)
    votes = [marker_label]
    if ct_label:
        votes += [ct_label, ct_label]

    winner, _ = Counter(votes).most_common(1)[0]
    cluster_labels[cl] = winner

adata.obs["cell_type"] = (
    adata.obs[LEIDEN_COL]
    .astype(str)
    .map(cluster_labels)
    .astype("category")
)

# Print cluster-level assignment table
print(f"\n  {'Cluster':<10} {'Marker best':<22} {'CellTypist':<22} {'Final label'}")
print(f"  {'-'*8:<10} {'-'*20:<22} {'-'*20:<22} {'-'*18}")
for cl in sorted(cluster_labels.keys(), key=int):
    n          = (adata.obs[LEIDEN_COL] == cl).sum()
    mkr        = score_df.loc[cl, "best_by_score"]
    ct_raw     = adata.obs.loc[adata.obs[LEIDEN_COL] == cl, "ct_cluster_label"].iloc[0]
    ct_key     = _ct_to_key(ct_raw) or "—"
    final      = cluster_labels[cl]
    print(f"  {cl:<10} {mkr:<22} {ct_key:<22} {final}  ({n:,} cells)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save outputs
# ─────────────────────────────────────────────────────────────────────────────
step("Saving outputs")

# Annotated UMAP
sc.pl.umap(adata, color=["cell_type", "sample"],
           legend_loc="on data", legend_fontsize=7,
           frameon=False, wspace=0.5, show=False)
plt.savefig(FIGURES_DIR / "test_umap_annotation.png",
            dpi=150, bbox_inches="tight")
plt.close("all")
print(f"  ✓ {FIGURES_DIR}/test_umap_annotation.png")

# Cluster summary CSV
summary = (
    adata.obs.groupby(["cell_type", "sample"])
    .size()
    .unstack(fill_value=0)
    .assign(total=lambda df: df.sum(axis=1))
    .sort_values("total", ascending=False)
)
summary.to_csv(TABLES_DIR / "test_cluster_summary.csv")
print(f"  ✓ {TABLES_DIR}/test_cluster_summary.csv")

# Annotated AnnData
out_h5ad = PROC_DIR / "adata_annotated_test.h5ad"
adata.write(str(out_h5ad))
print(f"  ✓ {out_h5ad}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Final report
# ─────────────────────────────────────────────────────────────────────────────
elapsed = time.time() - _START
banner("Results", char="─")

print(f"  Runtime         : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
print(f"  Cells           : {adata.n_obs:,}")
print(f"  Clusters        : {n_clusters}")
print(f"  Cell types found: {adata.obs.cell_type.nunique()}")
print(f"  Annotation mode : {'CellTypist + marker score' if celltypist_ok else 'marker score only'}")

print("\n  Cell type breakdown:")
ct_counts = adata.obs["cell_type"].value_counts()
for ct, n in ct_counts.items():
    pct = n / adata.n_obs * 100
    bar = "█" * int(pct / 2)
    print(f"    {ct:<22} {n:5,} cells  {pct:5.1f}%  {bar}")

print("\n  Output files:")
for p in [
    FIGURES_DIR / "test_umap_clusters.png",
    FIGURES_DIR / "test_umap_annotation.png",
    TABLES_DIR  / "test_cluster_summary.csv",
    out_h5ad,
]:
    ok   = "✓" if p.exists() else "✗"
    size = f"{p.stat().st_size // 1024} KB" if p.exists() else "MISSING"
    print(f"    {ok}  {p.name:<40} {size}")

# Pass / fail gate
banner("Test result", char="=")
if adata.obs.cell_type.nunique() >= 3:
    print("  ✓  PASSED  — pipeline is working correctly")
else:
    print("  ✗  FAILED  — fewer than 3 cell types identified")
    print("     Check that HCC1 and HCC2 data are present and complete.")
    sys.exit(1)
