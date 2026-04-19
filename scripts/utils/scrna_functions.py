"""
scrna_functions.py
==================
Shared scRNA-seq utility functions for notebooks 01, 02, and 03.

Functions
---------
load_samples        — load HCC1 / HCC2 MTX files and merge
qc_metrics          — annotate mitochondrial, ribosomal, haemoglobin genes
filter_cells        — apply gene count and MT% thresholds
normalize           — normalize_total + log1p, preserve raw counts
select_hvg          — highly variable gene selection
run_pca             — PCA + elbow plot
run_umap            — neighbor graph + UMAP
run_leiden          — Leiden clustering at multiple resolutions
save_adata          — write AnnData to disk with progress message
"""

import numpy as np
import scanpy as sc


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 01 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(raw_dir):
    """
    Load HCC1 (normal-adjacent) and HCC2 (tumor) 10x MTX triplets and merge.

    Parameters
    ----------
    raw_dir : Path
        Directory containing HCC1/ and HCC2/ subdirectories.

    Returns
    -------
    adata : AnnData
        Merged object with obs["sample"] = "normal (HCC1)" or "tumor (HCC2)".
    """
    adata1 = sc.read_10x_mtx(raw_dir / "HCC1", var_names="gene_symbols")
    adata2 = sc.read_10x_mtx(raw_dir / "HCC2", var_names="gene_symbols")

    adata1.obs["sample"] = "HCC1"
    adata2.obs["sample"] = "HCC2"
    adata1.var_names_make_unique()
    adata2.var_names_make_unique()

    adata = adata1.concatenate(adata2, batch_key="sample")
    adata.obs["sample"] = adata.obs["sample"].map(
        {"0": "normal (HCC1)", "1": "tumor (HCC2)"})

    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    print(adata.obs["sample"].value_counts().to_string())
    return adata


def qc_metrics(adata):
    """
    Annotate genes as mitochondrial / ribosomal / haemoglobin and
    compute per-cell QC metrics (counts, n_genes, pct_counts_mt).

    Parameters
    ----------
    adata : AnnData  (in-place, also returned)
    """
    adata.var["mt"]   = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"]   = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
    return adata


def filter_cells(adata, min_genes=200, max_genes=2500, max_mt_pct=5):
    """
    Remove low-quality cells.

    Parameters
    ----------
    adata : AnnData
    min_genes : int
        Minimum detected genes per cell (removes empty droplets).
    max_genes : int
        Maximum detected genes per cell (removes potential doublets).
    max_mt_pct : float
        Maximum mitochondrial gene percentage (removes dying cells).

    Returns
    -------
    adata : AnnData
        Filtered copy.
    """
    print(f"Before filtering : {adata.n_obs} cells")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    print(f"After min_genes  : {adata.n_obs} cells")
    sc.pp.filter_cells(adata, max_genes=max_genes)
    print(f"After max_genes  : {adata.n_obs} cells")
    adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct].copy()
    print(f"After MT filter  : {adata.n_obs} cells")
    return adata


def normalize(adata, target_sum=1e4):
    """
    Preserve raw counts in a layer, then normalize and log-transform.

    Parameters
    ----------
    adata : AnnData  (modified in-place)
    target_sum : float
        Library size to normalize to (default 10,000).

    Returns
    -------
    adata : AnnData
    """
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    print(f"Max expression after log1p: {adata.X.max():.2f}")
    return adata


def select_hvg(adata, n_top_genes=2000, batch_key="sample"):
    """
    Select highly variable genes (HVGs) for downstream analysis.

    Parameters
    ----------
    adata : AnnData  (modified in-place)
    n_top_genes : int
    batch_key : str
        Column in adata.obs used for batch-aware HVG selection.

    Returns
    -------
    adata : AnnData
    """
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,
                                batch_key=batch_key)
    print(f"HVGs selected: {adata.var.highly_variable.sum()}")
    return adata


def save_adata(adata, path):
    """Write AnnData to disk at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(path))
    print(f"Saved: {path}  ({adata.n_obs} cells × {adata.n_vars} genes)")


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02 — Clustering
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(adata, n_pcs=50):
    """
    Run PCA and display the variance-ratio elbow plot.

    Parameters
    ----------
    adata : AnnData  (modified in-place)
    n_pcs : int
        Number of principal components to compute.

    Returns
    -------
    adata : AnnData
    """
    sc.tl.pca(adata)
    sc.pl.pca_variance_ratio(adata, n_pcs=n_pcs, log=True)
    sc.pl.pca(adata, color=["sample", "pct_counts_mt"],
              dimensions=[(0, 1), (2, 3)], size=6)
    return adata


def run_umap(adata, n_neighbors=15, n_pcs=10):
    """
    Build the kNN neighbor graph and compute UMAP embedding.

    Parameters
    ----------
    adata : AnnData  (modified in-place)
    n_neighbors : int
        Number of nearest neighbors (k).
    n_pcs : int
        PCs used for neighbor computation.

    Returns
    -------
    adata : AnnData
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    print(f"UMAP computed  (n_neighbors={n_neighbors}, n_pcs={n_pcs})")
    return adata


def run_leiden(adata, resolutions=(0.3, 0.5, 1.0, 2.0)):
    """
    Run Leiden clustering at multiple resolutions.

    Each resolution is stored as adata.obs[f"leiden_res_{res:.2f}"].
    A UMAP grid showing all resolutions is plotted.

    Parameters
    ----------
    adata : AnnData  (modified in-place)
    resolutions : tuple of float

    Returns
    -------
    adata : AnnData
    """
    for res in resolutions:
        sc.tl.leiden(adata, key_added=f"leiden_res_{res:.2f}",
                     resolution=res, flavor="igraph")
        n_clusters = adata.obs[f"leiden_res_{res:.2f}"].nunique()
        print(f"  res={res:.2f}  →  {n_clusters} clusters")

    cols = [f"leiden_res_{r:.2f}" for r in resolutions]
    sc.pl.umap(adata, color=cols, legend_loc="on data")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03 — Annotation
# ─────────────────────────────────────────────────────────────────────────────

def run_celltypist(adata, mode="both"):
    """
    Annotate cells using CellTypist models.

    Parameters
    ----------
    adata : AnnData
        Input object. Must have raw counts in adata.layers["counts"].
    mode : {"high", "low", "both"}
        Which CellTypist model(s) to run:
          "high" — Immune_All_High  (coarse, ~30 broad types)
                   Adds adata.obs["celltypist_coarse"]
          "low"  — Immune_All_Low   (fine-grained, ~200 subtypes)
                   Adds adata.obs["celltypist_fine"]
          "both" — Run both models (default)
                   Adds both columns above

    Returns
    -------
    adata : AnnData
        Modified in-place, also returned.
    """
    import celltypist
    from celltypist import models

    if mode not in ("high", "low", "both"):
        raise ValueError(f"mode must be 'high', 'low', or 'both' — got '{mode}'")

    print(f"CellTypist mode: {mode}")

    # ── Prepare normalised dense copy ─────────────────────────────────────────
    adata_ct = adata.copy()
    adata_ct.X = adata_ct.layers["counts"]
    sc.pp.normalize_total(adata_ct, target_sum=1e4)
    sc.pp.log1p(adata_ct)
    if hasattr(adata_ct.X, "toarray"):
        adata_ct.X = adata_ct.X.toarray()

    # ── Download only what we need ────────────────────────────────────────────
    models_to_download = []
    if mode in ("high", "both"):
        models_to_download.append("Immune_All_High.pkl")
    if mode in ("low", "both"):
        models_to_download.append("Immune_All_Low.pkl")

    models.download_models(force_update=False, model=models_to_download)

    # ── Run selected model(s) ─────────────────────────────────────────────────
    if mode in ("high", "both"):
        model_high = models.Model.load(model="Immune_All_High.pkl")
        pred_high  = celltypist.annotate(adata_ct, model=model_high,
                                         majority_voting=True)
        adata.obs["celltypist_coarse"] = (pred_high.to_adata()
                                          .obs.loc[adata.obs.index, "majority_voting"])
        print(f"  coarse types (High model): {adata.obs.celltypist_coarse.nunique()}")

    if mode in ("low", "both"):
        model_low = models.Model.load(model="Immune_All_Low.pkl")
        pred_low  = celltypist.annotate(adata_ct, model=model_low,
                                        majority_voting=True)
        adata.obs["celltypist_fine"] = (pred_low.to_adata()
                                        .obs.loc[adata.obs.index, "majority_voting"])
        print(f"  fine types   (Low model) : {adata.obs.celltypist_fine.nunique()}")

    # ── Guarantee both columns always exist (fill absent one with placeholder) -
    if mode == "high" and "celltypist_fine" not in adata.obs.columns:
        adata.obs["celltypist_fine"] = "not_run"
    if mode == "low" and "celltypist_coarse" not in adata.obs.columns:
        adata.obs["celltypist_coarse"] = "not_run"

    return adata


def prep_seurat_object(adata, ro, default_converter, anndata2ri):
    """
    Copy adata, add logcounts layer, and push to R global env as adata_seurat.

    Parameters
    ----------
    adata : AnnData
    ro, default_converter, anndata2ri : rpy2 objects
        Must already be imported and R extension loaded in the notebook.

    Returns
    -------
    adata_seurat : AnnData copy (the R object is set in ro.globalenv)
    """
    from rpy2.robjects.conversion import localconverter

    adata_seurat = adata.copy()
    del adata_seurat.uns
    adata_seurat.layers["logcounts"] = adata_seurat.X.copy()

    with localconverter(default_converter + anndata2ri.converter):
        r_adata = ro.conversion.py2rpy(adata_seurat)
    ro.globalenv["adata_seurat"] = r_adata
    print("adata_seurat pushed to R global environment")
    return adata_seurat


def pull_r_col(adata, ro, default_converter, pandas2ri, col_name, obs_col):
    """
    Pull a single column from R's colData(adata_seurat) into adata.obs.

    Parameters
    ----------
    adata : AnnData
    ro, default_converter, pandas2ri : rpy2 objects
    col_name : str
        Column name in colData (R side).
    obs_col : str
        Column name to create in adata.obs (Python side).

    Returns
    -------
    adata : AnnData  (modified in-place)
    """
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + pandas2ri.converter):
        col_data = ro.conversion.rpy2py(
            ro.r("as.data.frame(colData(adata_seurat))"))
    adata.obs[obs_col] = col_data[col_name].values
    adata.obs[obs_col] = adata.obs[obs_col].astype("category")
    print(f"Pulled '{col_name}' → adata.obs['{obs_col}']")
    print(adata.obs[obs_col].value_counts().to_string())
    return adata


MARKER_SETS = {
    "Macrophage" : ["CD68","MARCO","CSF1R","MRC1","VSIG4","GPNMB",
                    "SPP1","C1QA","C1QB","TIMD4"],
    "Monocyte"   : ["CD14","LYZ","S100A8","S100A9","FCN1","VCAN","CXCL8"],
    "T_cell"     : ["CD3D","CD3E","TRAC","TRBC2","IL7R","CD2"],
    "CD8_T_cell" : ["CD8A","CD8B","GZMK","GZMA","GZMB","PRF1","CCL5"],
    "NK_ILC"     : ["NKG7","GNLY","NCAM1","KLRB1","KLRD1","TYROBP"],
    "B_cell"     : ["MS4A1","CD79A","CD79B","IGHM","IGHD"],
    "Plasma_cell": ["MZB1","JCHAIN","IGHG1","XBP1","SDC1","PRDM1"],
    "DC"         : ["FCER1A","CLEC10A","CST3","CLEC9A","CD1C"],
    "Hepatocyte" : ["ALB","APOC3","TTR","FGB","FGG","CYP3A4",
                    "GPC3","APOE","FABP1"],
    "Fibroblast" : ["COL1A1","COL1A2","COL3A1","DCN","LUM",
                    "ACTA2","PDGFRB","FAP"],
    "Endothelial": ["PECAM1","VWF","CDH5","CLDN5","LYVE1","ENG"],
}


def marker_score_clusters(adata, leiden_col="leiden_res_0.50",
                          marker_sets=None):
    """
    Compute mean log-normalised expression of each cell-type marker set
    per Leiden cluster, returning the best-scoring type per cluster.

    Parameters
    ----------
    adata : AnnData
    leiden_col : str
        adata.obs column with cluster assignments.
    marker_sets : dict or None
        {cell_type: [gene_list]}. Defaults to MARKER_SETS.

    Returns
    -------
    score_df : pd.DataFrame
        Clusters × cell-types score matrix with a 'best_by_score' column.
    """
    import numpy as np
    import pandas as pd

    if marker_sets is None:
        marker_sets = MARKER_SETS

    rows = []
    for cl in sorted(adata.obs[leiden_col].unique(), key=int):
        mask = adata.obs[leiden_col] == cl
        row  = {"cluster": cl}
        for ct, markers in marker_sets.items():
            present = [g for g in markers if g in adata.var_names]
            if present:
                expr = adata[mask, present].X
                if hasattr(expr, "toarray"):
                    expr = expr.toarray()
                row[ct] = float(expr.mean())
            else:
                row[ct] = 0.0
        rows.append(row)

    score_df = pd.DataFrame(rows).set_index("cluster")
    score_df["best_by_score"] = score_df.idxmax(axis=1)
    print("Best cell type by marker score per cluster:")
    print(score_df["best_by_score"].to_string())
    return score_df


def majority_vote(adata, score_df, leiden_col="leiden_res_0.50"):
    """
    Reconcile CellTypist, ScType, SingleR, and marker scores into a
    single cell-type label via 4-way majority vote.

    ScType gets double weight for parenchymal cell types
    (Hepatocyte, Fibroblast, Endothelial) because it uses a liver-specific DB.

    Parameters
    ----------
    adata : AnnData  (modified in-place, also returned)
    score_df : pd.DataFrame
        Output of marker_score_clusters().
    leiden_col : str

    Returns
    -------
    adata : AnnData
        With new adata.obs["manual_celltype"] column.
    vote_df : pd.DataFrame
        Per-cluster vote summary.
    """
    import pandas as pd
    from collections import Counter

    def _majority(series):
        return series.value_counts().index[0]

    vote_df = pd.DataFrame({
        "n_cells"     : adata.obs.groupby(leiden_col).size(),
        "CellTypist"  : adata.obs.groupby(leiden_col)["celltypist_fine"].apply(_majority),
        "ScType"      : adata.obs.groupby(leiden_col)["sctype_cell_type"].apply(_majority),
        "SingleR_HPCA": adata.obs.groupby(leiden_col)["SingleR_HPCA"].apply(_majority),
    })

    parenchymal = {"Hepatocyte", "Fibroblast", "Endothelial"}

    def _assign(row):
        votes = [row["CellTypist"], row["ScType"], row["SingleR_HPCA"],
                 score_df.loc[row.name, "best_by_score"]]
        if row["ScType"] in parenchymal:
            votes.append(row["ScType"])   # double weight
        winner, _ = Counter(votes).most_common(1)[0]
        return winner

    vote_df["final_label"] = vote_df.apply(_assign, axis=1)
    print("Cluster → final label:")
    print(vote_df[["n_cells", "final_label"]].to_string())

    cluster_map = vote_df["final_label"].to_dict()
    adata.obs["manual_celltype"] = (
        adata.obs[leiden_col]
        .astype(str)
        .map(cluster_map)
        .astype("category"))
    return adata, vote_df
