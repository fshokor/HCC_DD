"""
gsea_functions.py
=================
All logic for notebook 05 · Gene Set Enrichment Analysis.

Functions
---------
prepare_ranked_list   — build ranked gene list from DEA results
run_gsea_r            — run clusterProfiler GSEA via rpy2
print_gsea_summary    — print top terms per ontology from saved CSVs
"""

from __future__ import annotations
 
from pathlib import Path
import re
 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
 


# HCC-relevant biological themes (used for targeted reporting)
HCC_THEMES = {
    "Lipid metabolism"    : "lipid|fatty.acid|cholesterol|PPAR|lipoprotein",
    "Glycolysis / energy" : "glycolysis|gluconeogenesis|glucose|TCA|oxidative.phosphorylation",
    "PI3K-AKT / Wnt"     : "PI3K|AKT|Wnt|beta.catenin|mTOR|MAPK",
    "Immune regulation"   : "immune|inflamm|cytokine|T.cell|B.cell|interferon|NF.kB",
}

# R script template — executed via ro.r() after setting variables in globalenv
_R_GSEA_SCRIPT = r"""
suppressPackageStartupMessages({
    library(clusterProfiler); library(org.Hs.eg.db)
    library(enrichplot);      library(ggplot2)
})

rnk <- read.table(paste0(PROC_DIR, "/ranked_genes_log2fc.tsv"),
                  sep="\t", header=FALSE, stringsAsFactors=FALSE)
colnames(rnk) <- c("SYMBOL", "log2FC")
rnk <- rnk[order(rnk$log2FC, decreasing=TRUE), ]

mapping  <- bitr(rnk$SYMBOL, fromType="SYMBOL",
                 toType="ENTREZID", OrgDb=org.Hs.eg.db)
rnk2     <- merge(rnk, mapping, by="SYMBOL")
rnk2     <- rnk2[order(rnk2$log2FC, decreasing=TRUE), ]
geneList <- rnk2$log2FC
names(geneList) <- rnk2$ENTREZID

cat(sprintf("Genes mapped to Entrez: %d / %d\n",
            length(geneList), nrow(rnk)))

PARAMS <- list(minGSSize=15, maxGSSize=500,
               pvalueCutoff=0.05, verbose=FALSE)
set.seed(42)

cat("Running GO-BP...\n")
gsea_bp <- do.call(gseGO, c(
    list(geneList=geneList, OrgDb=org.Hs.eg.db,
         ont="BP", keyType="ENTREZID"), PARAMS))

cat("Running GO-MF...\n")
gsea_mf <- do.call(gseGO, c(
    list(geneList=geneList, OrgDb=org.Hs.eg.db,
         ont="MF", keyType="ENTREZID"), PARAMS))

cat("Running GO-CC...\n")
gsea_cc <- do.call(gseGO, c(
    list(geneList=geneList, OrgDb=org.Hs.eg.db,
         ont="CC", keyType="ENTREZID"), PARAMS))

cat("Running KEGG...\n")
gsea_kegg <- do.call(gseKEGG, c(
    list(geneList=geneList, organism="hsa"), PARAMS))

cat(sprintf("\nGO-BP : %d terms\n",   nrow(as.data.frame(gsea_bp))))
cat(sprintf("GO-MF : %d terms\n",   nrow(as.data.frame(gsea_mf))))
cat(sprintf("GO-CC : %d terms\n",   nrow(as.data.frame(gsea_cc))))
cat(sprintf("KEGG  : %d pathways\n", nrow(as.data.frame(gsea_kegg))))

# ── Dot plots ─────────────────────────────────────────────────────────────
safe_dot <- function(obj, title, showCat=15) {
    df <- as.data.frame(obj)
    if (nrow(df) == 0) {
        cat(sprintf("No results for '%s'\n", title)); return()
    }
    p <- dotplot(obj, showCategory=min(showCat, nrow(df))) +
         ggtitle(title) + theme_bw(base_size=11)
    fname <- paste0(FIGURES_DIR, "/gsea_",
                    gsub(" ", "_", tolower(title)), ".png")
    ggsave(fname, p, width=10, height=7, dpi=200)
    cat(sprintf("Saved: %s\n", fname))
    print(p)
}
safe_dot(gsea_bp,   "GO Biological Process")
safe_dot(gsea_mf,   "GO Molecular Function")
safe_dot(gsea_cc,   "GO Cellular Component")
safe_dot(gsea_kegg, "KEGG Pathways")

# ── Ridge plot ────────────────────────────────────────────────────────────
bp_df <- as.data.frame(gsea_bp)
if (nrow(bp_df) > 0) {
    p <- ridgeplot(gsea_bp, showCategory=min(15, nrow(bp_df))) +
         ggtitle("GO-BP enrichment distribution") + theme_bw(base_size=10)
    ggsave(paste0(FIGURES_DIR, "/gsea_ridgeplot_bp.png"),
           p, width=10, height=8, dpi=200)
    cat(sprintf("Saved: %s/gsea_ridgeplot_bp.png\n", FIGURES_DIR))
}

# ── HCC theme plots ───────────────────────────────────────────────────────
themes <- list(
    "Lipid_metabolism"  = "lipid|fatty.acid|cholesterol|PPAR|lipoprotein",
    "Glycolysis_energy" = "glycolysis|gluconeogenesis|glucose|TCA|oxidative.phosphorylation",
    "PI3K_AKT_Wnt"     = "PI3K|AKT|Wnt|beta.catenin|mTOR|MAPK",
    "Immune_regulation" = "immune|inflamm|cytokine|T.cell|B.cell|interferon|NF.kB"
)
for (theme_name in names(themes)) {
    hits <- grep(themes[[theme_name]], bp_df$Description, ignore.case=TRUE)
    if (length(hits) == 0) next
    term_id <- bp_df$ID[hits[1]]
    tryCatch({
        p <- gseaplot2(gsea_bp, geneSetID=term_id,
                       title=paste0(theme_name, ": ",
                                    bp_df$Description[hits[1]]))
        ggsave(paste0(FIGURES_DIR, "/gsea_theme_", theme_name, ".png"),
               p, width=10, height=6, dpi=200)
        cat(sprintf("Saved theme: %s\n", theme_name))
    }, error=function(e) {
        cat(sprintf("Could not plot theme '%s': %s\n",
                    theme_name, conditionMessage(e)))
    })
}

# ── Export CSV tables ─────────────────────────────────────────────────────
write.csv(as.data.frame(gsea_bp),
          paste0(TABLES_DIR, "/gsea_go_bp.csv"),   row.names=FALSE)
write.csv(as.data.frame(gsea_mf),
          paste0(TABLES_DIR, "/gsea_go_mf.csv"),   row.names=FALSE)
write.csv(as.data.frame(gsea_cc),
          paste0(TABLES_DIR, "/gsea_go_cc.csv"),   row.names=FALSE)
write.csv(as.data.frame(gsea_kegg),
          paste0(TABLES_DIR, "/gsea_kegg.csv"),    row.names=FALSE)
cat("CSV tables saved.\n")
"""


# ─────────────────────────────────────────────────────────────────────────────
def prepare_ranked_list(dea_path, proc_dir):
    """
    Build and save a ranked gene list from DEA results.

    Ranks all genes by log2FC (not just significant ones).
    Saves to data/processed/ranked_genes_log2fc.tsv for R.

    Parameters
    ----------
    dea_path : Path
        Path to dea_results.csv.
    proc_dir : Path
        Output directory (data/processed/).

    Returns
    -------
    ranked : pd.DataFrame
        Columns: gene, log2FC — sorted descending by log2FC.
    """
    dea = pd.read_csv(dea_path)
    ranked = (dea[["gene", "log2FC"]].dropna()
              .groupby("gene", as_index=False).mean()
              .sort_values("log2FC", ascending=False))

    out = proc_dir / "ranked_genes_log2fc.tsv"
    ranked.to_csv(out, sep="\t", index=False, header=False)
    print(f"Ranked list: {len(ranked)} genes → {out}")
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
def run_gsea_r(ro, proc_dir, figures_dir, tables_dir):
    """
    Execute the full GSEA pipeline in R via rpy2.

    Passes Python path variables into R's globalenv, then runs
    the embedded R script (_R_GSEA_SCRIPT) which:
      - Maps gene symbols → Entrez IDs
      - Runs gseGO (BP/MF/CC) and gseKEGG
      - Saves dot plots, ridge plot, and theme-specific GSEA plots
      - Exports CSV tables

    Parameters
    ----------
    ro : rpy2.robjects module
        Must already be imported with %load_ext rpy2.ipython.
    proc_dir, figures_dir, tables_dir : Path
        Project paths passed into R as string variables.
    """
    ro.globalenv["PROC_DIR"]    = str(proc_dir)
    ro.globalenv["FIGURES_DIR"] = str(figures_dir)
    ro.globalenv["TABLES_DIR"]  = str(tables_dir)

    try:
        ro.r(_R_GSEA_SCRIPT)
        print("\n✓ GSEA complete")
    except Exception as e:
        print(f"\n✗ R error: {e}")
        print("Check that clusterProfiler and org.Hs.eg.db are installed.")
        print("Run: Rscript env/r_packages.R")


# ─────────────────────────────────────────────────────────────────────────────
def print_gsea_summary(tables_dir):
    """
    Print a concise summary of GSEA results from the saved CSV tables.

    Shows the count of enriched terms per ontology and the top 5 terms
    by NES for each HCC-relevant biological theme.

    Parameters
    ----------
    tables_dir : Path
        results/tables/ directory.
    """
    print("── GSEA results summary ──\n")
    for label, fname in [("GO-BP",  "gsea_go_bp.csv"),
                          ("GO-MF",  "gsea_go_mf.csv"),
                          ("GO-CC",  "gsea_go_cc.csv"),
                          ("KEGG",   "gsea_kegg.csv")]:
        fpath = tables_dir / fname
        if not fpath.exists():
            print(f"  {label}: file not found (R step may have failed)")
            continue
        df = pd.read_csv(fpath)
        print(f"  {label}: {len(df)} enriched terms")
        if len(df) > 0 and "Description" in df.columns and "NES" in df.columns:
            top = df.nlargest(3, "NES")[["Description", "NES", "p.adjust"]]
            print(top.to_string(index=False))
        print()

    bp_file = tables_dir / "gsea_go_bp.csv"
    if not bp_file.exists():
        return

    bp_df = pd.read_csv(bp_file)
    print("── HCC-relevant pathway hits (GO-BP) ──\n")
    for theme, pattern in HCC_THEMES.items():
        hits = bp_df[bp_df["Description"].str.contains(
            pattern, case=False, na=False, regex=True)]
        print(f"  {theme}: {len(hits)} terms")
        if len(hits) > 0:
            top = hits.nlargest(3, "NES")[["Description", "NES"]].to_string(index=False)
            for line in top.split("\n")[1:]:
                print(f"    {line.strip()}")
        print()

# ─────────────────────────────────────────────────────────────────────────────

def query_gene_pathways(
    gene: str,
    tables_dir,
    dea_path=None,
    ontologies: tuple = ("GO-BP", "GO-MF", "GO-CC", "KEGG"),
    adj_p_thresh: float = 0.05,
) -> pd.DataFrame:
    """
    Look up which enriched pathways a gene belongs to (as a leading-edge gene).
 
    Uses the `core_enrichment` column saved by clusterProfiler's gseGO /
    gseKEGG — these are the genes that actually drove the enrichment score
    for each pathway, not just any gene in the gene set.
 
    Parameters
    ----------
    gene : str
        Gene symbol (case-insensitive, e.g. "APOE", "FTL").
    tables_dir : Path or str
        Directory containing gsea_go_bp.csv, gsea_go_mf.csv,
        gsea_go_cc.csv, and gsea_kegg.csv.
    dea_path : Path or str or None
        Path to dea_results.csv (optional).  If supplied, the gene's
        log2FC and adjusted p-value are added to the result table.
    ontologies : tuple
        Which ontologies to search.  Default: all four.
    adj_p_thresh : float
        Only return pathways with p.adjust ≤ this value.
 
    Returns
    -------
    pd.DataFrame with columns:
        ontology, pathway_id, description, NES, adj_p,
        regulation (up / down), [log2FC, dea_padj if dea_path given]
 
    Empty DataFrame if the gene is not a leading-edge gene in any pathway.
 
    Examples
    --------
    >>> result = query_gene_pathways("APOE", TABLES_DIR, PROC_DIR / "dea_results.csv")
    >>> print(result[["description", "NES", "regulation"]])
    """
    tables_dir = Path(tables_dir)
    gene_upper = gene.strip().upper()
 
    # File name map
    ont_files = {
        "GO-BP" : tables_dir / "gsea_go_bp.csv",
        "GO-MF" : tables_dir / "gsea_go_mf.csv",
        "GO-CC" : tables_dir / "gsea_go_cc.csv",
        "KEGG"  : tables_dir / "gsea_kegg.csv",
    }
 
    hits = []
    for ont in ontologies:
        fpath = ont_files.get(ont)
        if fpath is None or not fpath.exists():
            continue
 
        df = pd.read_csv(fpath)
 
        # Normalise column names (clusterProfiler uses different names
        # depending on the version and export method)
        df.columns = [c.strip().lower().replace(".", "_") for c in df.columns]
        desc_col = next(
            (c for c in df.columns if c in ("description", "pathway", "term")),
            df.columns[1],
        )
        id_col = next(
            (c for c in df.columns if c in ("id", "pathway_id", "go_id")),
            df.columns[0],
        )
 
        # Filter by significance
        if "p_adjust" in df.columns:
            df = df[df["p_adjust"] <= adj_p_thresh]
        elif "padj" in df.columns:
            df = df[df["padj"] <= adj_p_thresh]
 
        if "core_enrichment" not in df.columns:
            continue
 
        # Search core_enrichment for the gene (slash-separated symbols)
        mask = df["core_enrichment"].fillna("").str.upper().str.contains(
            rf"(?:^|/)(?:{gene_upper})(?:/|$)", regex=True
        )
        matched = df[mask].copy()
 
        if matched.empty:
            continue
 
        nes_col  = "nes"  if "nes"  in matched.columns else "enrichmentscore"
        padj_col = "p_adjust" if "p_adjust" in matched.columns else "padj"
 
        matched["ontology"]    = ont
        matched["pathway_id"]  = matched[id_col]
        matched["description"] = matched[desc_col]
        matched["NES"]         = pd.to_numeric(matched[nes_col], errors="coerce")
        matched["adj_p"]       = pd.to_numeric(matched[padj_col], errors="coerce")
        matched["regulation"]  = matched["NES"].apply(
            lambda n: "up" if n > 0 else "down"
        )
 
        hits.append(
            matched[["ontology", "pathway_id", "description",
                      "NES", "adj_p", "regulation"]]
        )
 
    if not hits:
        print(f"  '{gene}' is not a leading-edge gene in any enriched pathway "
              f"(adj p ≤ {adj_p_thresh}).")
        return pd.DataFrame(
            columns=["ontology", "pathway_id", "description",
                     "NES", "adj_p", "regulation"]
        )
 
    result = pd.concat(hits, ignore_index=True)
    result = result.sort_values("NES", ascending=False).reset_index(drop=True)
 
    # Optionally enrich with DEA fold-change
    if dea_path is not None:
        dea_path = Path(dea_path)
        if dea_path.exists():
            dea = pd.read_csv(dea_path)
            dea.columns = [c.strip().lower() for c in dea.columns]
            gene_col = next(
                (c for c in dea.columns if c in ("gene", "genes", "symbol")),
                dea.columns[0],
            )
            fc_col = next(
                (c for c in dea.columns if "log2" in c and "fc" in c),
                None,
            )
            pv_col = next(
                (c for c in dea.columns if "adj" in c and "p" in c),
                None,
            )
            row = dea[dea[gene_col].str.upper() == gene_upper]
            if not row.empty:
                if fc_col:
                    result["log2FC"]   = float(row[fc_col].iloc[0])
                if pv_col:
                    result["dea_padj"] = float(row[pv_col].iloc[0])
 
    return result

# ───────────────────────────────────────────────────────────────────────────── 
 
def plot_gene_pathway_membership(
    gene: str,
    result_df: pd.DataFrame,
    figures_dir=None,
    top_n: int = 20,
) -> plt.Figure | None:
    """
    Lollipop chart of NES values for all pathways a gene belongs to.
 
    Up-regulated pathways (NES > 0) are shown in blue; down-regulated
    (NES < 0) in red.  Each dot is sized by –log10(adj_p).
 
    Parameters
    ----------
    gene : str
    result_df : pd.DataFrame
        Output of query_gene_pathways().
    figures_dir : Path or str or None
        If given, saves the figure as
        ``{figures_dir}/gene_pathways_{gene}.png``.
    top_n : int
        Maximum number of pathways to show (takes the ones with the
        largest |NES|).
 
    Returns
    -------
    matplotlib Figure, or None if result_df is empty.
    """
    if result_df.empty:
        print(f"No pathway data to plot for '{gene}'.")
        return None
 
    df = (result_df
          .assign(abs_nes=result_df["NES"].abs())
          .nlargest(top_n, "abs_nes")
          .sort_values("NES")
          .reset_index(drop=True))
 
    # Truncate long pathway names
    df["label"] = df["description"].apply(
        lambda s: (s[:55] + "…") if len(str(s)) > 56 else str(s)
    )
 
    colors  = ["#D85A30" if n < 0 else "#4361ee" for n in df["NES"]]
    sizes   = (-np.log10(df["adj_p"].clip(lower=1e-20)) * 18).clip(lower=20)
 
    fig_h   = max(5.0, len(df) * 0.42)
    fig, ax = plt.subplots(figsize=(11, fig_h), facecolor="white")
 
    # Lollipop stems
    for i, (nes, col) in enumerate(zip(df["NES"], colors)):
        ax.hlines(i, 0, nes, color=col, linewidth=1.4, alpha=0.55)
 
    # Dots
    ax.scatter(df["NES"], range(len(df)),
               c=colors, s=sizes, zorder=3, alpha=0.90, edgecolors="white",
               linewidths=0.5)
 
    # Zero line
    ax.axvline(0, color="#aaa", linewidth=0.9, ls="--", zorder=1)
 
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=8.5)
    ax.set_xlabel("Normalised Enrichment Score (NES)", fontsize=10)
    ax.set_title(
        f"Pathways where  {gene.upper()}  is a leading-edge gene\n"
        f"Blue = up-regulated in tumour (NES > 0)  ·  "
        f"Red = down-regulated (NES < 0)  ·  "
        f"Dot size = −log₁₀(adj p)",
        fontsize=10, pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
 
    # Ontology colour legend
    ont_present = df["ontology"].unique()
    ont_markers = [mpatches.Patch(color="#ccc", label=o) for o in ont_present]
    ax.legend(handles=ont_markers, loc="lower right",
              fontsize=8, title="Ontology", title_fontsize=8,
              framealpha=0.85)
 
    # Annotate ontology next to each dot
    for i, (nes, ont) in enumerate(zip(df["NES"], df["ontology"])):
        ax.text(nes + (0.03 if nes >= 0 else -0.03), i,
                ont, va="center",
                ha="left" if nes >= 0 else "right",
                fontsize=7, color="#555")
 
    plt.tight_layout()
 
    if figures_dir is not None:
        out = Path(figures_dir) / f"gene_pathways_{gene.upper()}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
 
    return fig


# ── Biological theme definitions ─────────────────────────────────────────────
# Each theme maps to a list of keyword patterns.
# The first matching theme wins; order matters for ambiguous pathways.
PATHWAY_THEMES = {
    "Metabolic process": [
        r"lipid", r"fatty.acid", r"cholesterol", r"glycol", r"glucon",
        r"oxidative.phosphorylation", r"TCA", r"metaboli", r"PPAR",
    ],
    "Immune regulation": [
        r"immune", r"inflamm", r"cytokine", r"antigen.process",
        r"T.cell", r"B.cell", r"interferon", r"NF.?kB", r"innate",
        r"adaptive", r"lymphocyte", r"macrophage", r"neutrophil",
    ],
    "Oncogenic signaling": [
        r"PI3K", r"AKT", r"mTOR", r"Wnt", r"beta.catenin", r"MAPK",
        r"RAS", r"p53", r"apoptosis", r"cell.cycle", r"proliferat",
        r"cancer", r"tumor", r"oncogen",
    ],
    "Cellular interaction": [
        r"focal.adhesion", r"ECM", r"extracellular.matrix", r"integrin",
        r"cell.adhesion", r"cell.migration", r"angiogenesis",
        r"collagen", r"fibronectin",
    ],
}
 
# Fallback theme for pathways that don't match any keyword group
_OTHER_THEME = "Other biological processes"
 
 
def _assign_theme(description: str) -> str:
    desc = str(description)
    for theme, patterns in PATHWAY_THEMES.items():
        for pat in patterns:
            if re.search(pat, desc, re.IGNORECASE):
                return theme
    return _OTHER_THEME
 
 
def _top_genes(core_enrichment: str, dea_df: pd.DataFrame | None,
               n: int = 3) -> str:
    """
    Return the top n leading-edge genes with direction arrows if DEA is given.
    Genes are ordered by |log2FC| descending when DEA is available.
    """
    if not isinstance(core_enrichment, str) or not core_enrichment.strip():
        return "—"
 
    genes = [g.strip() for g in core_enrichment.split("/") if g.strip()]
 
    if dea_df is not None and not dea_df.empty:
        # Normalise DEA column names
        dea = dea_df.copy()
        dea.columns = [c.strip().lower() for c in dea.columns]
        gene_col = next(
            (c for c in dea.columns if c in ("gene", "genes", "symbol")),
            dea.columns[0],
        )
        fc_col = next(
            (c for c in dea.columns if "log2" in c and "fc" in c), None
        )
 
        if fc_col:
            fc_map = (
                dea[[gene_col, fc_col]]
                .dropna()
                .set_index(gene_col)[fc_col]
                .to_dict()
            )
            # Sort by |log2FC| descending, annotate direction
            scored = sorted(
                genes,
                key=lambda g: abs(fc_map.get(g.upper(), 0)),
                reverse=True,
            )
            top = scored[:n]
            annotated = []
            for g in top:
                fc = fc_map.get(g.upper(), None)
                if fc is None:
                    annotated.append(g)
                elif fc > 0:
                    annotated.append(f"{g} ↑")
                else:
                    annotated.append(f"{g} ↓")
            return ", ".join(annotated)
 
    return ", ".join(genes[:n])
 
 
def _bio_function(theme: str, description: str) -> str:
    """Map theme → concise biological function label."""
    _MAP = {
        "Metabolic process"   : "Energy production, metabolic adaptation",
        "Immune regulation"   : "Inflammation and immune modulation",
        "Oncogenic signaling" : "Tumour growth and survival mechanisms",
        "Cellular interaction": "Cell communication and tissue integrity",
        _OTHER_THEME          : "Other regulatory processes",
    }
    return _MAP.get(theme, "Regulatory process")
 
 
def generate_pathway_summary_table(
    tables_dir,
    figures_dir=None,
    dea_path=None,
    adj_p_thresh: float = 0.05,
    top_genes_per_row: int = 3,
    ontologies: tuple = ("GO-BP", "GO-MF", "GO-CC", "KEGG"),
    extra_themes: dict | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Build a publication-style pathway summary table (like Table 1 of Wang et al.).
 
    Parameters
    ----------
    tables_dir : Path or str
        Directory containing gsea_go_bp.csv, gsea_go_mf.csv,
        gsea_go_cc.csv, and / or gsea_kegg.csv.
    figures_dir : Path or str or None
        If given, saves pathway_summary_table.png here.
    dea_path : Path or str or None
        Path to dea_results.csv.  If given, gene direction arrows (↑↓)
        are added and genes are ranked by |log2FC|.
    adj_p_thresh : float
        Only include pathways with p.adjust ≤ this value.
    top_genes_per_row : int
        Number of leading-edge genes to list per row (default 3).
    ontologies : tuple
        Which GSEA result files to include.
    extra_themes : dict or None
        Additional theme → [pattern, ...] entries to add to PATHWAY_THEMES.
 
    Returns
    -------
    summary_df : pd.DataFrame
        Columns: Pathway, Enriched pathway, Key genes, Biological function
    html_table : str
        Self-contained HTML <table> element for embedding in reports.
 
    Examples
    --------
    >>> df, html = generate_pathway_summary_table(
    ...     TABLES_DIR,
    ...     figures_dir=FIGURES_DIR,
    ...     dea_path=PROC_DIR / "dea_results.csv",
    ... )
    >>> display(df)
    """
    tables_dir = Path(tables_dir)
 
    # Optionally extend themes
    themes = dict(PATHWAY_THEMES)
    if extra_themes:
        themes.update(extra_themes)
 
    # Load DEA for gene annotation
    dea_df = None
    if dea_path is not None:
        p = Path(dea_path)
        if p.exists():
            dea_df = pd.read_csv(p)
 
    # File map
    ont_files = {
        "GO-BP": tables_dir / "gsea_go_bp.csv",
        "GO-MF": tables_dir / "gsea_go_mf.csv",
        "GO-CC": tables_dir / "gsea_go_cc.csv",
        "KEGG" : tables_dir / "gsea_kegg.csv",
    }
 
    # ── Load and combine all GSEA results ────────────────────────────────────
    all_rows = []
    for ont in ontologies:
        fpath = ont_files.get(ont)
        if fpath is None or not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        df.columns = [c.strip() for c in df.columns]
 
        # Normalise key column names
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("description", "pathway", "term"):
                col_map[c] = "Description"
            elif cl in ("p.adjust", "padj", "p_adjust"):
                col_map[c] = "padj"
            elif cl in ("nes", "enrichmentscore", "enrichment_score"):
                col_map[c] = "NES"
            elif cl in ("core_enrichment", "leadingedge", "leading_edge"):
                col_map[c] = "core_enrichment"
        df = df.rename(columns=col_map)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        for req in ("Description", "padj", "NES"):
            if req not in df.columns:
                continue

        df = df[df["padj"] <= adj_p_thresh].copy()
        df["ontology"] = ont
        all_rows.append(df)
 
    if not all_rows:
        print("No significant GSEA results found — check adj_p_thresh or file paths.")
        return pd.DataFrame(), ""
 
    combined = pd.concat(all_rows, ignore_index=True)
 
    # ── Assign themes and pick best pathway per theme ─────────────────────────
    combined["theme"] = combined["Description"].apply(_assign_theme)
 
    theme_order = list(themes.keys()) + [_OTHER_THEME]
    rows = []
 
    for theme in theme_order:
        subset = combined[combined["theme"] == theme]
        if subset.empty:
            continue
        # Best = lowest adj p; break ties with largest |NES|
        best = (subset
                .assign(abs_nes=subset["NES"].abs())
                .sort_values(["padj", "abs_nes"], ascending=[True, False])
                .iloc[0])
 
        # Collect all pathway names in this theme for the "Enriched pathway" cell
        desc_list = subset.nsmallest(2, "padj")["Description"].tolist()
        enriched_str = " and ".join(desc_list) if len(desc_list) > 1 else desc_list[0]
        # Capitalise first letter, truncate if very long
        enriched_str = enriched_str[:80]
 
        key_genes = (
            _top_genes(best.get("core_enrichment", ""), dea_df,
                       n=top_genes_per_row)
            if "core_enrichment" in best.index else "—"
        )
 
        rows.append({
            "Pathway"           : theme,
            "Enriched pathway"  : enriched_str,
            "Key genes"         : key_genes,
            "Biological function": _bio_function(theme, best["Description"]),
            "_NES"              : best["NES"],
            "_padj"             : best["padj"],
            "_ontology"         : best["ontology"],
        })
 
    if not rows:
        print("No rows generated — no pathways matched any theme.")
        return pd.DataFrame(), ""
 
    summary_df = pd.DataFrame(rows)
    display_df = summary_df[
        ["Pathway", "Enriched pathway", "Key genes", "Biological function"]
    ].reset_index(drop=True)
 
    # ── Save CSV ──────────────────────────────────────────────────────────────
    if figures_dir is not None:
        csv_out = Path(tables_dir) / "pathway_summary_table.csv"
        display_df.to_csv(csv_out, index=False)
        print(f"Saved CSV : {csv_out}")
 
    # ── Generate publication figure ───────────────────────────────────────────
    if figures_dir is not None:
        fig = _render_table_figure(display_df)
        out = Path(figures_dir) / "pathway_summary_table.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PNG : {out}")
 
    # ── Generate HTML snippet ─────────────────────────────────────────────────
    html_table = _render_html_table(display_df)
 
    return display_df, html_table


# ── Figure renderer ───────────────────────────────────────────────────────────
 
def _render_table_figure(df: pd.DataFrame) -> plt.Figure:
    """Render the summary DataFrame as a styled publication-style table figure."""
 
    n_rows = len(df)
    n_cols = len(df.columns)
 
    # Dynamic figure height
    fig_h = 1.2 + n_rows * 0.72
    fig, ax = plt.subplots(figsize=(14, fig_h), facecolor="white")
    ax.axis("off")
 
    # Title — placed above the table with enough margin
    fig.text(
        0.5, 0.995,
        "Table · Summary of key enriched pathways, associated genes, "
        "and biological functions",
        ha="center", va="top",
        fontsize=11, fontweight="bold", color="#1a1a2e",
        transform=fig.transFigure,
    )
 
    # Column widths (fractions of figure width)
    col_widths = [0.16, 0.28, 0.26, 0.28]
    col_headers = list(df.columns)
 
    # Header colours
    HEADER_BG   = "#2c2c2c"
    HEADER_FG   = "white"
    ROW_BG_ODD  = "#f9f9f9"
    ROW_BG_EVEN = "white"
    BORDER_COL  = "#cccccc"
 
    # Theme accent colours (left column)
    THEME_COLS = {
        "Metabolic process"   : "#4361ee",
        "Immune regulation"   : "#7209b7",
        "Oncogenic signaling" : "#d85a30",
        "Cellular interaction": "#1D9E75",
        "Other biological processes": "#888780",
    }
 
    # Build table data as list-of-lists (wrap long text manually)
    def _wrap(text, max_chars=48):
        """Simple word-wrap for table cells."""
        words, lines, cur = text.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = f"{cur} {w}".strip()
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return "\n".join(lines)
 
    cell_data = [[_wrap(str(v), 44) for v in row] for _, row in df.iterrows()]
 
    # ── Draw using matplotlib ─────────────────────────────────────────────────
    row_height = 0.78 / (n_rows + 1)   # +1 for header
    header_y   = 0.85                   # start below title
    table_left  = 0.02
    table_width = 0.96
 
    # Header row
    x = table_left
    for ci, (hdr, cw) in enumerate(zip(col_headers, col_widths)):
        w = cw * table_width
        rect = plt.Rectangle(
            (x, header_y), w, row_height,
            transform=fig.transFigure,
            facecolor=HEADER_BG, edgecolor=BORDER_COL,
            linewidth=0.6, clip_on=False,
        )
        fig.add_artist(rect)
        fig.text(
            x + w / 2, header_y + row_height / 2,
            hdr,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=HEADER_FG,
            transform=fig.transFigure,
        )
        x += w
 
    # Data rows
    for ri, row_vals in enumerate(cell_data):
        row_y  = header_y - (ri + 1) * row_height
        bg_col = ROW_BG_ODD if ri % 2 == 0 else ROW_BG_EVEN
        x      = table_left
        theme  = df.iloc[ri]["Pathway"]
        accent = THEME_COLS.get(theme, "#888780")
 
        for ci, (val, cw) in enumerate(zip(row_vals, col_widths)):
            w = cw * table_width
 
            # Background
            rect = plt.Rectangle(
                (x, row_y), w, row_height,
                transform=fig.transFigure,
                facecolor=bg_col, edgecolor=BORDER_COL,
                linewidth=0.5, clip_on=False,
            )
            fig.add_artist(rect)
 
            # Accent bar on the left edge of the first column
            if ci == 0:
                bar = plt.Rectangle(
                    (x, row_y), 0.004, row_height,
                    transform=fig.transFigure,
                    facecolor=accent, edgecolor="none",
                    clip_on=False,
                )
                fig.add_artist(bar)
 
            # Cell text
            fontweight = "bold" if ci == 0 else "normal"
            fig.text(
                x + w / 2, row_y + row_height / 2,
                val,
                ha="center", va="center",
                fontsize=8.5, color="#1a1a2e",
                fontweight=fontweight,
                transform=fig.transFigure,
                multialignment="center",
            )
            x += w
 
    # Bottom border
    fig.add_artist(plt.Line2D(
        [table_left, table_left + table_width],
        [header_y - n_rows * row_height,
         header_y - n_rows * row_height],
        transform=fig.transFigure,
        color=BORDER_COL, linewidth=0.8,
    ))
 
    # Colour legend for themes
    patches = [
        mpatches.Patch(color=c, label=t)
        for t, c in THEME_COLS.items()
        if t in df["Pathway"].values
    ]
    if patches:
        fig.legend(
            handles=patches,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.01),
            fontsize=7.5, framealpha=0.9,
            title="Biological theme", title_fontsize=8,
            ncol=len(patches),
        )
 
    return fig


# ── HTML renderer ─────────────────────────────────────────────────────────────
 
def _render_html_table(df: pd.DataFrame) -> str:
    """Return a styled HTML table string for embedding in reports."""
 
    THEME_COLS_HEX = {
        "Metabolic process"   : "#4361ee",
        "Immune regulation"   : "#7209b7",
        "Oncogenic signaling" : "#d85a30",
        "Cellular interaction": "#1D9E75",
        "Other biological processes": "#888780",
    }
 
    header_html = "".join(
        f"<th style='background:#2c2c2c;color:white;padding:8px 12px;"
        f"font-size:13px;text-align:left;border:1px solid #555'>{col}</th>"
        for col in df.columns
    )
 
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg    = "#f9f9f9" if i % 2 == 0 else "white"
        theme = row["Pathway"]
        acc   = THEME_COLS_HEX.get(theme, "#888780")
        cells = ""
        for j, (col, val) in enumerate(row.items()):
            border_left = (
                f"border-left:4px solid {acc};" if j == 0 else ""
            )
            fw = "font-weight:600;" if j == 0 else ""
            cells += (
                f"<td style='padding:8px 12px;font-size:12.5px;"
                f"border:1px solid #dee2e6;background:{bg};"
                f"{border_left}{fw}'>{val}</td>"
            )
        rows_html += f"<tr>{cells}</tr>"
 
    return (
        f"<table style='border-collapse:collapse;width:100%;font-family:"
        f"\"Segoe UI\",sans-serif;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
    )
 