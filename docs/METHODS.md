# Methods

Detailed computational methods for the HCC drug discovery pipeline.
Corresponds to Wang et al. (2025) *npj Precision Oncology* 9:309
([doi:10.1038/s41698-025-00952-3](https://doi.org/10.1038/s41698-025-00952-3))
and the extensions described in this repository.

---

## Dataset

Single-cell RNA sequencing data were obtained from NCBI GEO accession
[GSE166635](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166635).
The dataset contains cells from two conditions profiled on the 10x Genomics
Chromium platform:

| Sample | Condition | Cells (raw) |
|--------|-----------|-------------|
| HCC1 | Tumor-adjacent normal tissue | ~12,500 |
| HCC2 | Tumor tissue | ~12,700 |

Raw count matrices (MTX format) are downloaded automatically by
`scripts/data_download.py` from the NCBI GEO FTP server.

---

## 01 · Preprocessing (`01_preprocessing.ipynb`, `scripts/scrna_functions.py`)

Raw count matrices were loaded with Scanpy (v≥1.10) using `sc.read_10x_mtx`.
The two samples were merged with `concatenate`, preserving sample identity in
`adata.obs["sample"]`.

**QC metrics** were computed per cell: number of detected genes
(`n_genes_by_counts`), total RNA counts (`total_counts`), and percentage of
mitochondrial gene expression (`pct_counts_mt`). Ribosomal (RPS/RPL) and
haemoglobin (HB*) gene flags were also added for reference.

**Cell filtering** applied three thresholds simultaneously:

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Minimum genes | 200 | Remove empty droplets |
| Maximum genes | 2,500 | Remove potential doublets |
| Max mitochondrial % | 5% | Remove dying / low-quality cells |

Raw counts were preserved in `adata.layers["counts"]` before normalisation.
**Normalisation** scaled each cell to 10,000 counts with `sc.pp.normalize_total`,
followed by log(x+1) transformation with `sc.pp.log1p`.

**Highly variable gene (HVG) selection** used `sc.pp.highly_variable_genes` with
`n_top_genes=2000` and `batch_key="sample"` to account for batch differences
between HCC1 and HCC2. The 2,000 HVGs capture ~85% of the total variance while
suppressing noise from lowly expressed genes.

---

## 02 · Clustering (`02_clustering.ipynb`, `scripts/scrna_functions.py`)

**Dimensionality reduction** applied PCA to the HVG-subset expression matrix.
The top 10 principal components were retained based on the elbow of the
variance-ratio scree plot.

**Neighbor graph construction** used `sc.pp.neighbors` with k=15 nearest
neighbors in 10 PC dimensions. UMAP embeddings were computed with
`sc.tl.umap` for visualisation.

**Leiden clustering** was run at four resolutions (0.3, 0.5, 1.0, 2.0) using
the igraph implementation (`flavor="igraph"`). Resolution 0.5 produced 11
clusters and was selected for downstream annotation based on biological
interpretability and alignment with known liver cell-type proportions.

---

## 03 · Cell-type annotation (`03_annotation.ipynb`, `scripts/scrna_functions.py`)

Three automated annotation methods were applied independently and then reconciled:

### CellTypist
`Immune_All_High.pkl` (coarse) and `Immune_All_Low.pkl` (fine) models with
majority voting at the Leiden cluster level. Input: normalised log-counts.

### ScType
Liver-specific marker database (`ScTypeDB_full.xlsx`) sourced from the
[ScType GitHub repository](https://github.com/IanevskiAleksandr/sc-type).
Run via rpy2 in R. The `sctype_score_` function was applied to log-normalised
expression; the highest-scoring cell type per Leiden cluster was recorded.

### SingleR
Human Primary Cell Atlas (HPCA) reference dataset (`celldex::HumanPrimaryCellAtlasData()`).
Spearman correlation scores were computed against all reference profiles; pruned
labels (NA for low-confidence calls) were used.

### Marker gene scoring (manual)
Mean log-normalised expression of canonical marker gene sets per Leiden cluster.
11 cell types were scored using 6–10 markers each (defined in
`scrna_functions.MARKER_SETS`).

### 4-way majority vote
The four evidence sources were combined per cluster:

- CellTypist fine label
- ScType label
- SingleR HPCA label
- Best label by marker score

**ScType received double weight** for parenchymal cell types (Hepatocyte,
Fibroblast, Endothelial) because it is the only tool using a liver-specific
reference database. Final labels were stored in `adata.obs["manual_celltype"]`.

---

## 04 · Differential expression analysis (`04_dea.ipynb`, `scripts/dea_functions.py`)

The Wilcoxon rank-sum test was applied via `sc.tl.rank_genes_groups` comparing
tumor tissue (HCC2) against normal-adjacent tissue (HCC1). DEGs were retained if:

- Adjusted p-value < 0.05 (Benjamini-Hochberg correction)
- |log2 fold change| > 1

This produced **1,178 significant DEGs**: 926 upregulated in tumor, 252 downregulated.
Results exported to `data/processed/dea_results.csv`.

---

## 05 · GSEA (`05_gsea.ipynb`, `scripts/gsea_functions.py`)

A ranked gene list sorted descending by log2FC was prepared from all DEGs
(significant and non-significant) and passed to clusterProfiler in R via rpy2.

**Gene symbol → Entrez ID mapping** used `bitr` from `org.Hs.eg.db`.

**GSEA** was run with `gseGO` and `gseKEGG` using:

| Parameter | Value |
|-----------|-------|
| minGSSize | 15 |
| maxGSSize | 500 |
| pvalueCutoff | 0.05 |
| Seed | 42 |

Four ontologies were analysed: GO Biological Process (BP), GO Molecular Function
(MF), GO Cellular Component (CC), and KEGG. Results exported as CSV tables and
dot/ridge plots. Four HCC-relevant themes were highlighted by pattern-matching
pathway descriptions: lipid metabolism, glycolysis/energy, PI3K-AKT/Wnt, and
immune regulation.

---

## P1 · PPI Network (`P1_ppi_network.ipynb`, `scripts/ppi_functions.py`)

The 1,178 significant DEGs were submitted to the
[STRING API](https://string-db.org/api/json/network) in batches of 500 genes.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| Species | Homo sapiens (taxon 9606) |
| Network type | Functional |
| Minimum combined score | 400 (medium confidence) |

A NetworkX undirected graph was constructed from returned interactions, with
edge weights set to STRING combined scores. Isolated nodes were removed.

**Hub score** = normalised mean of four centrality measures per gene:

| Measure | Function |
|---------|----------|
| Degree centrality | `nx.degree_centrality` |
| Betweenness centrality | `nx.betweenness_centrality(weight="weight")` |
| Closeness centrality | `nx.closeness_centrality` |
| Eigenvector centrality | `nx.eigenvector_centrality(max_iter=500, weight="weight")` |

Each measure was normalised to [0, 1] before averaging. Hub scores are used as
node features in the GNN (notebook P4).

---

## P2 · Survival filter (`P2_survival_filter.ipynb`, `scripts/survival_functions.py`)

Clinical and gene expression data for TCGA-LIHC (~374 HCC patients, primary
tumors only) were downloaded from the
[UCSC Xena public hub](https://xenabrowse.com). If the download fails, the
notebook falls back to realistic simulated data (controlled random seed,
protective effect added for APOE/ALB, risk effect for XIST/FTL).

For each DEG with available expression data, patients were split at the median
expression level. The following tests were run:

- **Kaplan-Meier log-rank test** — `lifelines.statistics.logrank_test`
- **Cox proportional-hazards regression** — `lifelines.CoxPHFitter(penalizer=0.1)`
  on standardised expression

**Filter criteria** (all three must pass):

| Criterion | Threshold |
|-----------|-----------|
| KM log-rank p | < 0.05 |
| Cox p | < 0.05 |
| Hazard ratio | < 0.8 or > 1.2 |

Genes passing all filters are exported as `survival_filtered_genes.csv` and
receive a +0.10 bonus in the DGI composite score (notebook P3).

---

## P3 · Drug–gene interactions (`P3_drug_gene_interactions.ipynb`, `scripts/dgi_functions.py`)

Drug-gene interactions were queried from up to three databases:

| Database | Protocol | Data retrieved |
|----------|----------|----------------|
| [DGIdb](https://dgidb.org) | GraphQL API | Interaction type, directionality, approval, publications |
| [ChEMBL](https://www.ebi.ac.uk/chembl) | REST API | Mechanism of action, max clinical phase |
| [OpenTargets](https://platform.opentargets.org) | GraphQL API | Approval status, phase, mechanism |

A curated fallback dataset (37 literature-based interactions for 16 hub genes,
compiled from Wang et al. 2025) supplements any genes not covered by live APIs.
All sources are selectable independently via boolean flags in the notebook.

**Composite score** (0–1) per drug-gene pair:

```
score = 0.35 × interaction_score_norm
      + 0.20 × publications_norm (capped at 30)
      + 0.20 × clinical_phase / 4
      + 0.15 × approved (binary)
      + 0.10 × hub_score_norm
      + 0.10 × survival_target (binary bonus)
```

Gene-drug pairs were deduplicated (highest interaction score kept per pair)
before scoring.

---

## P4 · GNN (`P4_gnn.ipynb`, `scripts/gnn_functions.py`)

### Graph construction

A heterogeneous bipartite graph was constructed with PyTorch Geometric:

- **Nodes:** genes (features: hub score, survival target flag) and drugs
  (15 binary/numeric features from P3)
- **Edges:** bidirectional drug-gene interactions weighted by composite score
- **Node features** were standardised with `StandardScaler`
- **Split:** 70% train / 15% validation / 15% test (stratified random, seed=42)

### Model architectures

Three architectures were trained and compared:

| Model | Design |
|-------|--------|
| GCN | 2-layer Graph Convolutional Network, mean aggregation, BatchNorm |
| GAT | Graph Attention Network, 4 attention heads, BatchNorm |
| GraphSAGE | Inductive mean-neighbourhood aggregation, BatchNorm |

All models share the same edge prediction head: `concat(embed_src, embed_dst) → Linear(128) → ReLU → Dropout(0.3) → Linear(1) → Sigmoid`.

### Training

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 0.005 |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=15, factor=0.5) |
| Max epochs | 300 |
| Early stopping patience | 40 |
| Gradient clipping | 1.0 |

### Evaluation

The best model (highest test R²) was selected and used to score all drug-gene
pairs in the graph. Drug candidates were ranked by predicted score descending.

### Google Colab compatibility

Notebook P4 auto-detects the Colab runtime, installs PyTorch Geometric with
the correct CUDA wheel URL, and uses Colab session paths for all file I/O.
No manual configuration is required beyond uploading `dgi_edges_gnn.csv`.

---

## Software versions

| Package | Version |
|---------|---------|
| Python | 3.12 |
| scanpy | ≥1.10 |
| anndata | ≥0.10 |
| celltypist | ≥1.6 |
| rpy2 | ≥3.5 |
| anndata2ri | ≥1.3 |
| torch | ≥2.2 |
| torch-geometric | ≥2.5 |
| lifelines | ≥0.28 |
| networkx | ≥3.2 |
| scikit-learn | ≥1.4 |
| R | ≥4.3 |
| Seurat | ≥5.0 |
| SingleR | ≥2.0 |
| celldex | ≥1.0 |
| clusterProfiler | ≥4.0 |
| org.Hs.eg.db | ≥3.17 |
