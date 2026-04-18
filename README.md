# HCC Drug Discovery Pipeline

Integrating single-cell RNA sequencing and graph neural networks for
multi-targeted drug design in hepatocellular carcinoma.

**Based on:** Wang et al. (2025) — *Integrating single-cell RNA sequencing and
artificial intelligence for multitargeted drug design for combating resistance
in liver cancer.* npj Precision Oncology 9:309.
[doi:10.1038/s41698-025-00952-3](https://doi.org/10.1038/s41698-025-00952-3)

---

## Overview

This repository reproduces and extends the computational pipeline from the paper
above, covering the full journey from raw scRNA-seq data to ranked drug candidates.
It combines single-cell transcriptomics (Scanpy, CellTypist, SingleR, ScType)
with graph-based deep learning (PyTorch Geometric) to identify and prioritise
therapeutic targets in hepatocellular carcinoma (HCC).

```
Raw MTX files (GEO: GSE166635)
        │
        ▼
 01 · Preprocessing & QC          scrna_functions.py
        │
        ▼
 02 · Clustering                   scrna_functions.py
        │
        ▼
 03 · Cell-type annotation         scrna_functions.py
        │   CellTypist · ScType (R) · SingleR (R) · 4-way majority vote
        ▼
 04 · Differential expression      dea_functions.py
        │   Wilcoxon rank-sum · volcano plot
        ▼
 05 · GSEA & pathway networks      gsea_functions.py
        │   clusterProfiler (R) · GO-BP/MF/CC · KEGG
        ▼
 06 · Macrophage sub-cluster       (original contribution beyond the paper)
        │   M1 / M2 / Kupffer / TAM module scoring
        ▼
 P1 · PPI network & hub genes      ppi_functions.py
        │   STRING API · NetworkX · composite centrality score
        ▼
 P2 · Survival filter              survival_functions.py
        │   Kaplan–Meier · Cox regression · TCGA-LIHC (n≈374)
        ▼
 P3 · Drug–gene interactions       dgi_functions.py
        │   DGIdb · ChEMBL · OpenTargets · curated fallback
        ▼
 P4 · GNN training & ranking       gnn_functions.py
             GCN · GAT · GraphSAGE (PyTorch Geometric)
             Google Colab / GPU ready
```

---

## Repository structure

```
HCC_DD/
├── notebooks/
│   ├── 01_preprocessing.ipynb          QC, normalisation, HVG selection
│   ├── 02_clustering.ipynb             PCA, UMAP, Leiden clustering
│   ├── 03_annotation.ipynb             CellTypist + ScType + SingleR + vote
│   ├── 04_dea.ipynb                    Wilcoxon DEA + volcano plot
│   ├── 05_gsea.ipynb                   GSEA via clusterProfiler (R)
│   ├── 06_immune_infiltration.ipynb    Macrophage sub-cluster (original)
│   ├── P1_ppi_network.ipynb            STRING PPI + hub gene ranking
│   ├── P2_survival_filter.ipynb        KM + Cox survival analysis
│   ├── P3_drug_gene_interactions.ipynb DGIdb / ChEMBL / OpenTargets
│   └── P4_gnn.ipynb                    GNN training + drug ranking
│                                       (Colab / GPU compatible)
│
├── scripts/
│   ├── scrna_functions.py              scRNA-seq utilities (notebooks 01–03)
│   ├── dea_functions.py                DEA utilities (notebook 04)
│   ├── gsea_functions.py               GSEA utilities (notebook 05)
│   ├── ppi_functions.py                PPI utilities (notebook P1)
│   ├── survival_functions.py           Survival utilities (notebook P2)
│   ├── dgi_functions.py                DGI utilities (notebook P3)
│   ├── gnn_functions.py                GNN models + training (notebook P4)
│   ├── data_download.py                Downloads GSE166635, writes paths.py
│   └── utils/
│       ├── __init__.py
│       ├── graph_utils.py              PPI + GNN graph construction
│       ├── plot_utils.py               All matplotlib figure functions
│       └── api_clients.py              DGIdb / ChEMBL / OpenTargets clients
│
├── env/
│   ├── environment.yml                 Conda environment (CPU)
│   ├── environment_gpu.yml             Conda environment (CUDA 12.1)
│   ├── requirements.txt                pip fallback
│   ├── setup_env.sh                    Automated setup script
│   ├── r_packages.R                    R package installer
│   └── .python-version                 Python 3.12 pin
│
├── results/
│   ├── figures/                        PNG figures
│   ├── tables/                         CSV outputs
│   └── reports/                        Text summaries
│
├── models/                             GNN weights and scalers (.pt, .pkl)
│
├── docs/
│   ├── METHODS.md                      Detailed computational methods
│   └── data_sources.md                 Data licences and download URLs
│
├── .gitignore
└── README.md
```

> **Note:** `data/` and `paths.py` are excluded from git.
> They are created locally by running `python scripts/data_download.py`.

---

## Quick start

### 1 — Clone the repo

```bash
git clone https://github.com/fshokor/HCC_DD.git
cd HCC_DD
```

### 2 — Set up the environment

```bash
chmod +x env/setup_env.sh

# CPU-only (works on any machine, sufficient for this project)
./env/setup_env.sh

# GPU support (CUDA 12.1, for notebook P4)
./env/setup_env.sh --gpu

conda activate hcc_drug_discovery
```

#### R packages (required for notebooks 03 and 05)

```bash
Rscript env/r_packages.R
```

### 3 — Download the data

```bash
python scripts/data_download.py
```

Downloads GSE166635 (~204 MB) from NCBI GEO, extracts HCC1 and HCC2 MTX
triplets, and writes `paths.py` at the repo root. All notebooks find their
files automatically from `paths.py`, regardless of where the repo was cloned.

### 4 — Run the pipeline

**Step by step** (recommended):

```bash
jupyter lab
# Run in order: 01 → 02 → 03 → 04 → 05 → 06 → P1 → P2 → P3 → P4
```

**P4 on Google Colab with GPU:**

1. Upload `results/tables/dgi_edges_gnn.csv` to the Colab session (Files → Upload)
2. Open `notebooks/P4_gnn.ipynb` in Colab
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells — Colab mode is auto-detected, PyG installs automatically

---

## Notebook guide

| Notebook | Input | Output | Logic in |
|----------|-------|--------|----------|
| `01_preprocessing` | `data/raw/HCC1,HCC2/` | `adata_processed.h5ad` | `scrna_functions.py` |
| `02_clustering` | `adata_processed.h5ad` | `adata_clustered.h5ad` | `scrna_functions.py` |
| `03_annotation` | `adata_clustered.h5ad` | `adata_annotated.h5ad` | `scrna_functions.py` |
| `04_dea` | `adata_annotated.h5ad` | `dea_results.csv` | `dea_functions.py` |
| `05_gsea` | `dea_results.csv` | `gsea_*.csv`, figures | `gsea_functions.py` |
| `06_immune_infiltration` | `adata_annotated.h5ad` | macrophage figures | — |
| `P1_ppi_network` | `dea_results.csv` | `hub_genes.csv` | `ppi_functions.py` |
| `P2_survival_filter` | `hub_genes.csv` | `survival_filtered_genes.csv` | `survival_functions.py` |
| `P3_drug_gene_interactions` | `hub_genes.csv` | `dgi_edges_gnn.csv` | `dgi_functions.py` |
| `P4_gnn` | `dgi_edges_gnn.csv` | `gnn_drug_ranking.csv`, `gcn_best.pt` | `gnn_functions.py` |

Each notebook imports its logic from the corresponding `scripts/*_functions.py`
file. Notebooks contain only configuration and single-line function calls,
keeping them readable and easy to modify.

---

## Database selection (notebook P3)

Notebook P3 lets you choose which drug databases to query. Edit the configuration
cell before running:

```python
USE_DGIDB       = True    # DGIdb GraphQL API
USE_CHEMBL      = True    # ChEMBL REST API
USE_OPENTARGETS = True    # OpenTargets GraphQL API
USE_CURATED     = True    # Built-in curated fallback (fills gaps automatically)
```

The curated fallback activates automatically when no live API returns results
for a gene, or when all live sources are disabled.

---

## Key outputs

| File | Description |
|------|-------------|
| `data/processed/dea_results.csv` | 1,178 significant DEGs (padj<0.05, \|log2FC\|>1) |
| `results/tables/hub_genes.csv` | Hub genes ranked by composite centrality score |
| `results/tables/survival_filtered_genes.csv` | Survival-significant genes → GNN priority targets |
| `results/tables/gnn_drug_ranking.csv` | All drugs ranked by GNN-predicted interaction score |
| `results/tables/gnn_node_embeddings.csv` | 64-dim learned node representations |
| `results/figures/ppi_network.png` | PPI network coloured by regulation |
| `results/figures/km_plots.png` | Kaplan–Meier survival grid |
| `results/figures/gnn_drug_ranking.png` | Top 25 drug candidates chart |
| `models/gcn_best.pt` | Best trained GNN weights |
| `models/feature_scaler.pkl` | Feature scaler for inference |

---

## Methods summary

| Step | Tool | Key parameters |
|------|------|----------------|
| QC & preprocessing | Scanpy ≥1.10 | min_genes=200, max_genes=2500, max_mt=5% |
| HVG selection | Scanpy | n_top_genes=2000, batch_key="sample" |
| Clustering | Leiden (igraph) | resolution=0.5, 11 clusters |
| Cell-type annotation | CellTypist + ScType + SingleR | 4-way majority vote; ScType double-weighted for parenchymal types |
| DEA | Wilcoxon rank-sum | padj<0.05, \|log2FC\|>1 |
| GSEA | clusterProfiler (R) | GO-BP/MF/CC + KEGG, minGSSize=15, p<0.05 |
| PPI | STRING API | score≥400, functional network |
| Survival filter | lifelines KM + Cox | TCGA-LIHC (n≈374), KM p<0.05, Cox p<0.05, \|HR−1\|>0.2 |
| Drug interactions | DGIdb + ChEMBL + OpenTargets | Composite score: interaction(35%) + publications(20%) + phase(20%) + approval(15%) + hub(10%) |
| GNN | PyTorch Geometric | GCN / GAT / GraphSAGE compared; best model by test R² |

Full methods → [`docs/METHODS.md`](docs/METHODS.md)  
Data sources and licences → [`docs/data_sources.md`](docs/data_sources.md)

---

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12 | managed by conda |
| R | ≥4.3 | annotation (notebook 03) and GSEA (notebook 05) only |
| GPU | optional | P4 trains in <5 min on CPU; use Colab T4 for faster runs |

See [`env/environment.yml`](env/environment.yml) for the full Python package list.
R packages are installed separately with `Rscript env/r_packages.R`.

---

## Original contributions

This repository goes beyond the Wang et al. (2025) paper in four ways:

**Macrophage sub-cluster analysis** (`06_immune_infiltration.ipynb`). Re-clusters
tumor-associated macrophages and scores M1, M2, Kupffer, and TAM phenotypes,
comparing normal vs tumor macrophage composition and identifying phenotype-specific
DEGs not reported in the paper.

**Modular architecture.** Each analysis step is implemented as a function script
(`scripts/*_functions.py`) imported by a thin notebook. All functions are
independently testable and reusable outside of Jupyter.

**Configurable database selection.** Notebook P3 lets users enable or disable
individual drug databases (DGIdb, ChEMBL, OpenTargets) rather than querying
all three unconditionally, making it easier to work offline or reproduce results
from a single source.

**Google Colab compatibility.** Notebook P4 auto-detects the Colab environment,
installs the correct PyG wheels for the available CUDA version, and adapts all
file paths for the Colab session filesystem — no manual setup required.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{wang2025hcc,
  title   = {Integrating single-cell RNA sequencing and artificial intelligence
             for multitargeted drug design for combating resistance in liver cancer},
  author  = {Wang, Houhong and Yang, Youyuan and Zhang, Junfeng and others},
  journal = {npj Precision Oncology},
  volume  = {9},
  pages   = {309},
  year    = {2025},
  doi     = {10.1038/s41698-025-00952-3}
}
```

---

## Licence

MIT — see [LICENSE](LICENSE) for details.
