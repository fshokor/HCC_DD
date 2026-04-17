# HCC Drug Discovery Pipeline

Integrating single-cell RNA sequencing and graph neural networks for
multi-targeted drug design in hepatocellular carcinoma.

Based on: **Wang et al. (2025)** — *Integrating single-cell RNA sequencing and
artificial intelligence for multitargeted drug design for combating resistance
in liver cancer.* npj Precision Oncology 9:309.
[doi:10.1038/s41698-025-00952-3](https://doi.org/10.1038/s41698-025-00952-3)

---

## Overview

This repository reproduces and extends the computational pipeline from the paper
above, covering the full journey from raw scRNA-seq data to ranked drug candidates.

```
Raw MTX files (GEO: GSE166635)
        │
        ▼
 Preprocessing & QC          [notebook 01]
        │
        ▼
 Clustering & annotation      [notebook 02-03]   ← CellTypist · ScType · SingleR
        │
        ▼
 Differential expression      [notebook 04]      ← Wilcoxon / Scanpy
        │
        ▼
 GSEA & pathway networks      [notebook 05]      ← DisGeNET · Reactome · KEGG
        │
        ▼
 PPI network & hub genes      [notebook 09]      ← STRING API · NetworkX
        │
        ▼
 Survival filter              [notebook 11]      ← Kaplan–Meier · Cox · TCGA-LIHC
        │
        ▼
 Drug–gene interactions       [notebook 12]      ← DGIdb · ChEMBL · OpenTargets
        │
        ▼
 GNN training & ranking       [notebooks 13-14]  ← GCN · GAT · GraphSAGE (PyG)
        │
        ▼
 Macrophage sub-cluster analysis [notebook 06]   ← original contribution
```

---

## Repository structure

```
hcc-drug-discovery/
├── data/
│   ├── raw/                  ← original MTX files from GEO GSE166635
│   └── processed/            ← intermediate files (dea_results.csv, hub_genes.csv …)
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_annotation.ipynb
│   ├── 04_dea.ipynb
│   ├── 05_gsea.ipynb
│   ├── 06_immune_infiltration.ipynb
│   ├── 09_ppi.ipynb
│   ├── 11_survival.ipynb
│   ├── 12_dgi.ipynb
│   ├── 13_gnn_build.ipynb
│   ├── 14_gnn_train.ipynb
│   └── pipeline.ipynb        ← all-in-one notebook (steps 09–14)
│
├── scripts/
│   ├── ppi_analysis.py
│   ├── survival_analysis.py
│   ├── drug_gene_interaction.py
│   ├── gnn_drug_discovery.py
│   └── utils/
│       ├── __init__.py
│       ├── graph_utils.py    ← PPI graph + GNN graph construction
│       ├── plot_utils.py     ← all matplotlib figure functions
│       └── api_clients.py    ← DGIdb / ChEMBL / OpenTargets clients
│
├── results/
│   ├── figures/              ← PNG outputs (network, KM plots, rankings …)
│   ├── tables/               ← CSV outputs (drug_ranking, hub_genes …)
│   └── reports/              ← text summaries
│
├── models/                   ← saved GNN weights (.pt) and scalers (.pkl)
│
├── env/
│   ├── environment.yml       ← conda environment (Python packages)
│   ├── requirements.txt      ← pip fallback
│   ├── setup_env.sh          ← automated setup script
│   └── r_packages.R          ← R package installer (Seurat · SingleR · ScType)
│
└── docs/
    ├── METHODS.md
    └── data_sources.md
```

---

## Quick start

### 1 — Get the data

Download the scRNA-seq data from GEO:

```bash
# GSE166635 — HCC tumor and adjacent normal tissue (10x Genomics)
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166635

# Download GSE166635_RAW.tar (204 MB) and extract into:
mkdir -p data/raw/HCC1 data/raw/HCC2
# HCC1 = tumor-adjacent normal, HCC2 = tumor tissue
```

Expected structure after extraction:

```
data/raw/
├── HCC1/
│   ├── barcodes.tsv.gz
│   ├── features.tsv.gz
│   └── matrix.mtx.gz
└── HCC2/
    ├── barcodes.tsv.gz
    ├── features.tsv.gz
    └── matrix.mtx.gz
```

### 2 — Set up the conda environment

```bash
chmod +x env/setup_env.sh

# CPU-only (works on any machine — sufficient for this project)
./env/setup_env.sh

# GPU support (CUDA 12.1)
./env/setup_env.sh --gpu

# Activate
conda activate hcc_drug_discovery
```

#### R packages (required for annotation notebooks 02–03)

```bash
Rscript env/r_packages.R
```

### 3 — Run the pipeline

**Option A — step by step** (recommended for understanding):

```bash
jupyter lab
# Open notebooks/ and run in order: 01 → 02 → 03 → 04 → 05 → 06 → 09 → 11 → 12 → 13 → 14
```

**Option B — all-in-one** (steps 09–14 only):

```bash
jupyter lab notebooks/pipeline.ipynb
```

**Option C — command line** (steps 09–14 as standalone scripts):

```bash
cd scripts/
python ppi_analysis.py          # step 09 — produces hub_genes.csv
python survival_analysis.py     # step 11 — produces survival_filtered_genes.csv
python drug_gene_interaction.py # step 12 — produces dgi_edges_gnn.csv
python gnn_drug_discovery.py    # steps 13-14 — produces gnn_drug_ranking.csv
```

---

## Key outputs

| File | Description |
|------|-------------|
| `data/processed/adata_processed.h5ad` | Preprocessed AnnData object |
| `data/processed/dea_results.csv` | 1,178 differentially expressed genes |
| `results/tables/hub_genes.csv` | Hub genes ranked by composite centrality |
| `results/tables/survival_filtered_genes.csv` | Genes with significant survival association |
| `results/tables/gnn_drug_ranking.csv` | All drugs ranked by GNN-predicted score |
| `results/figures/ppi_network.png` | PPI network coloured by regulation |
| `results/figures/km_plots.png` | Kaplan–Meier survival grid |
| `results/figures/gnn_drug_ranking.png` | Top 25 drug candidates |
| `models/gcn_best.pt` | Best trained GNN weights |

---

## Methods summary

| Step | Tool | Key parameters |
|------|------|----------------|
| QC & preprocessing | Scanpy | min_genes=200, max_genes=2500, max_mt=5% |
| Clustering | Leiden (igraph) | resolution=0.5 |
| Cell-type annotation | CellTypist + ScType + SingleR | 4-way majority vote |
| DEA | Wilcoxon rank-sum | padj<0.05, |log2FC|>1 |
| GSEA | clusterProfiler (R) | DisGeNET · Reactome · KEGG |
| PPI | STRING API | score≥400, functional network |
| Survival | lifelines (KM + Cox) | TCGA-LIHC (n=374) |
| Drug–gene interactions | DGIdb + ChEMBL + OpenTargets | composite scoring |
| GNN | PyTorch Geometric | GCN / GAT / GraphSAGE compared |

Full details in [`docs/METHODS.md`](docs/METHODS.md).

---

## Requirements

- Python 3.12
- R 4.3+ (for annotation steps only)
- No GPU required — all models train in < 5 min on CPU

See [`env/environment.yml`](env/environment.yml) for the full package list.

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

## License

MIT License — see [LICENSE](LICENSE) for details.
