# HCC Drug Discovery Pipeline

**Integrating single-cell RNA sequencing and graph neural networks for
multi-targeted drug design in hepatocellular carcinoma.**

> Based on: Wang et al. (2025) — *Integrating single-cell RNA sequencing and
> artificial intelligence for multitargeted drug design for combating resistance
> in liver cancer.* npj Precision Oncology 9:309.
> [doi:10.1038/s41698-025-00952-3](https://doi.org/10.1038/s41698-025-00952-3)

---

## Overview

Hepatocellular carcinoma (HCC) is the most common primary liver cancer,
accounting for 75–85% of all liver cancers worldwide. It is typically diagnosed
at an advanced stage, carries a poor prognosis, and rapidly develops resistance
to conventional therapies. A key challenge is that HCC is highly heterogeneous
at the cellular and molecular level — and standard bulk RNA-sequencing averages
over this diversity, masking critical tumour-immune interactions.

This pipeline addresses that challenge by combining **single-cell RNA sequencing
(scRNA-seq)** with **graph neural network (GNN)-based drug discovery** to
systematically:

1. **Map the HCC tumour microenvironment** at single-cell resolution — identifying
   distinct cell populations (hepatocytes, macrophages, T cells, fibroblasts,
   endothelial cells) and their transcriptional states in tumour vs. normal-adjacent
   tissue.
2. **Identify prognostic biomarkers** through differential expression analysis and
   survival correlation — flagging genes such as *APOE* and *ALB* (protective) and
   *XIST* and *FTL* (risk-associated).
3. **Prioritise therapeutic targets** via protein–protein interaction network
   analysis and survival filtering of hub genes.
4. **Rank drug repurposing candidates** using a GNN trained on a bipartite
   drug–gene interaction graph, producing a scored list of approved compounds
   with predicted therapeutic relevance to HCC.

Each step generates a **self-contained HTML report** that biology experts can
read directly — no coding knowledge required.

---

## Pipeline

![Pipeline diagram](docs/pipeline.svg)

The pipeline runs in three main notebooks, each producing a human-readable
HTML report alongside its analytical outputs:

| Notebook | What it does |
|----------|-------------|
| **01 · scRNA-seq Analysis** | QC, normalisation, UMAP clustering, 4-way cell-type annotation, DEA, GSEA |
| **02 · Target Prioritisation** | PPI hub gene network (STRING), survival filter (TCGA-LIHC), drug–gene interactions (DGIdb, ChEMBL, OpenTargets) |
| **03 · GNN Drug Ranking** | Trains GCN / GAT / GraphSAGE on the interaction graph, re-scores all drug–gene pairs, produces a ranked repurposing list |

---

## Quick start

### 1 — Clone

```bash
git clone https://github.com/fshokor/HCC_DD.git
cd HCC_DD
```

### 2 — Environment

```bash
# CPU (sufficient for notebooks 01–02)
conda env create -f env/environment.yml
conda activate hcc_drug_discovery

# GPU (CUDA 12.1, recommended for notebook 03)
conda env create -f env/environment_gpu.yml
conda activate hcc_drug_discovery
```

**R packages** (required for cell-type annotation and GSEA in notebook 01):

```bash
Rscript env/r_packages.R
```

**Key dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| scanpy | ≥ 1.9 | scRNA-seq analysis |
| celltypist | ≥ 1.6 | Automated cell-type annotation |
| torch | ≥ 2.0 | GNN training |
| torch-geometric | ≥ 2.4 | Graph neural networks |
| rpy2 | ≥ 3.5 | Python ↔ R bridge |
| networkx | ≥ 3.0 | PPI network construction |
| numpy | < 2.0 | (pinned for torch-geometric compatibility) |

### 3 — Download data

```bash
python scripts/data_download.py
```

Downloads GSE166635 (~204 MB) from NCBI GEO, extracts the HCC1 (normal-adjacent)
and HCC2 (tumour) 10x Genomics MTX files, and writes `paths.py` at the repo root.
All notebooks import paths from this file automatically.

### 4 — Test the installation

```bash
python scripts/run_pipeline_test.py
```

Runs the full scRNA-seq pipeline (QC → clustering → annotation) using Python-only
methods — no R required. Exits with code 0 on success and produces four output
files in `results/`: a UMAP figure, cluster summary CSV, and annotated `.h5ad`.
Use this to verify your environment before running the full notebooks.

### 5 — Run the pipeline

Open JupyterLab and run the notebooks in order:

```bash
jupyter lab
```

```
01_scrna_analysis.ipynb   →   02_target_prioritisation.ipynb   →   03_gnn_drug_ranking.ipynb
```

> **Notebook 03 on Google Colab (GPU):**
> 1. Upload `results/tables/dgi_edges_gnn.csv` via Files → Upload
> 2. Open `notebooks/03_gnn_drug_ranking.ipynb` in Colab
> 3. Runtime → Change runtime type → **T4 GPU**
> 4. Run all — Colab mode is auto-detected, PyG installs automatically

---

## Notebook guide

| Notebook | Input | Key outputs | Logic script |
|----------|-------|-------------|-------------|
| `01_scrna_analysis.ipynb` | `data/raw/HCC1,HCC2/` | `adata_annotated.h5ad` · `dea_results.csv` · `gsea_*.csv` · figures · HTML report | `scrna_functions.py` · `dea_functions.py` · `gsea_functions.py` |
| `02_target_prioritisation.ipynb` | `dea_results.csv` | `hub_genes.csv` · `survival_filtered_genes.csv` · `dgi_edges_gnn.csv` · `dgi_summary_dashboard.png` · HTML report | `ppi_functions.py` · `survival_functions.py` · `dgi_functions.py` |
| `03_gnn_drug_ranking.ipynb` | `dgi_edges_gnn.csv` | `gnn_drug_ranking.csv` · `gcn_best.pt` · `drug_gene_network.png` · HTML report | `gnn_functions.py` |

Each notebook contains only configuration and single-line function calls.
All analysis logic lives in the corresponding `scripts/*_functions.py` file,
making it independently testable and reusable.

---

## Key outputs

After running all three notebooks, the following files are produced in `results/`:

**Processed data**

| File | Description |
|------|-------------|
| `tables/dea_results.csv` | 1,178+ differentially expressed genes with log₂FC, adjusted p-value, and direction |
| `tables/hub_genes.csv` | PPI hub genes ranked by composite centrality score |
| `tables/survival_filtered_genes.csv` | Hub genes with significant survival association (Kaplan–Meier + Cox) |
| `tables/dgi_edges_gnn.csv` | Scored drug–gene interaction graph (GNN-ready edge list) |
| `tables/gnn_drug_ranking.csv` | Final ranked drug candidates with GNN score, original score, and delta |
| `models/gcn_best.pt` | Trained GraphSAGE model weights |

**Figures**

| File | Description |
|------|-------------|
| `figures/umap_annotation.png` | UMAP coloured by majority-vote cell-type annotation |
| `figures/umap_leiden.png` | UMAP coloured by Leiden cluster + sample origin |
| `figures/volcano_plot.png` | DEA volcano plot — tumour vs. normal-adjacent |
| `figures/dgi_summary_dashboard.png` | 5-panel drug–gene interaction summary |
| `figures/dgi_panel_*.png` | Individual dashboard panels (A–E) saved separately |
| `figures/gnn_drug_ranking.png` | Top 25 drug candidates bar chart |
| `figures/drug_gene_network.png` | Bipartite drug–gene interaction network |
| `figures/gnn_training_curves.png` | Training loss curves for all three GNN models |

**Reports** (open in any browser)

| File | Audience |
|------|----------|
| `reports/01_scrna_analysis_report.html` | Biology experts reviewing cell-type composition, QC metrics, DEGs, and pathways |
| `reports/02_target_prioritisation_report.html` | Reviewing hub genes, survival associations, and drug candidates |
| `reports/03_gnn_drug_ranking_report.html` | Reviewing model performance and the final ranked repurposing list |

---

## Original contributions

This repository adds the following beyond the Wang et al. (2025) paper:

- **Multi-source cell-type annotation.** Four evidence sources (CellTypist,
  ScType with liver-specific markers, SingleR/HPCA, and curated marker scoring)
  combined via majority vote per cluster — more robust than any single method.
- **Modular architecture.** Each step is separated into a `*_functions.py` script
  imported by a thin notebook. All logic is independently testable.
- **Automated HTML reports.** Running the final cell of any notebook produces a
  complete, self-contained HTML summary for biology stakeholders.
- **Three GNN architectures compared.** GCN, GAT, and GraphSAGE are all trained
  and evaluated; the best model (by R²) is used for ranking.
- **Drug–gene network visualisation.** A bipartite network plot shows the
  structural relationships between top-ranked drugs and their target hub genes.
- **Per-panel figure saving.** The DGI dashboard saves each panel individually
  in addition to the combined figure, for use in presentations and reports.
- **Google Colab compatibility.** Notebook 03 auto-detects the Colab environment
  and installs the correct PyG wheels for GPU-accelerated training.
- **Pipeline test script.** `run_pipeline_test.py` validates the full scRNA-seq
  pipeline in a Python-only mode (no R required) for CI and environment checks.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{wang2025hcc,
  title     = {Integrating single-cell {RNA} sequencing and artificial intelligence
               for multitargeted drug design for combating resistance in liver cancer},
  author    = {Wang, Houhong and Yang, Youyuan and Zhang, Junfeng and
               Chen, Wenli and Dai, Jingyou and Li, Changquan and Li, Qing},
  journal   = {npj Precision Oncology},
  volume    = {9},
  pages     = {309},
  year      = {2025},
  doi       = {10.1038/s41698-025-00952-3}
}
```

---

## Licence

MIT — see [LICENSE](LICENSE) for details.
