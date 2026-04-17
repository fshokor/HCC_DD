"""
PPI Network Analysis for HCC scRNA-seq Study
=============================================
Input  : DEA results CSV/TSV with columns: gene, log2FC, adj_pvalue
Outputs:
  1. ppi_network.png          — publication-ready network figure
  2. hub_genes.csv            — ranked hub gene list (for GNN input)
  3. ppi_edges_cytoscape.csv  — edge list importable into Cytoscape

Requirements:
    pip install pandas numpy requests networkx matplotlib seaborn
"""

# ── Import shared paths (works from any working directory) ────────────────────
import sys as _sys
from pathlib import Path as _Path

def _find_repo_root(start):
    for p in [start, *start.parents]:
        if (p / "paths.py").exists():
            return p
    return start.parent   # fallback: assume script is in scripts/

_repo = _find_repo_root(_Path(__file__).resolve().parent)
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))

try:
    from paths import REPO_ROOT, PROC_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR
    _paths_loaded = True
except ImportError:
    _paths_loaded = False
    REPO_ROOT    = _Path(__file__).resolve().parent.parent
    PROC_DIR     = REPO_ROOT / "data" / "processed"
    FIGURES_DIR  = REPO_ROOT / "results" / "figures"
    TABLES_DIR   = REPO_ROOT / "results" / "tables"
    MODELS_DIR   = REPO_ROOT / "models"
    for _d in [PROC_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        _d.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ── Shared utilities (import from utils/ when running inside the repo) ────────
# from utils import build_ppi_graph, compute_hub_scores, plot_ppi_network
# Uncomment the line above and remove duplicate function definitions
# if you have cloned the full hcc-drug-discovery repo.
# ─────────────────────────────────────────────────────────────────────────────

import time
import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ─────────────────────────────────────────────
# 0. CONFIGURATION — edit these paths / thresholds
# ─────────────────────────────────────────────

DEA_FILE       = "dea_results.csv"   # your DEA output file
GENE_COL       = "gene"              # column name for gene symbols
LOG2FC_COL     = "log2FC"            # column name for log2 fold change
PADJ_COL       = "adj_pvalue"        # column name for adjusted p-value

LOG2FC_THRESH  = 1.0                 # |log2FC| cutoff
PADJ_THRESH    = 0.05                # adjusted p-value cutoff

STRING_SCORE   = 400                 # STRING interaction confidence (0–1000)
                                     # 400 = medium, 700 = high, 900 = very high
TOP_HUB_N      = 20                  # how many hub genes to report

OUT_DIR = TABLES_DIR.parent / "ppi_output"   # or set to any Path  # output folder
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & FILTER DEA RESULTS
# ─────────────────────────────────────────────

print("── Step 1: Loading DEA results ──")

sep = "\t" if DEA_FILE.endswith(".tsv") else ","
dea = pd.read_csv(DEA_FILE, sep=sep)

# Rename to standard column names if needed
dea = dea.rename(columns={
    GENE_COL:   "gene",
    LOG2FC_COL: "log2FC",
    PADJ_COL:   "adj_pvalue",
})

# Filter to significant DEGs
sig = dea[
    (dea["adj_pvalue"] < PADJ_THRESH) &
    (dea["log2FC"].abs() >= LOG2FC_THRESH)
].copy()

sig["regulation"] = np.where(sig["log2FC"] > 0, "up", "down")

print(f"  Total DEGs after filtering : {len(sig)}")
print(f"  Upregulated                : {(sig.regulation == 'up').sum()}")
print(f"  Downregulated              : {(sig.regulation == 'down').sum()}")

gene_list = sig["gene"].dropna().unique().tolist()
print(f"  Unique genes sent to STRING: {len(gene_list)}")

# ─────────────────────────────────────────────
# 2. QUERY STRING DATABASE
# ─────────────────────────────────────────────

print("\n── Step 2: Querying STRING API ──")

STRING_URL = "https://string-db.org/api/json/network"

# STRING accepts up to ~2000 genes per request
# Split into batches of 500 if needed
BATCH_SIZE = 500
all_edges = []

for i in range(0, len(gene_list), BATCH_SIZE):
    batch = gene_list[i : i + BATCH_SIZE]
    print(f"  Querying batch {i // BATCH_SIZE + 1} ({len(batch)} genes)...")

    params = {
        "identifiers"       : "\r".join(batch),
        "species"           : 9606,           # Homo sapiens
        "required_score"    : STRING_SCORE,
        "network_type"      : "functional",   # or "physical"
        "caller_identity"   : "hcc_ppi_script",
    }

    try:
        resp = requests.post(STRING_URL, data=params, timeout=60)
        resp.raise_for_status()
        edges = resp.json()
        all_edges.extend(edges)
        print(f"    → {len(edges)} interactions returned")
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Request failed: {e}")
        print("    Check your internet connection or try reducing batch size.")
        raise

    time.sleep(1)  # be polite to the API

if not all_edges:
    raise ValueError(
        "No interactions returned from STRING. "
        "Try lowering STRING_SCORE or checking gene names."
    )

# Build edge dataframe
edges_df = pd.DataFrame(all_edges)

# Keep only relevant columns (STRING field names vary slightly by version)
keep_cols = {
    "preferredName_A": "gene_A",
    "preferredName_B": "gene_B",
    "score":           "combined_score",
}
edges_df = edges_df.rename(columns=keep_cols)[list(keep_cols.values())]

# Remove self-loops and duplicates
edges_df = edges_df[edges_df["gene_A"] != edges_df["gene_B"]]
edges_df["pair"] = edges_df.apply(
    lambda r: tuple(sorted([r["gene_A"], r["gene_B"]])), axis=1
)
edges_df = edges_df.drop_duplicates("pair").drop(columns="pair")
edges_df["combined_score"] = pd.to_numeric(
    edges_df["combined_score"], errors="coerce"
)

print(f"\n  Total unique interactions: {len(edges_df)}")

# ─────────────────────────────────────────────
# 3. BUILD NETWORKX GRAPH
# ─────────────────────────────────────────────

print("\n── Step 3: Building PPI graph ──")

G = nx.Graph()

# Add nodes with DEA attributes
for _, row in sig.iterrows():
    G.add_node(
        row["gene"],
        log2FC=row["log2FC"],
        adj_pvalue=row["adj_pvalue"],
        regulation=row["regulation"],
    )

# Add edges
for _, row in edges_df.iterrows():
    if row["gene_A"] in G and row["gene_B"] in G:
        G.add_edge(
            row["gene_A"],
            row["gene_B"],
            weight=float(row["combined_score"]),
        )

# Remove isolated nodes (genes with no interactions)
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

print(f"  Nodes (genes with interactions): {G.number_of_nodes()}")
print(f"  Edges (interactions)           : {G.number_of_edges()}")
print(f"  Isolated nodes removed         : {len(isolates)}")

# ─────────────────────────────────────────────
# 4. COMPUTE HUB GENE SCORES
# ─────────────────────────────────────────────

print("\n── Step 4: Computing hub gene centrality ──")

# Four centrality measures used in bioinformatics hub detection
degree_c     = nx.degree_centrality(G)
betweenness_c = nx.betweenness_centrality(G, weight="weight")
closeness_c  = nx.closeness_centrality(G)
eigenvector_c = nx.eigenvector_centrality(G, max_iter=500, weight="weight")

hub_df = pd.DataFrame({
    "gene"              : list(G.nodes()),
    "degree"            : [G.degree(n) for n in G.nodes()],
    "degree_centrality" : [degree_c[n] for n in G.nodes()],
    "betweenness"       : [betweenness_c[n] for n in G.nodes()],
    "closeness"         : [closeness_c[n] for n in G.nodes()],
    "eigenvector"       : [eigenvector_c[n] for n in G.nodes()],
})

# Composite hub score: average of normalised centralities
for col in ["degree_centrality", "betweenness", "closeness", "eigenvector"]:
    hub_df[f"{col}_norm"] = (
        hub_df[col] - hub_df[col].min()
    ) / (hub_df[col].max() - hub_df[col].min() + 1e-9)

hub_df["hub_score"] = hub_df[
    [c for c in hub_df.columns if c.endswith("_norm")]
].mean(axis=1)

# Merge DEA info back in
hub_df = hub_df.merge(
    sig[["gene", "log2FC", "adj_pvalue", "regulation"]],
    on="gene", how="left"
)
hub_df = hub_df.sort_values("hub_score", ascending=False).reset_index(drop=True)

top_hubs = hub_df.head(TOP_HUB_N)
print(f"\n  Top {TOP_HUB_N} hub genes:")
print(top_hubs[["gene", "degree", "hub_score", "log2FC", "regulation"]]
      .to_string(index=False))

# ─────────────────────────────────────────────
# 5. VISUALISE THE NETWORK
# ─────────────────────────────────────────────

print("\n── Step 5: Generating network figure ──")

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Use a subgraph of the top connected nodes for readability
TOP_NODES = 80   # adjust up/down as needed
top_node_set = set(hub_df.head(TOP_NODES)["gene"].tolist())
H = G.subgraph(top_node_set).copy()

fig, ax = plt.subplots(figsize=(18, 15))
ax.set_facecolor("#fafafa")
fig.patch.set_facecolor("#fafafa")

# FIX 1: use kamada_kawai layout — it clusters connected nodes together
# which naturally groups hub genes, making edges shorter and more visible.
# Falls back to spring layout if the graph is too large (>500 nodes).
if H.number_of_nodes() <= 500:
    pos = nx.kamada_kawai_layout(H, weight="weight")
else:
    pos = nx.spring_layout(H, seed=42, k=2.5 / np.sqrt(H.number_of_nodes()))

# Node colours by regulation (up = coral, down = teal)
reg_map = {n: G.nodes[n].get("regulation", "up") for n in H.nodes()}
node_colors = ["#D85A30" if reg_map[n] == "up" else "#1D9E75"
               for n in H.nodes()]

# Node sizes proportional to hub score
hub_score_map = hub_df.set_index("gene")["hub_score"].to_dict()
node_sizes = [400 + 3500 * hub_score_map.get(n, 0) for n in H.nodes()]

# FIX 2: normalise edge widths to a visible range [0.5, 4.0]
# rather than dividing by 400 (which produces near-zero values)
edges_list = list(H.edges(data=True))
raw_scores = [d.get("weight", 400) for _, _, d in edges_list]
s_min, s_max = min(raw_scores), max(raw_scores)

def norm_width(s):
    if s_max == s_min:
        return 2.0
    return 0.5 + 3.5 * (s - s_min) / (s_max - s_min)

def norm_alpha(s):
    if s_max == s_min:
        return 0.55
    return 0.30 + 0.50 * (s - s_min) / (s_max - s_min)

# FIX 3: draw edges BEFORE nodes so nodes render on top (not buried under edges)
for u, v, d in edges_list:
    w = d.get("weight", 400)
    nx.draw_networkx_edges(
        H, pos,
        edgelist=[(u, v)],
        width=norm_width(w),
        alpha=norm_alpha(w),
        edge_color="#555555",
        ax=ax,
        style="solid",
    )

# Draw nodes on top of edges
nx.draw_networkx_nodes(
    H, pos,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.92,
    linewidths=0.8,
    edgecolors="white",
    ax=ax,
)

# Labels for top 15 hub genes only
top_label_genes = set(hub_df.head(15)["gene"].tolist())
labels = {n: n for n in H.nodes() if n in top_label_genes}
nx.draw_networkx_labels(
    H, pos,
    labels=labels,
    font_size=8.5,
    font_weight="bold",
    font_color="#111111",
    ax=ax,
)

# Legend
legend_elements = [
    mpatches.Patch(facecolor="#D85A30", label="Upregulated"),
    mpatches.Patch(facecolor="#1D9E75", label="Downregulated"),
    Line2D([0],[0], color="#555555", linewidth=0.8, alpha=0.4,
           label="Low confidence edge"),
    Line2D([0],[0], color="#555555", linewidth=3.5, alpha=0.85,
           label="High confidence edge"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.85)

ax.set_title(
    f"HCC PPI Network — top {TOP_NODES} hub genes\n"
    f"(STRING score ≥ {STRING_SCORE}  |  |log2FC| ≥ {LOG2FC_THRESH}  |  padj < {PADJ_THRESH})\n"
    f"Node size = hub score  ·  Edge width = STRING confidence",
    fontsize=13, pad=18,
)
ax.axis("off")
plt.tight_layout()

fig_path = OUT_DIR / "ppi_network.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# ─────────────────────────────────────────────
# 6. EXPORT OUTPUTS
# ─────────────────────────────────────────────

print("\n── Step 6: Exporting files ──")

# --- 6a. Hub gene list (GNN input) ---
hub_out_cols = [
    "gene", "degree", "hub_score",
    "degree_centrality", "betweenness", "closeness", "eigenvector",
    "log2FC", "adj_pvalue", "regulation",
]
hub_path = OUT_DIR / "hub_genes.csv"
hub_df[hub_out_cols].to_csv(hub_path, index=False)
print(f"  Saved: {hub_path}  ({len(hub_df)} genes)")

# --- 6b. Cytoscape edge list ---
# Cytoscape expects: source, target, interaction, weight
cyto_df = edges_df.copy()
cyto_df = cyto_df[
    cyto_df["gene_A"].isin(G.nodes()) & cyto_df["gene_B"].isin(G.nodes())
].copy()

# Add DEA attributes for both endpoints (useful in Cytoscape styling)
for side, col in [("A", "gene_A"), ("B", "gene_B")]:
    cyto_df = cyto_df.merge(
        sig[["gene", "log2FC", "adj_pvalue", "regulation"]].rename(
            columns={
                "gene"      : col,
                "log2FC"    : f"log2FC_{side}",
                "adj_pvalue": f"padj_{side}",
                "regulation": f"regulation_{side}",
            }
        ),
        on=col, how="left",
    )

cyto_df.insert(2, "interaction", "pp")   # protein-protein
cyto_df = cyto_df.rename(columns={
    "gene_A": "source",
    "gene_B": "target",
    "combined_score": "STRING_score",
})

cyto_path = OUT_DIR / "ppi_edges_cytoscape.csv"
cyto_df.to_csv(cyto_path, index=False)
print(f"  Saved: {cyto_path}  ({len(cyto_df)} edges)")

# --- 6c. Node attribute table for Cytoscape ---
node_attr = hub_df[hub_out_cols].copy()
node_attr_path = OUT_DIR / "ppi_nodes_cytoscape.csv"
node_attr.to_csv(node_attr_path, index=False)
print(f"  Saved: {node_attr_path}  (node attributes for Cytoscape)")

# ─────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────

print("\n══════════════════════════════════════════")
print("  PPI ANALYSIS COMPLETE")
print("══════════════════════════════════════════")
print(f"  Genes queried          : {len(gene_list)}")
print(f"  Interactions found     : {G.number_of_edges()}")
print(f"  Nodes in final network : {G.number_of_nodes()}")
print(f"\n  Outputs in: {OUT_DIR.resolve()}/")
print("    • ppi_network.png")
print("    • hub_genes.csv")
print("    • ppi_edges_cytoscape.csv")
print("    • ppi_nodes_cytoscape.csv")
print(f"\n  Top 5 hub genes:")
for _, row in hub_df.head(5).iterrows():
    print(f"    {row['gene']:<12} degree={int(row['degree']):<4} "
          f"hub_score={row['hub_score']:.3f}  [{row['regulation']}]")
