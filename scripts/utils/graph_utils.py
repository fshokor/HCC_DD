"""
graph_utils.py
==============
Shared graph-building utilities used by the PPI (step 09) and
GNN (steps 13-14) scripts.

Functions
---------
build_ppi_graph       — NetworkX graph from STRING edge list
compute_hub_scores    — four centrality measures + composite hub score
build_gnn_graph       — PyTorch Geometric Data object from DGI edge list
edge_tensors          — helper to extract (src, dst, labels) tensors
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
# from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# PPI utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_ppi_graph(sig_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """
    Build a NetworkX graph from a filtered DEA dataframe and STRING edges.

    Parameters
    ----------
    sig_df : pd.DataFrame
        Significant DEGs with columns: gene, log2FC, adj_pvalue, regulation.
    edges_df : pd.DataFrame
        STRING edge list with columns: gene_A, gene_B, combined_score.

    Returns
    -------
    G : nx.Graph
        Undirected graph with node attributes (log2FC, regulation) and
        edge weights (STRING combined score). Isolated nodes are removed.
    """
    G = nx.Graph()

    for _, row in sig_df.iterrows():
        G.add_node(
            row["gene"],
            log2FC=row["log2FC"],
            adj_pvalue=row["adj_pvalue"],
            regulation=row["regulation"],
        )

    for _, row in edges_df.iterrows():
        if row["gene_A"] in G and row["gene_B"] in G:
            G.add_edge(
                row["gene_A"], row["gene_B"],
                weight=float(row["combined_score"]),
            )

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G


def compute_hub_scores(G: nx.Graph, sig_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute four centrality measures per node and combine into a composite
    hub score (normalised mean of all four).

    Parameters
    ----------
    G : nx.Graph
        PPI graph, already pruned of isolates.
    sig_df : pd.DataFrame
        DEA results to merge back (log2FC, regulation, adj_pvalue).

    Returns
    -------
    hub_df : pd.DataFrame
        One row per gene, sorted by hub_score descending.
        Columns: gene, degree, hub_score, deg_c, betweenness,
                 closeness, eigenvector, log2FC, adj_pvalue, regulation.
    """
    deg_c = nx.degree_centrality(G)
    bet_c = nx.betweenness_centrality(G, weight="weight")
    clo_c = nx.closeness_centrality(G)
    eig_c = nx.eigenvector_centrality(G, max_iter=500, weight="weight")

    hub_df = pd.DataFrame({
        "gene"       : list(G.nodes()),
        "degree"     : [G.degree(n) for n in G.nodes()],
        "deg_c"      : [deg_c[n]  for n in G.nodes()],
        "betweenness": [bet_c[n]  for n in G.nodes()],
        "closeness"  : [clo_c[n]  for n in G.nodes()],
        "eigenvector": [eig_c[n]  for n in G.nodes()],
    })

    for col in ["deg_c", "betweenness", "closeness", "eigenvector"]:
        mn, mx = hub_df[col].min(), hub_df[col].max()
        hub_df[f"{col}_n"] = (hub_df[col] - mn) / (mx - mn + 1e-9)

    hub_df["hub_score"] = hub_df[
        [c for c in hub_df.columns if c.endswith("_n")]
    ].mean(axis=1)

    hub_df = hub_df.merge(
        sig_df[["gene", "log2FC", "adj_pvalue", "regulation"]],
        on="gene", how="left",
    )
    return hub_df.sort_values("hub_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# GNN utilities
# ─────────────────────────────────────────────────────────────────────────────

DRUG_FEAT_COLS = [
    "approved", "immunotherapy", "anti_neoplastic",
    "clinical_phase", "interaction_score", "n_publications",
    "source_DGIdb", "source_ChEMBL", "source_OpenTargets",
    "type_inhibitor", "type_agonist", "type_antagonist",
    "type_antibody", "type_binder", "type_activator",
]
GENE_FEAT_COLS = ["hub_score", "survival_target"]


def build_gnn_graph(
    edges_df: pd.DataFrame,
    drug_feat_cols: list = DRUG_FEAT_COLS,
    gene_feat_cols: list = GENE_FEAT_COLS,
) -> tuple:
    """
    Build a PyTorch Geometric Data object and associated index maps
    from a drug-gene edge list (output of step 12).

    Parameters
    ----------
    edges_df : pd.DataFrame
        Edge list with columns: gene, drug, composite_score, + feature cols.
    drug_feat_cols : list
        Column names to use as drug node features.
    gene_feat_cols : list
        Column names to use as gene node features.

    Returns
    -------
    graph_data : torch_geometric.data.Data
        PyG Data object with x (node features) and edge_index.
    node2idx : dict
        Maps node name → integer index.
    idx2node : dict
        Maps integer index → node name.
    labels : torch.Tensor
        Edge-level regression targets (composite_score).
    gene_set : set
        Set of gene node names.
    drug_set : set
        Set of drug node names.
    scaler : StandardScaler
        Fitted scaler (save this alongside the model for inference).
    """
    all_nodes = pd.concat([edges_df["gene"], edges_df["drug"]]).unique().tolist()
    node2idx  = {n: i for i, n in enumerate(all_nodes)}
    idx2node  = {i: n for n, i in node2idx.items()}
    n_nodes   = len(all_nodes)
    gene_set  = set(edges_df["gene"].unique())
    drug_set  = set(edges_df["drug"].unique())

    all_feat_cols = drug_feat_cols + gene_feat_cols
    feat_dim      = len(all_feat_cols)
    node_feats    = np.zeros((n_nodes, feat_dim), dtype=np.float32)

    gene_rows = edges_df.drop_duplicates("gene").set_index("gene")
    for gene, idx in node2idx.items():
        if gene in gene_set and gene in gene_rows.index:
            row = gene_rows.loc[gene]
            for j, col in enumerate(gene_feat_cols):
                if col in row.index:
                    node_feats[idx, len(drug_feat_cols) + j] = float(row[col])

    drug_rows = edges_df.drop_duplicates("drug").set_index("drug")
    for drug, idx in node2idx.items():
        if drug in drug_set and drug in drug_rows.index:
            row = drug_rows.loc[drug]
            for j, col in enumerate(drug_feat_cols):
                if col in row.index:
                    node_feats[idx, j] = float(row[col])

    scaler     = StandardScaler()
    node_feats = scaler.fit_transform(node_feats).astype(np.float32)

    src_n = [node2idx[g] for g in edges_df["gene"]]
    dst_n = [node2idx[d] for d in edges_df["drug"]]
    edge_index = torch.tensor([src_n + dst_n, dst_n + src_n], dtype=torch.long)
    labels     = torch.tensor(edges_df["composite_score"].values, dtype=torch.float32)
    graph_data = Data(x=torch.tensor(node_feats, dtype=torch.float32),
                      edge_index=edge_index)

    return graph_data, node2idx, idx2node, labels, gene_set, drug_set, scaler


def edge_tensors(
    df: pd.DataFrame,
    idx_list: list,
    node2idx: dict,
    labels: torch.Tensor,
) -> tuple:
    """
    Extract (src, dst, y) tensors for a given list of edge indices.

    Parameters
    ----------
    df : pd.DataFrame
        Original edge dataframe (gene, drug columns).
    idx_list : list
        Row indices to select.
    node2idx : dict
        Node-name to integer-index mapping.
    labels : torch.Tensor
        Full label tensor (all edges).

    Returns
    -------
    src, dst : torch.Tensor
        Source (gene) and destination (drug) node indices.
    y : torch.Tensor
        Regression targets for the selected edges.
    """
    src = torch.tensor([node2idx[df.iloc[i]["gene"]] for i in idx_list])
    dst = torch.tensor([node2idx[df.iloc[i]["drug"]] for i in idx_list])
    return src, dst, labels[idx_list]
