"""
gnn_functions.py
================
All logic for notebook P4 · GNN Drug Candidate Ranking.

Functions
---------
build_graph         — construct PyG Data object from DGI edge list
make_edge_tensors   — extract (src, dst, labels) for a split
GCNModel            — 2-layer Graph Convolutional Network
GATModel            — Graph Attention Network (4 heads)
SAGEModel           — GraphSAGE (mean aggregation)
train_model         — training loop with early stopping
evaluate_model      — R² / MSE / MAE on a held-out split
rank_drugs          — score all edges with the best model
export_results      — save weights, embeddings, ranking CSV
plot_training       — training loss curves
plot_comparison     — R² / MSE / MAE bar chart
plot_scatter        — predicted vs actual scatter
plot_ranking        — horizontal bar chart of top drugs
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DRUG_FEAT_COLS = [
    "approved", "immunotherapy", "anti_neoplastic", "clinical_phase",
    "interaction_score", "n_publications", "source_DGIdb", "source_ChEMBL",
    "source_OpenTargets", "type_inhibitor", "type_agonist", "type_antagonist",
    "type_antibody", "type_binder", "type_activator",
]
GENE_FEAT_COLS = ["hub_score", "survival_target"]
MODEL_COLORS   = {"GCN": "#534AB7", "GAT": "#D85A30", "GraphSAGE": "#1D9E75"}
PHASE_COLORS   = {0:"#D3D1C7",1:"#B5D4F4",2:"#378ADD",3:"#185FA5",4:"#1D9E75"}


# ─────────────────────────────────────────────────────────────────────────────
def build_graph(edges_df, device, random_seed=42):
    """
    Build a PyTorch Geometric Data object from the DGI edge list.

    Constructs a bipartite graph where genes and drugs are nodes, and
    each drug-gene interaction is a bidirectional edge.
    Node features are standardised with StandardScaler.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Output of P3 (dgi_edges_gnn.csv).
        Required columns: gene, drug, composite_score + feature columns.
    device : torch.device
    random_seed : int

    Returns
    -------
    graph_data : torch_geometric.data.Data
        Node feature matrix x, bidirectional edge_index.
    node2idx, idx2node : dict
        Node-name ↔ integer-index maps.
    labels : torch.Tensor
        Edge-level regression targets (composite_score).
    gene_set, drug_set : set
        Node names by type.
    scaler : StandardScaler
        Fitted scaler (save alongside model weights for inference).
    tr_idx, va_idx, te_idx : list
        Train / val / test edge indices.
    """
    all_nodes = pd.concat([edges_df.gene, edges_df.drug]).unique().tolist()
    node2idx  = {n: i for i, n in enumerate(all_nodes)}
    idx2node  = {i: n for n, i in node2idx.items()}
    gene_set  = set(edges_df.gene.unique())
    drug_set  = set(edges_df.drug.unique())
    n_nodes   = len(all_nodes)

    all_feat_cols = DRUG_FEAT_COLS + GENE_FEAT_COLS
    feat_dim      = len(all_feat_cols)
    node_feats    = np.zeros((n_nodes, feat_dim), dtype=np.float32)

    gene_rows = edges_df.drop_duplicates("gene").set_index("gene")
    for gene, idx in node2idx.items():
        if gene in gene_set and gene in gene_rows.index:
            row = gene_rows.loc[gene]
            for j, col in enumerate(GENE_FEAT_COLS):
                if col in row.index:
                    node_feats[idx, len(DRUG_FEAT_COLS)+j] = float(row[col])

    drug_rows = edges_df.drop_duplicates("drug").set_index("drug")
    for drug, idx in node2idx.items():
        if drug in drug_set and drug in drug_rows.index:
            row = drug_rows.loc[drug]
            for j, col in enumerate(DRUG_FEAT_COLS):
                if col in row.index:
                    node_feats[idx, j] = float(row[col])

    scaler     = StandardScaler()
    node_feats = scaler.fit_transform(node_feats).astype(np.float32)

    src_n = [node2idx[g] for g in edges_df.gene]
    dst_n = [node2idx[d] for d in edges_df.drug]
    edge_index = torch.tensor([src_n+dst_n, dst_n+src_n], dtype=torch.long)
    labels     = torch.tensor(edges_df.composite_score.values, dtype=torch.float32)
    graph_data = Data(x=torch.tensor(node_feats, dtype=torch.float32),
                      edge_index=edge_index).to(device)

    indices = list(range(len(edges_df)))
    tr_idx, te_idx = train_test_split(indices, test_size=0.15,
                                      random_state=random_seed)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.15/0.85,
                                      random_state=random_seed)

    print(f"Nodes      : {n_nodes}  ({len(gene_set)} genes + {len(drug_set)} drugs)")
    print(f"Edges      : {edge_index.shape[1]}  (bidirectional)")
    print(f"Feature dim: {feat_dim}")
    print(f"Split      : train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")
    print(f"Device     : {device}")
    return (graph_data, node2idx, idx2node, labels,
            gene_set, drug_set, scaler, tr_idx, va_idx, te_idx)


# ─────────────────────────────────────────────────────────────────────────────
def make_edge_tensors(edges_df, idx_list, node2idx, labels, device):
    """
    Extract source, destination, and label tensors for a list of edge indices.
    """
    src = torch.tensor(
        [node2idx[edges_df.iloc[i].gene] for i in idx_list]).to(device)
    dst = torch.tensor(
        [node2idx[edges_df.iloc[i].drug] for i in idx_list]).to(device)
    return src, dst, labels[idx_list].to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

class GCNModel(nn.Module):
    """2-layer Graph Convolutional Network with BatchNorm and edge prediction head."""
    def __init__(self, in_dim, hidden=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.bn2   = nn.BatchNorm1d(out_dim)
        self.head  = nn.Sequential(
            nn.Linear(out_dim*2, out_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(out_dim, 1), nn.Sigmoid())
        self.drop  = dropout

    def encode(self, x, ei):
        x = F.relu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=self.drop, training=self.training)
        return F.relu(self.bn2(self.conv2(x, ei)))

    def forward(self, x, ei, src, dst):
        z = self.encode(x, ei)
        return self.head(torch.cat([z[src], z[dst]], dim=1)).squeeze(-1)


class GATModel(nn.Module):
    """Graph Attention Network with multi-head attention."""
    def __init__(self, in_dim, hidden=32, out_dim=64, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden*heads, out_dim, heads=1,
                             dropout=dropout, concat=False)
        self.bn1   = nn.BatchNorm1d(hidden*heads)
        self.bn2   = nn.BatchNorm1d(out_dim)
        self.head  = nn.Sequential(
            nn.Linear(out_dim*2, out_dim), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(out_dim, 1), nn.Sigmoid())
        self.drop  = dropout

    def encode(self, x, ei):
        x = F.elu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=self.drop, training=self.training)
        return F.elu(self.bn2(self.conv2(x, ei)))

    def forward(self, x, ei, src, dst):
        z = self.encode(x, ei)
        return self.head(torch.cat([z[src], z[dst]], dim=1)).squeeze(-1)


class SAGEModel(nn.Module):
    """GraphSAGE with mean neighbourhood aggregation."""
    def __init__(self, in_dim, hidden=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, out_dim, aggr="mean")
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(out_dim)
        self.head  = nn.Sequential(
            nn.Linear(out_dim*2, out_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(out_dim, 1), nn.Sigmoid())
        self.drop  = dropout

    def encode(self, x, ei):
        x = F.relu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=self.drop, training=self.training)
        return F.relu(self.bn2(self.conv2(x, ei)))

    def forward(self, x, ei, src, dst):
        z = self.encode(x, ei)
        return self.head(torch.cat([z[src], z[dst]], dim=1)).squeeze(-1)


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, graph_data, edges_df, node2idx, labels,
                tr_idx, va_idx, device,
                lr=0.005, weight_decay=1e-4,
                n_epochs=300, patience=40):
    """
    Train a GNN model with Adam, ReduceLROnPlateau, and early stopping.

    Parameters
    ----------
    model : nn.Module
        GCNModel, GATModel, or SAGEModel.
    graph_data : Data
        PyG graph on device.
    edges_df : pd.DataFrame
        Original edge list (needed to look up node indices).
    node2idx : dict
    labels : torch.Tensor
    tr_idx, va_idx : list
        Training and validation edge indices.
    device : torch.device
    lr, weight_decay, n_epochs, patience : hyperparameters

    Returns
    -------
    history : dict
        train_loss, val_loss, lr per epoch.
    best_val : float
        Best validation MSE achieved.
    """
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", patience=15, factor=0.5, min_lr=1e-5)

    s_tr, d_tr, y_tr = make_edge_tensors(edges_df, tr_idx, node2idx, labels, device)
    s_va, d_va, y_va = make_edge_tensors(edges_df, va_idx, node2idx, labels, device)

    best_val, best_state, no_imp = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for ep in range(1, n_epochs+1):
        model.train(); opt.zero_grad()
        loss = F.mse_loss(
            model(graph_data.x, graph_data.edge_index, s_tr, d_tr), y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(
                model(graph_data.x, graph_data.edge_index, s_va, d_va),
                y_va).item()
        sch.step(vl)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(vl)
        history["lr"].append(opt.param_groups[0]["lr"])

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            print(f"    Early stop ep {ep}  best_val={best_val:.4f}")
            break
        if ep % 50 == 0:
            print(f"    ep {ep:3d}  train={loss.item():.4f}  val={vl:.4f}")

    model.load_state_dict(best_state)
    return history, best_val


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, graph_data, edges_df, node2idx, labels,
                   idx_list, device):
    """
    Compute R², MSE, and MAE on a held-out split.

    Returns
    -------
    dict with keys: r2, mse, mae, pred (np.array), true (np.array).
    """
    model.eval()
    s, d, yt = make_edge_tensors(edges_df, idx_list, node2idx, labels, device)
    with torch.no_grad():
        yp = model(graph_data.x, graph_data.edge_index, s, d).cpu().numpy()
    yt = yt.cpu().numpy()
    return {
        "r2"  : float(r2_score(yt, yp)),
        "mse" : float(mean_squared_error(yt, yp)),
        "mae" : float(mean_absolute_error(yt, yp)),
        "pred": yp,
        "true": yt,
    }


# ─────────────────────────────────────────────────────────────────────────────
def rank_drugs(best_model, graph_data, edges_df, node2idx, device):
    """
    Score every drug-gene edge with the best model and return a ranked dataframe.

    Parameters
    ----------
    best_model : nn.Module
        Best trained model (already on device).
    graph_data : Data
    edges_df : pd.DataFrame
    node2idx : dict
    device : torch.device

    Returns
    -------
    ranking : pd.DataFrame
        All edges with gnn_score, original_score, score_delta, rank columns.
        Sorted by gnn_score descending.
    """
    best_model.eval()
    all_src = torch.tensor([node2idx[g] for g in edges_df.gene]).to(device)
    all_dst = torch.tensor([node2idx[d] for d in edges_df.drug]).to(device)
    with torch.no_grad():
        scores = best_model(
            graph_data.x, graph_data.edge_index, all_src, all_dst
        ).cpu().numpy()

    ranking = edges_df.copy()
    ranking["gnn_score"]      = scores.round(4)
    ranking["original_score"] = ranking.composite_score
    ranking["score_delta"]    = (ranking.gnn_score - ranking.original_score).round(4)
    ranking = ranking.sort_values("gnn_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1
    return ranking


# ─────────────────────────────────────────────────────────────────────────────
def export_results(best_name, best_model, all_results,
                   edges_df, node2idx, idx2node, gene_set,
                   embed_dim, scaler, models_dir, tables_dir):
    """
    Save model weights, feature scaler, node embeddings, and drug ranking CSV.

    Parameters
    ----------
    best_name : str
    best_model : nn.Module
    all_results : dict
        Keyed by model name, values contain 'embeddings' array.
    edges_df : pd.DataFrame
    node2idx, idx2node : dict
    gene_set : set
    embed_dim : int
    scaler : StandardScaler
    models_dir, tables_dir : Path
    """
    torch.save(best_model.state_dict(), models_dir / "gcn_best.pt")
    with open(models_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    n_nodes  = len(node2idx)
    drug_set = set(idx2node[i] for i in range(n_nodes)) - gene_set
    embed_df = pd.DataFrame(
        all_results[best_name]["embeddings"],
        columns=[f"dim_{i}" for i in range(embed_dim)])
    embed_df.insert(0, "node",      [idx2node[i] for i in range(n_nodes)])
    embed_df.insert(1, "node_type", ["gene" if embed_df.iloc[i].node in gene_set
                                      else "drug" for i in range(n_nodes)])
    embed_df.to_csv(tables_dir / "gnn_node_embeddings.csv", index=False)

    rank_cols = ["rank","gene","drug","gnn_score","original_score","score_delta",
                 "approved","clinical_phase","interaction_type","source"]
    ranking   = all_results[best_name]["ranking"]
    ranking[[c for c in rank_cols if c in ranking.columns]].to_csv(
        tables_dir / "gnn_drug_ranking.csv", index=False)

    print(f"Saved: models/gcn_best.pt")
    print(f"Saved: models/feature_scaler.pkl")
    print(f"Saved: tables/gnn_node_embeddings.csv  ({n_nodes} nodes)")
    print(f"Saved: tables/gnn_drug_ranking.csv     ({len(ranking)} drugs)")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(all_results, best_name, figures_dir):
    """Training loss curves for all models."""
    model_names = list(all_results.keys())
    fig, axes   = plt.subplots(1, len(model_names),
                               figsize=(6*len(model_names), 5), facecolor="white")
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        h   = all_results[name]["history"]
        ep  = range(1, len(h["train_loss"])+1)
        col = MODEL_COLORS[name]
        m   = all_results[name]["test"]
        ax.plot(ep, h["train_loss"], color=col, lw=2, label="Train")
        ax.plot(ep, h["val_loss"],   color=col, lw=1.5, ls="--",
                alpha=0.65, label="Val")
        ax.fill_between(ep, h["train_loss"], h["val_loss"], color=col, alpha=0.06)
        ax.axvline(int(np.argmin(h["val_loss"]))+1,
                   color=col, lw=0.8, ls=":", alpha=0.6)
        ax.set_title(
            f"{name}\nR²={m['r2']:.4f}  MSE={m['mse']:.4f}",
            fontsize=11,
            color=col if name==best_name else "black",
            fontweight="bold" if name==best_name else "normal")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
        if name == best_name:
            ax.set_facecolor("#f8f6ff")
    fig.suptitle(f"GNN training curves — best: {best_name}", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "gnn_training_curves.png", dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_comparison(all_results, figures_dir):
    """R² / MSE / MAE grouped bar chart."""
    model_names = list(all_results.keys())
    metric_defs = [("r2","R² (↑)",True),("mse","MSE (↓)",False),("mae","MAE (↓)",False)]
    fig, axes   = plt.subplots(1, 3, figsize=(13, 5), facecolor="white")
    for ax, (metric, label, higher) in zip(axes, metric_defs):
        vals   = [all_results[n]["test"][metric] for n in model_names]
        colors = [MODEL_COLORS[n] for n in model_names]
        best_v = max(vals) if higher else min(vals)
        bars   = ax.bar(model_names, vals, color=colors, width=0.5)
        for bar, val in zip(bars, vals):
            bar.set_alpha(1.0 if val == best_v else 0.55)
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(vals)*0.015,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, max(vals)*1.22)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Model comparison — test set", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "gnn_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_scatter(all_results, best_name, figures_dir):
    """Predicted vs actual scatter with residual lines."""
    model_names = list(all_results.keys())
    fig, axes   = plt.subplots(1, len(model_names),
                               figsize=(5*len(model_names), 5), facecolor="white")
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        m   = all_results[name]["test"]
        col = MODEL_COLORS[name]
        ax.scatter(m["true"], m["pred"], color=col, s=90, alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)
        lo = min(m["true"].min(), m["pred"].min()) - 0.05
        hi = max(m["true"].max(), m["pred"].max()) + 0.05
        ax.plot([lo,hi],[lo,hi], "k--", lw=1, alpha=0.45, zorder=2)
        for t, p in zip(m["true"], m["pred"]):
            ax.plot([t,t],[t,p], color=col, lw=0.6, alpha=0.30, zorder=1)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel("True score"); ax.set_ylabel("Predicted score")
        ax.set_title(f"{name}  R²={m['r2']:.4f}", fontsize=10,
                     color=col if name==best_name else "black")
        ax.spines[["top","right"]].set_visible(False)
        if name == best_name:
            ax.set_facecolor("#f8f6ff")
    fig.suptitle("Predicted vs actual — test set", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "gnn_predicted_vs_actual.png",
                dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_ranking(ranking, best_name, figures_dir, top_n=25):
    """Horizontal bar chart of top drug candidates ranked by GNN score."""
    top_df   = ranking.head(top_n).iloc[::-1].reset_index(drop=True)
    bar_cols = [PHASE_COLORS.get(int(p), "#888780")
                for p in top_df.clinical_phase]

    fig, ax = plt.subplots(figsize=(14, max(8, top_n*0.38)), facecolor="white")
    ax.barh(range(len(top_df)), top_df.gnn_score,
            color=bar_cols, alpha=0.88, height=0.72)
    for i, row in top_df.iterrows():
        ax.text(row.gnn_score+0.005, i,
                "\u2605" if row.approved else "\u25CB",
                va="center", fontsize=11,
                color="#1D9E75" if row.approved else "#888780")
        if "original_score" in row:
            ax.scatter(row.original_score, i, marker="|", s=60,
                       color="#444441", zorder=5, linewidths=1.5)

    labels = [row.drug + "   \u2192   " + row.gene
              for _, row in top_df.iterrows()]
    ax.set_yticks(range(len(top_df))); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(f"GNN-predicted interaction score ({best_name})", fontsize=11)
    ax.set_title(f"Top {top_n} drug candidates\n"
                 "\u2605 = approved   \u25CB = not approved   | = original score",
                 fontsize=11, pad=12)
    legend_p = [mpatches.Patch(color=c,
                               label=f"Phase {k}" if k>0 else "Preclinical",
                               alpha=0.88)
                for k, c in sorted(PHASE_COLORS.items())]
    ax.legend(handles=legend_p, loc="lower right",
              fontsize=8.5, framealpha=0.85, title="Clinical phase")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0, 1.10)
    ax.axvline(0.5, color="#ccc", lw=0.8, ls="--")
    plt.tight_layout()
    fig.savefig(figures_dir / "gnn_drug_ranking.png", dpi=200, bbox_inches="tight")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_drug_gene_network(ranking, best_name, figures_dir,
                           top_n=20,
                           min_gnn_score=0.0):
    """
    Bipartite drug–gene network visualisation using the GNN-ranked results.

    Layout
    ------
    Genes are arranged on the left column, drugs on the right column.
    Edge width and opacity encode the GNN-predicted interaction score.
    Node size for genes encodes how many top-ranked drugs interact with them.
    Drug nodes are coloured by clinical phase; gene nodes by hub score (if present).
    Approved drugs are marked with a gold border.

    Parameters
    ----------
    ranking : pd.DataFrame
        Output of rank_drugs() — must contain columns:
        drug, gene, gnn_score, approved, clinical_phase.
        Optional: hub_score (gene attribute).
    best_name : str
        Name of the winning model (used in title).
    figures_dir : Path
    top_n : int
        Number of top-ranked drug–gene pairs to display (default 20).
    min_gnn_score : float
        Only show edges with gnn_score ≥ this threshold (default 0.0).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from pathlib import Path
    figures_dir = Path(figures_dir)

    # ── Select edges to display ───────────────────────────────────────────────
    df = ranking.copy()
    if min_gnn_score > 0:
        df = df[df["gnn_score"] >= min_gnn_score]
    df = df.head(top_n).reset_index(drop=True)

    if df.empty:
        print("plot_drug_gene_network: no edges to display.")
        return None

    genes = df["gene"].unique().tolist()
    drugs = df["drug"].unique().tolist()

    # ── Build networkx graph ──────────────────────────────────────────────────
    G = nx.Graph()
    G.add_nodes_from(genes, node_type="gene")
    G.add_nodes_from(drugs, node_type="drug")
    for _, row in df.iterrows():
        G.add_edge(row["drug"], row["gene"],
                   gnn_score=row["gnn_score"],
                   approved=bool(row.get("approved", False)),
                   clinical_phase=int(row.get("clinical_phase", 0)))

    # ── Layout: genes left, drugs right ──────────────────────────────────────
    n_genes = len(genes)
    n_drugs = len(drugs)
    pos = {}
    # Space genes evenly on x=0
    for i, g in enumerate(genes):
        pos[g] = (0.0, (n_genes - 1 - i) / max(n_genes - 1, 1))
    # Space drugs evenly on x=1
    for i, d in enumerate(drugs):
        pos[d] = (1.0, (n_drugs - 1 - i) / max(n_drugs - 1, 1))

    # ── Node attributes ───────────────────────────────────────────────────────
    # Gene node size ~ number of top-ranked drug connections
    gene_degree = {g: sum(1 for _, row in df.iterrows() if row["gene"] == g)
                   for g in genes}
    gene_sizes  = [300 + gene_degree[g] * 120 for g in genes]

    # Drug node colour ~ clinical phase
    drug_phase_map = (df.drop_duplicates("drug")
                        .set_index("drug")["clinical_phase"]
                        .to_dict()
                      if "clinical_phase" in df.columns else {})
    drug_colors = [PHASE_COLORS.get(int(drug_phase_map.get(d, 0)), "#D3D1C7")
                   for d in drugs]

    # Approved drugs get a gold edge colour
    drug_approved = (df.drop_duplicates("drug")
                       .set_index("drug")["approved"]
                       .to_dict()
                     if "approved" in df.columns else {})
    drug_edgecolors = ["#E6B84A" if drug_approved.get(d, False) else "#cccccc"
                       for d in drugs]
    drug_linewidths = [2.5 if drug_approved.get(d, False) else 0.8
                       for d in drugs]

    # ── Edge attributes ───────────────────────────────────────────────────────
    edges     = list(G.edges(data=True))
    scores    = np.array([e[2]["gnn_score"] for e in edges])
    score_min, score_max = scores.min(), scores.max()
    norm      = (scores - score_min) / (score_max - score_min + 1e-9)
    edge_widths  = 0.5 + norm * 4.5   # 0.5 → 5.0
    edge_alphas  = 0.25 + norm * 0.65  # 0.25 → 0.90

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_h = max(8, max(n_genes, n_drugs) * 0.55)
    fig, ax = plt.subplots(figsize=(14, fig_h), facecolor="white")

    # Draw edges individually so alpha and width vary per edge
    for (u, v, data), width, alpha in zip(edges, edge_widths, edge_alphas):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1],
                color="#534AB7", linewidth=width, alpha=alpha, zorder=1)

    # Draw gene nodes
    gene_xs = [pos[g][0] for g in genes]
    gene_ys = [pos[g][1] for g in genes]
    ax.scatter(gene_xs, gene_ys, s=gene_sizes,
               color="#3a0ca3", edgecolors="#1a1a2e", linewidths=1.0,
               zorder=3, label="Gene")

    # Draw drug nodes
    drug_xs = [pos[d][0] for d in drugs]
    drug_ys = [pos[d][1] for d in drugs]
    ax.scatter(drug_xs, drug_ys, s=280,
               color=drug_colors, edgecolors=drug_edgecolors,
               linewidths=drug_linewidths, zorder=3, marker="D",
               label="Drug (◆)")

    # ── Labels ────────────────────────────────────────────────────────────────
    label_pad = 0.03
    for g in genes:
        ax.text(pos[g][0] - label_pad, pos[g][1], g,
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color="#1a1a2e")
    for d in drugs:
        # Append ★ for approved
        suffix = "  ★" if drug_approved.get(d, False) else ""
        ax.text(pos[d][0] + label_pad, pos[d][1], d + suffix,
                ha="left", va="center", fontsize=8,
                color="#1a1a2e")

    # ── Edge score colorbar ───────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Blues,
        norm=mcolors.Normalize(vmin=score_min, vmax=score_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.35, pad=0.01, aspect=20)
    cbar.set_label("GNN interaction score", fontsize=9)

    # ── Phase legend ──────────────────────────────────────────────────────────
    phase_labels = {0: "Preclinical", 1: "Phase 1", 2: "Phase 2",
                    3: "Phase 3",     4: "Approved (Phase 4)"}
    phase_patches = [
        mpatches.Patch(color=c, label=phase_labels[k], alpha=0.88)
        for k, c in sorted(PHASE_COLORS.items())
        if any(int(drug_phase_map.get(d, 0)) == k for d in drugs)
    ]
    approved_patch = mpatches.Patch(
        facecolor="white", edgecolor="#E6B84A", linewidth=2.5,
        label="FDA approved (gold border)")
    ax.legend(handles=phase_patches + [approved_patch],
              loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=8, framealpha=0.9,
              title="Drug clinical phase", title_fontsize=8)

    # ── Axes cosmetics ────────────────────────────────────────────────────────
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")

    # Column headers
    ax.text(0.0, 1.06, "Hub genes", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#3a0ca3",
            transform=ax.transData)
    ax.text(1.0, 1.06, "Drug candidates", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#185FA5",
            transform=ax.transData)

    ax.set_title(
        f"Drug–Gene Interaction Network — top {top_n} pairs ({best_name})\n"
        f"Edge width & opacity = GNN score  |  Gene size = no. of top interactions  "
        f"|  ★ = FDA approved",
        fontsize=10, pad=14)

    plt.tight_layout()
    out = figures_dir / "drug_gene_network.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")
    return fig
