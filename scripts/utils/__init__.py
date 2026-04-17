"""
utils — shared helpers for the HCC drug discovery pipeline.

Modules
-------
graph_utils   — PPI graph construction + GNN graph building
plot_utils    — all matplotlib figure functions
api_clients   — DGIdb / ChEMBL / OpenTargets HTTP clients
"""

from .graph_utils import (
    build_ppi_graph,
    compute_hub_scores,
    build_gnn_graph,
    edge_tensors,
)
from .plot_utils import (
    plot_ppi_network,
    plot_km_grid,
    plot_cox_forest,
    plot_drug_ranking,
    plot_training_curves,
    plot_model_comparison,
    plot_scatter,
)
from .api_clients import (
    safe_request,
    query_dgidb,
    query_chembl,
    query_opentargets,
    get_curated_fallback,
)

__all__ = [
    "build_ppi_graph", "compute_hub_scores", "build_gnn_graph", "edge_tensors",
    "plot_ppi_network", "plot_km_grid", "plot_cox_forest", "plot_drug_ranking",
    "plot_training_curves", "plot_model_comparison", "plot_scatter",
    "safe_request", "query_dgidb", "query_chembl", "query_opentargets",
    "get_curated_fallback",
]
