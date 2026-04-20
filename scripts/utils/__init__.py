"""
utils — shared helpers for the HCC drug discovery pipeline.

Modules
-------
graph_utils      — PPI graph construction + GNN graph building
plot_utils       — all matplotlib figure functions
api_clients      — DGIdb / ChEMBL / OpenTargets HTTP clients
report_functions — HTML report generators for notebooks 01–03
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
from .scrna_functions import (
    load_samples,
    qc_metrics,
    filter_cells,
    normalize,
    select_hvg,
    save_adata,
    run_pca,
    run_umap,
    run_leiden,
    run_celltypist,
    prep_seurat_object,
    pull_r_col,
    marker_score_clusters,
    majority_vote,
    MARKER_SETS,
)

from .gsea_functions import (
    prepare_ranked_list,
    run_gsea_r,
    print_gsea_summary,
)

from .dea_functions import (
    run_wilcoxon,
    plot_volcano,
    export_dea,
)
from .dgi_functions import (
    load_dgi_inputs,
    collect_interactions,
    build_dgi_dataframe,
    build_gnn_edge_list,
    plot_dgi_dashboard,
)

from .gnn_functions import (
    param_count,
    build_graph,
    make_edge_tensors,
    GCNModel,
    GATModel,
    SAGEModel,
    train_model,
    evaluate_model,
    rank_drugs,
    export_results,
    plot_training,
    plot_comparison,
    plot_scatter,
    plot_ranking,
)
from .ppi_functions import (
    load_dea,
    query_string,
    build_and_score,
    export_ppi,
)
from .survival_functions import (
    load_gene_list,
    fetch_tcga_lihc,
    simulate_tcga,
    run_survival,
    filter_survivors,
    export_survival,
)
from .report_functions import (
    generate_scrna_report,
    generate_target_report,
    generate_gnn_report,
)

__all__ = [
    "build_ppi_graph", "compute_hub_scores", "build_gnn_graph", "edge_tensors",
    "plot_ppi_network", "plot_km_grid", "plot_cox_forest", "plot_drug_ranking",
    "plot_training_curves", "plot_model_comparison", "plot_scatter",
    "safe_request", "query_dgidb", "query_chembl", "query_opentargets",
    "get_curated_fallback",
    "load_samples", "qc_metrics", "filter_cells", "normalize", "select_hvg",
    "save_adata", "run_pca", "run_umap", "run_leiden", "run_celltypist",
    "prep_seurat_object", "pull_r_col", "marker_score_clusters",
    "majority_vote", 
    "plot_ppi_network", "plot_km_grid", "plot_cox_forest", "plot_drug_ranking",
    "plot_training_curves", "plot_model_comparison", "plot_scatter",
    "prepare_ranked_list", "run_gsea_r", "print_gsea_summary", 
    "build_ppi_graph", "compute_hub_scores", "build_gnn_graph", "edge_tensors",
    "run_wilcoxon", "plot_volcano", "export_dea",
    "plot_dgi_dashboard",
    "safe_request", "query_dgidb", "query_chembl", "query_opentargets",
    "get_curated_fallback",
    "generate_scrna_report", "generate_target_report", "generate_gnn_report",
]
