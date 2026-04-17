# =============================================================================
# r_packages.R
# =============================================================================
# Installs all R packages required for the annotation notebooks (step 03).
#
# Usage (run once after setting up the conda environment):
#   Rscript env/r_packages.R
#
# Requires: R >= 4.3
# Approximate install time: 10–20 min (downloads from Bioconductor + CRAN)
# =============================================================================

message("=== HCC Drug Discovery — R package installer ===")
message("R version: ", R.version$major, ".", R.version$minor)

# ── 1. CRAN packages ─────────────────────────────────────────────────────────

cran_packages <- c(
  "BiocManager",   # Bioconductor installer — must be first
  "remotes",       # GitHub installs
  "dplyr",         # data wrangling (used throughout)
  "ggplot2",       # plotting
  "pheatmap",      # heatmaps for SingleR score visualisation
  "viridis",       # colour palettes
  "openxlsx",      # read ScType marker database (.xlsx)
  "HGNChelper"     # HGNC gene symbol validation (used by ScType)
)

message("\n── Installing CRAN packages ──")
missing_cran <- cran_packages[!cran_packages %in% installed.packages()[, "Package"]]
if (length(missing_cran) > 0) {
  message("Installing: ", paste(missing_cran, collapse = ", "))
  install.packages(
    missing_cran,
    repos      = "https://cloud.r-project.org",
    quiet      = TRUE,
    ask        = FALSE,
    Ncpus      = max(1L, parallel::detectCores() - 1L)
  )
} else {
  message("All CRAN packages already installed.")
}

# ── 2. Bioconductor packages ──────────────────────────────────────────────────

bioc_packages <- c(
  "BiocGenerics",      # base Bioconductor generics
  "SingleCellExperiment", # SCE class (bridge between Python and R)
  "SingleR",           # automated cell-type annotation
  "celldex",           # reference datasets for SingleR (HPCA, Blueprint etc.)
  "scater",            # QC and visualisation utilities
  "Seurat"             # used by ScType scoring
)

message("\n── Installing Bioconductor packages ──")

# Seurat ≥5 lives on CRAN; older versions were on Bioc — check both
missing_bioc <- bioc_packages[!bioc_packages %in% installed.packages()[, "Package"]]

if (length(missing_bioc) > 0) {
  message("Installing: ", paste(missing_bioc, collapse = ", "))

  # Separate Seurat (CRAN) from pure Bioc packages
  seurat_needed <- "Seurat" %in% missing_bioc
  bioc_needed   <- setdiff(missing_bioc, "Seurat")

  if (seurat_needed) {
    install.packages("Seurat",
                     repos  = "https://cloud.r-project.org",
                     quiet  = TRUE,
                     ask    = FALSE,
                     Ncpus  = max(1L, parallel::detectCores() - 1L))
  }

  if (length(bioc_needed) > 0) {
    BiocManager::install(
      bioc_needed,
      ask     = FALSE,
      update  = FALSE,
      version = BiocManager::version()
    )
  }
} else {
  message("All Bioconductor packages already installed.")
}

# ── 3. ScType (GitHub) ────────────────────────────────────────────────────────
# ScType is not on CRAN or Bioconductor — sourced at runtime from GitHub.
# The annotation notebook loads it with:
#   source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/...")
# No installation needed, but we verify internet connectivity here.

message("\n── Checking ScType GitHub access ──")
tryCatch({
  con <- url("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")
  open(con); close(con)
  message("✓ ScType GitHub URL reachable (will be sourced at runtime).")
}, error = function(e) {
  message("✗ Cannot reach ScType GitHub URL: ", conditionMessage(e))
  message("  If your machine is offline, download the two ScType files manually:")
  message("  https://github.com/IanevskiAleksandr/sc-type/blob/master/R/gene_sets_prepare.R")
  message("  https://github.com/IanevskiAleksandr/sc-type/blob/master/R/sctype_score_.R")
  message("  Place them in scripts/utils/ and update the source() calls in notebook 03.")
})

# ── 4. Verify all packages load ───────────────────────────────────────────────

message("\n── Verifying package loading ──")
all_ok <- TRUE

all_packages <- c(cran_packages, bioc_packages)
all_packages <- all_packages[all_packages != "BiocManager"]   # skip meta-pkg

for (pkg in all_packages) {
  ok <- tryCatch({
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
    TRUE
  }, error = function(e) FALSE)

  status <- if (ok) "✓" else "✗"
  ver    <- if (ok) as.character(packageVersion(pkg)) else "FAILED"
  message(sprintf("  %s  %-25s %s", status, pkg, ver))
  if (!ok) all_ok <- FALSE
}

# ── 5. Summary ────────────────────────────────────────────────────────────────

message("\n", strrep("=", 52))
if (all_ok) {
  message("  All R packages installed and verified.")
  message("  You can now run notebooks 02 and 03.")
} else {
  message("  Some packages failed — see ✗ lines above.")
  message("  Try re-running this script or installing failed")
  message("  packages manually in an R session.")
}
message(strrep("=", 52))
