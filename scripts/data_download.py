"""
data_download.py
================
Downloads the scRNA-seq dataset GSE166635 from NCBI GEO and prepares
the data/ folder structure expected by the pipeline.

Usage
-----
    python scripts/data_download.py

What it does
------------
1. Detects the repo root automatically (the folder containing this script's
   parent — no hardcoded paths needed).
2. Creates data/raw/HCC1/ and data/raw/HCC2/ inside the repo.
3. Downloads GSE166635_RAW.tar from NCBI GEO FTP (~204 MB).
4. Extracts and organises the MTX triplets for HCC1 and HCC2.
5. Verifies the expected files are present before finishing.

The script is safe to re-run — it skips downloads that already exist.
"""

import os
import sys
import tarfile
import hashlib
import urllib.request
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Detect repo root from this script's location
#     Works regardless of where the user runs the script from.
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent   # scripts/
REPO_ROOT  = SCRIPT_DIR.parent                 # repo root

DATA_DIR   = REPO_ROOT / "data"
RAW_DIR    = DATA_DIR  / "raw"
PROC_DIR   = DATA_DIR  / "processed"

print(f"Repo root  : {REPO_ROOT}")
print(f"Data dir   : {DATA_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GEO download configuration
# ─────────────────────────────────────────────────────────────────────────────

GEO_ACCESSION = "GSE166635"

# Primary: NCBI GEO FTP
GEO_FTP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE166nnn/"
    "GSE166635/suppl/GSE166635_RAW.tar"
)

# Fallback: NCBI GEO HTTPS
GEO_HTTPS_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/"
    "?acc=GSE166635&format=file"
)

TAR_FILE = DATA_DIR / "GSE166635_RAW.tar"

# Expected MTX file triplets per sample
EXPECTED_FILES = {
    "HCC1": ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"],
    "HCC2": ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"],
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def make_dirs():
    """Create the full data folder structure."""
    for d in [RAW_DIR / "HCC1", RAW_DIR / "HCC2", PROC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("Folder structure ready.")


def download_with_progress(url: str, dest: Path):
    """
    Download a file with a simple progress bar.
    Shows MB downloaded and percentage if Content-Length is available.
    """
    print(f"\nDownloading:\n  {url}\n  → {dest}")

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb  = downloaded / 1_048_576
            tot = total_size / 1_048_576
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  {mb:.1f}/{tot:.1f} MB",
                  end="", flush=True)
        else:
            mb = downloaded / 1_048_576
            print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    print()  # newline after progress bar


def download_geo_tar():
    """Download GSE166635_RAW.tar, trying FTP then HTTPS fallback."""
    if TAR_FILE.exists():
        size_mb = TAR_FILE.stat().st_size / 1_048_576
        print(f"Archive already present ({size_mb:.0f} MB) — skipping download.")
        return

    urls = [GEO_FTP_URL, GEO_HTTPS_URL]
    for url in urls:
        try:
            download_with_progress(url, TAR_FILE)
            print(f"Download complete: {TAR_FILE.name} "
                  f"({TAR_FILE.stat().st_size / 1_048_576:.0f} MB)")
            return
        except Exception as e:
            print(f"\nFailed ({e}) — trying next URL...")
            if TAR_FILE.exists():
                TAR_FILE.unlink()

    print("\n✗ All download URLs failed.")
    print("  Manual download instructions:")
    print(f"  1. Open https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_ACCESSION}")
    print(f"  2. Download GSE166635_RAW.tar (Supplementary file)")
    print(f"  3. Place it at: {TAR_FILE}")
    sys.exit(1)


def extract_and_organise():
    """
    Extract GSE166635_RAW.tar and place files into data/raw/HCC1/ and HCC2/.

    The archive contains files named like:
        GSM5076749_HCC1_barcodes.tsv.gz
        GSM5076749_HCC1_features.tsv.gz
        GSM5076749_HCC1_matrix.mtx.gz
        GSM5076750_HCC2_barcodes.tsv.gz
        ...
    We strip the GSM prefix and sort into HCC1/ and HCC2/ subdirectories.
    """
    all_present = all(
        (RAW_DIR / sample / fname).exists()
        for sample, fnames in EXPECTED_FILES.items()
        for fname in fnames
    )
    if all_present:
        print("MTX files already extracted — skipping extraction.")
        return

    print(f"\nExtracting {TAR_FILE.name} ...")
    with tarfile.open(TAR_FILE, "r") as tar:
        members = tar.getmembers()
        print(f"  Archive contains {len(members)} files.")
        for member in members:
            name = Path(member.name).name   # strip any directory prefix
            # Determine sample (HCC1 or HCC2)
            sample = None
            for s in ["HCC1", "HCC2"]:
                if s in name:
                    sample = s
                    break
            if sample is None:
                continue

            # Determine file type (barcodes / features / matrix / genes)
            dest_name = None
            if "barcodes" in name.lower():
                dest_name = "barcodes.tsv.gz"
            elif "features" in name.lower() or "genes" in name.lower():
                dest_name = "features.tsv.gz"
            elif "matrix" in name.lower():
                dest_name = "matrix.mtx.gz"

            if dest_name is None:
                continue

            dest_path = RAW_DIR / sample / dest_name
            if dest_path.exists():
                continue

            print(f"  {name}  →  {sample}/{dest_name}")
            member.name = dest_name          # rename on extraction
            tar.extract(member, path=RAW_DIR / sample)

    print("Extraction complete.")


def verify():
    """Check all expected files are present and non-empty."""
    print("\nVerifying files ...")
    all_ok = True
    for sample, fnames in EXPECTED_FILES.items():
        for fname in fnames:
            fpath = RAW_DIR / sample / fname
            if fpath.exists() and fpath.stat().st_size > 0:
                kb = fpath.stat().st_size // 1024
                print(f"  ✓  {sample}/{fname}  ({kb} KB)")
            else:
                print(f"  ✗  {sample}/{fname}  MISSING or EMPTY")
                all_ok = False

    if all_ok:
        print("\n✓ All files verified. You are ready to run the notebooks.")
        print(f"\n  DATA_DIR  = {DATA_DIR}")
        print(f"  RAW_DIR   = {RAW_DIR}")
        print(f"  PROC_DIR  = {PROC_DIR}")
    else:
        print("\n✗ Some files are missing. Please re-run this script.")
        sys.exit(1)


def cleanup_tar(keep: bool = False):
    """Optionally remove the .tar archive after successful extraction."""
    if not keep and TAR_FILE.exists():
        size_mb = TAR_FILE.stat().st_size / 1_048_576
        TAR_FILE.unlink()
        print(f"Removed archive ({size_mb:.0f} MB freed).")


def write_paths_config():
    """
    Write a paths.py file into the repo root so every notebook and script
    can import project paths without any hardcoding.

    Any script in the repo can then do:
        from paths import REPO_ROOT, DATA_DIR, RAW_DIR, PROC_DIR, RESULTS_DIR
    """
    config_path = REPO_ROOT / "paths.py"
    content = f'''\
"""
paths.py  —  auto-generated by scripts/data_download.py
=========================================================
Central path registry for the hcc-drug-discovery project.

Import from any notebook or script:
    from paths import REPO_ROOT, RAW_DIR, PROC_DIR, RESULTS_DIR, FIGURES_DIR

All paths are absolute and derived from this file\'s location,
so the repo can be cloned anywhere without editing paths.
"""

from pathlib import Path

# Repo root = the folder containing this file
REPO_ROOT   = Path(__file__).resolve().parent

DATA_DIR    = REPO_ROOT / "data"
RAW_DIR     = DATA_DIR  / "raw"
PROC_DIR    = DATA_DIR  / "processed"

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR  = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"

MODELS_DIR  = REPO_ROOT / "models"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Convenience: ensure output dirs exist when paths.py is imported
for _d in [PROC_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
'''
    config_path.write_text(content)
    print(f"\nCreated: {config_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and prepare GSE166635 data for the HCC pipeline."
    )
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep the .tar archive after extraction (default: delete it)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download — use if you already placed the .tar file manually",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  HCC Drug Discovery — Data Setup")
    print("=" * 60)

    make_dirs()

    if not args.skip_download:
        download_geo_tar()

    extract_and_organise()
    verify()
    cleanup_tar(keep=args.keep_tar)
    write_paths_config()

    print("\n" + "=" * 60)
    print("  Setup complete. Next steps:")
    print("=" * 60)
    print("  1. Activate your conda environment:")
    print("       conda activate hcc_drug_discovery")
    print("  2. Open JupyterLab:")
    print("       jupyter lab")
    print("  3. Run the notebooks in order, starting with:")
    print("       notebooks/01_preprocessing.ipynb")
    print("=" * 60)
