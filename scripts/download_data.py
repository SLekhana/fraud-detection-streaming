"""
Download IEEE-CIS Fraud Detection dataset from Kaggle.

Prerequisites:
    1. Create a Kaggle account at https://www.kaggle.com
    2. Go to Account → API → Create New Token → downloads kaggle.json
    3. Place kaggle.json at ~/.kaggle/kaggle.json
    4. Run: python scripts/download_data.py

Dataset: https://www.kaggle.com/competitions/ieee-fraud-detection
Files:
    - train_transaction.csv  (590,540 rows × 394 cols)
    - train_identity.csv     (144,233 rows × 41 cols)
    - test_transaction.csv
    - test_identity.csv
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path


def check_kaggle_credentials() -> bool:
    """Check if kaggle.json exists."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ kaggle.json not found at ~/.kaggle/kaggle.json")
        print()
        print("Setup instructions:")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Scroll to 'API' section → 'Create New Token'")
        print("  3. This downloads kaggle.json")
        print("  4. Run: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("  5. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("  6. Re-run this script")
        return False
    return True


def download_dataset(output_dir: str = "data/") -> None:
    """Download IEEE-CIS dataset using kaggle API."""
    if not check_kaggle_credentials():
        sys.exit(1)

    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle CLI ...")
        os.system("pip install kaggle --quiet")
        import kaggle  # noqa: F401

    os.makedirs(output_dir, exist_ok=True)

    print("📥 Downloading IEEE-CIS Fraud Detection dataset ...")
    print("   (590K transactions, ~500MB) — this takes 2-5 minutes")
    print()

    os.system(
        f"kaggle competitions download -c ieee-fraud-detection -p {output_dir}"
    )

    # Unzip
    zip_path = os.path.join(output_dir, "ieee-fraud-detection.zip")
    if os.path.exists(zip_path):
        print(f"\n📦 Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        os.remove(zip_path)
        print("✅ Extraction complete")

    # Verify
    expected = ["train_transaction.csv", "train_identity.csv"]
    missing = [f for f in expected if not os.path.exists(os.path.join(output_dir, f))]
    if missing:
        print(f"⚠️  Missing files: {missing}")
        print("   You may need to accept the competition rules at:")
        print("   https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
    else:
        sizes = {
            f: f"{os.path.getsize(os.path.join(output_dir, f)) / 1e6:.1f} MB"
            for f in expected
        }
        print("\n✅ Dataset ready:")
        for fname, size in sizes.items():
            print(f"   {output_dir}/{fname}  ({size})")
        print("\nNext step:")
        print("   python scripts/train.py --data-dir data/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/", help="Where to save data")
    args = parser.parse_args()
    download_dataset(args.output_dir)
