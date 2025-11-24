"""
Build a comprehensive dataset by combining all available WiLI data for English, French, and Spanish.
This creates a larger dataset than the default subset by using all available samples from WiLI-2018.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich import print as rprint
from sklearn.model_selection import train_test_split

from src.config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR, TARGET_LANGUAGES

WILI_DATA_DIR = RAW_DATA_DIR / "wili-2018"


def load_all_wili_data() -> pd.DataFrame:
    """Load all WiLI data for the target languages."""
    if not WILI_DATA_DIR.exists():
        raise FileNotFoundError(
            f"WiLI data not found at {WILI_DATA_DIR}. "
            "Run 'python -m src.data.wili_downloader download' first."
        )
    
    # Load train and test data
    train_texts = (WILI_DATA_DIR / "x_train.txt").read_text(encoding="utf-8").splitlines()
    train_labels = (WILI_DATA_DIR / "y_train.txt").read_text(encoding="utf-8").splitlines()
    test_texts = (WILI_DATA_DIR / "x_test.txt").read_text(encoding="utf-8").splitlines()
    test_labels = (WILI_DATA_DIR / "y_test.txt").read_text(encoding="utf-8").splitlines()
    
    # Combine train and test
    all_texts = train_texts + test_texts
    all_labels = train_labels + test_labels
    
    # Create DataFrame
    df = pd.DataFrame({"text": all_texts, "language": all_labels})
    
    # Filter to target languages
    df = df[df["language"].isin(TARGET_LANGUAGES)]
    
    return df


def create_comprehensive_dataset(
    output_path: Path | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    Create a comprehensive dataset using all available WiLI data for the target languages.
    
    Args:
        output_path: Where to save the CSV (defaults to PROCESSED_DATA_DIR/comprehensive_dataset.csv)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    rprint("[bold cyan]Loading all WiLI data for target languages...")
    
    # Load all data
    df = load_all_wili_data()
    
    rprint(f"[green]✓ Loaded {len(df)} total samples")
    rprint(f"[cyan]Samples per language:")
    for lang, count in df["language"].value_counts().items():
        rprint(f"  {lang}: {count}")
    
    # Create stratified splits
    rprint(f"\n[bold cyan]Creating stratified splits ({train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test)...")
    
    # First split: train vs temp (val+test)
    temp_ratio = val_ratio + test_ratio
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["language"],
        random_state=42,
    )
    
    # Second split: val vs test
    val_size = val_ratio / temp_ratio
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1 - val_size,
        stratify=df_temp["language"],
        random_state=42,
    )
    
    # Add split column
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    
    # Combine all splits
    df_final = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # Shuffle within each split
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "comprehensive_dataset.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    # Save metadata
    metadata = {
        "total_samples": len(df_final),
        "samples_per_language": df_final["language"].value_counts().to_dict(),
        "samples_per_split": df_final["split"].value_counts().to_dict(),
        "samples_per_language_per_split": {
            split: df_final[df_final["split"] == split]["language"].value_counts().to_dict()
            for split in ["train", "val", "test"]
        },
        "source": "WiLI-2018 (all available data)",
        "languages": TARGET_LANGUAGES,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "created_at": pd.Timestamp.now().isoformat(),
    }
    metadata_path = output_path.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    rprint(f"\n[bold green]✅ Comprehensive dataset created successfully!")
    rprint(f"[cyan]Saved to: {output_path}")
    rprint(f"[cyan]Total samples: {len(df_final)}")
    rprint(f"[cyan]Samples per split:")
    for split, count in df_final["split"].value_counts().items():
        rprint(f"  {split}: {count}")
    rprint(f"[cyan]Samples per language:")
    for lang, count in df_final["language"].value_counts().items():
        rprint(f"  {lang}: {count}")
    
    return df_final


def main():
    parser = argparse.ArgumentParser(
        description="Build comprehensive dataset from all available WiLI data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/processed/comprehensive_dataset.csv)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    create_comprehensive_dataset(
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
