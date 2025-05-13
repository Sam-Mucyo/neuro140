#!/usr/bin/env python3
"""
Merge Results Script for Parallel Topic Modeling

This script merges all checkpoint files from parallel processing into a single
comprehensive CSV file, ensuring all post IDs are unique and providing detailed statistics.
"""

import os
import pandas as pd
import glob
import argparse
import logging
from collections import Counter
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def merge_checkpoint_files(checkpoint_dir, output_file, verify_unique=True, stats_file=None):
    """
    Merge all checkpoint files in the given directory into a single CSV file.
    
    Args:
        checkpoint_dir: Directory containing checkpoint CSV files
        output_file: Path to save the merged CSV file
        verify_unique: Whether to verify and enforce unique post IDs
        stats_file: Optional path to save processing statistics as JSON
    
    Returns:
        DataFrame with merged results
    """
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.csv"))
    
    if not checkpoint_files:
        logger.error(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Load and merge all checkpoint files
    all_dfs = []
    file_stats = {}
    
    for file in checkpoint_files:
        try:
            df = pd.read_csv(file)
            file_stats[os.path.basename(file)] = {
                "rows": len(df),
                "worker": os.path.basename(file).split("_")[1] if "_" in os.path.basename(file) else "unknown"
            }
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} records from {os.path.basename(file)}")
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    
    if not all_dfs:
        logger.error("No valid checkpoint files could be loaded")
        return None
    
    # Concatenate all DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined data contains {len(merged_df)} total records")
    
    # Check for duplicate IDs
    if "id" in merged_df.columns:
        id_counts = Counter(merged_df["id"].astype(str))
        duplicate_ids = {id: count for id, count in id_counts.items() if count > 1}
        
        if duplicate_ids:
            logger.warning(f"Found {len(duplicate_ids)} duplicate post IDs")
            for id, count in list(duplicate_ids.items())[:10]:  # Show first 10 duplicates
                logger.warning(f"ID {id} appears {count} times")
            
            if verify_unique:
                logger.info("Keeping only the first occurrence of each duplicate ID")
                merged_df = merged_df.drop_duplicates(subset=["id"], keep="first")
    else:
        logger.warning("No 'id' column found in the data")
    
    # Save merged results
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(merged_df)} merged records to {output_file}")
    
    # Compile statistics
    stats = {
        "total_files": len(checkpoint_files),
        "total_records_before_deduplication": len(pd.concat(all_dfs, ignore_index=True)),
        "total_records_after_deduplication": len(merged_df),
        "duplicate_ids_count": len(duplicate_ids) if "id" in merged_df.columns and duplicate_ids else 0,
        "file_stats": file_stats
    }
    
    # Save statistics if requested
    if stats_file:
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved processing statistics to {stats_file}")
    
    return merged_df

def analyze_results(df, output_dir):
    """
    Perform basic analysis on the merged results.
    
    Args:
        df: DataFrame with merged results
        output_dir: Directory to save analysis results
    """
    if df is None or len(df) == 0:
        logger.error("No data to analyze")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Basic statistics
    stats = {
        "total_records": len(df),
        "columns": list(df.columns)
    }
    
    # Analyze themes if available
    if "themes" in df.columns:
        # Convert string representation of lists to actual lists
        try:
            themes = df["themes"].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else x)
            all_themes = []
            for theme_list in themes:
                if isinstance(theme_list, list):
                    all_themes.extend(theme_list)
            
            theme_counts = Counter(all_themes)
            stats["top_themes"] = dict(theme_counts.most_common(20))
            
            # Save theme counts
            theme_df = pd.DataFrame({
                "theme": list(theme_counts.keys()),
                "count": list(theme_counts.values())
            }).sort_values("count", ascending=False)
            
            theme_df.to_csv(os.path.join(output_dir, "theme_counts.csv"), index=False)
            logger.info(f"Saved theme counts to {os.path.join(output_dir, 'theme_counts.csv')}")
        except Exception as e:
            logger.error(f"Error analyzing themes: {str(e)}")
    
    # Analyze emotional tone if available
    if "emotional_tone" in df.columns:
        tone_counts = Counter(df["emotional_tone"])
        stats["emotional_tone_counts"] = dict(tone_counts)
        
        # Save tone counts
        tone_df = pd.DataFrame({
            "tone": list(tone_counts.keys()),
            "count": list(tone_counts.values())
        }).sort_values("count", ascending=False)
        
        tone_df.to_csv(os.path.join(output_dir, "emotional_tone_counts.csv"), index=False)
        logger.info(f"Saved emotional tone counts to {os.path.join(output_dir, 'emotional_tone_counts.csv')}")
    
    # Save statistics
    with open(os.path.join(output_dir, "analysis_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved analysis statistics to {os.path.join(output_dir, 'analysis_stats.json')}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Merge parallel processing checkpoint files")
    parser.add_argument("--checkpoint-dir", default="../results/parallel/checkpoints",
                      help="Directory containing checkpoint files")
    parser.add_argument("--output-file", default="../results/parallel/merged_results.csv",
                      help="Path to save the merged CSV file")
    parser.add_argument("--stats-file", default="../results/parallel/merge_stats.json",
                      help="Path to save processing statistics")
    parser.add_argument("--analysis-dir", default="../results/parallel/analysis",
                      help="Directory to save analysis results")
    parser.add_argument("--no-verify", action="store_true",
                      help="Skip verification and deduplication of post IDs")
    args = parser.parse_args()
    
    # Merge checkpoint files
    merged_df = merge_checkpoint_files(
        args.checkpoint_dir,
        args.output_file,
        verify_unique=not args.no_verify,
        stats_file=args.stats_file
    )
    
    # Analyze results
    if merged_df is not None:
        analyze_results(merged_df, args.analysis_dir)

if __name__ == "__main__":
    main()
