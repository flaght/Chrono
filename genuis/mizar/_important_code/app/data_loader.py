"""Data loading module for factor performance data"""
import pandas as pd
import os
from typing import Dict, List, Optional
from .config import Config
from .utils import (create_name_id, build_factor_image_path, build_performance_summary_path, 
                    file_exists, find_factor_image_by_scanning, find_factor_summary_by_scanning,
                    find_factor_by_formula_matching)


def load_factor_data(config: Config) -> pd.DataFrame:
    """
    Load factor data from final.csv.
    
    Args:
        config: Configuration object
    
    Returns:
        DataFrame with factor data and additional metadata
    """
    csv_path = config.csv_path
    
    # Resolve path
    if not os.path.isabs(csv_path):
        # If csv_path already starts with 'records/', use it as is with base_path
        if csv_path.startswith('records/'):
            # Remove 'records/' prefix and join with base_path
            csv_path = os.path.join(config.base_path, csv_path.replace('records/', '', 1))
        else:
            # Join directly with base_path
            csv_path = os.path.join(config.base_path, csv_path)
    
    # Normalize the path
    csv_path = os.path.abspath(csv_path)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = ['formula', 'direction', 'source', 'level', 'category', 'score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    # Generate factor IDs for each row (for display purposes)
    df['factor_id'] = df['formula'].apply(lambda x: create_name_id(x))
    
    # Build source directory paths (we'll scan these directories when displaying details)
    # Store source directory path instead of specific image/summary paths
    # since we don't know the exact factor_id (directory name) mapping
    df['source_dir'] = df.apply(
        lambda row: _get_source_directory(
            base_dir=config.base_path,
            method=config.method,
            instruments=config.instruments,
            task_id=config.task_id,
            period=config.period,
            category=str(row['category']),
            source=str(row['source'])
        ),
        axis=1
    )
    
    # Try to find image and summary by scanning (this may not find the exact match,
    # but will at least check if any images/summaries exist in the source directory)
    df['image_path'] = df.apply(
        lambda row: find_factor_image_by_scanning(
            base_dir=config.base_path,
            method=config.method,
            instruments=config.instruments,
            task_id=config.task_id,
            period=config.period,
            category=str(row['category']),
            source=str(row['source'])
        ) if pd.notna(row['source_dir']) else "",
        axis=1
    )
    
    df['summary_path'] = df.apply(
        lambda row: find_factor_summary_by_scanning(
            base_dir=config.base_path,
            method=config.method,
            instruments=config.instruments,
            task_id=config.task_id,
            period=config.period,
            category=str(row['category']),
            source=str(row['source'])
        ) if pd.notna(row['source_dir']) else "",
        axis=1
    )
    
    # Check if files exist
    df['image_exists'] = df['image_path'].apply(lambda x: bool(x) and file_exists(x))
    df['summary_exists'] = df['summary_path'].apply(lambda x: bool(x) and file_exists(x))
    
    # Convert types
    df['source'] = df['source'].astype(str)
    df['category'] = df['category'].astype(str)
    df['direction'] = df['direction'].astype(int)
    df['level'] = df['level'].astype(int)
    df['score'] = df['score'].astype(float)
    
    return df


def filter_factors(df: pd.DataFrame, search_term: Optional[str] = None,
                   category: Optional[str] = None,
                   min_score: Optional[float] = None,
                   max_score: Optional[float] = None,
                   level: Optional[int] = None,
                   direction: Optional[int] = None,
                   source: Optional[str] = None,
                   has_image: Optional[bool] = None) -> pd.DataFrame:
    """
    Filter factors based on various criteria.
    
    Args:
        df: Factor DataFrame
        search_term: Search term for formula, detail, or desc fields
        category: Filter by category
        min_score: Minimum score
        max_score: Maximum score
        level: Filter by level
        direction: Filter by direction (-1, 1)
        source: Filter by source
        has_image: Filter by whether image exists
    
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    # Search term
    if search_term:
        search_lower = search_term.lower()
        mask = (
            filtered['formula'].str.lower().str.contains(search_lower, na=False) |
            filtered.get('detail', pd.Series()).str.lower().str.contains(search_lower, na=False) |
            filtered.get('desc', pd.Series()).str.lower().str.contains(search_lower, na=False)
        )
        filtered = filtered[mask]
    
    # Category filter
    if category:
        filtered = filtered[filtered['category'] == category]
    
    # Score range
    if min_score is not None:
        filtered = filtered[filtered['score'] >= min_score]
    if max_score is not None:
        filtered = filtered[filtered['score'] <= max_score]
    
    # Level filter
    if level is not None:
        filtered = filtered[filtered['level'] == level]
    
    # Direction filter
    if direction is not None:
        filtered = filtered[filtered['direction'] == direction]
    
    # Source filter
    if source:
        filtered = filtered[filtered['source'] == str(source)]
    
    # Image existence filter
    if has_image is not None:
        filtered = filtered[filtered['image_exists'] == has_image]
    
    return filtered


def sort_factors(df: pd.DataFrame, sort_by: str = 'score', ascending: bool = False) -> pd.DataFrame:
    """
    Sort factors DataFrame.
    
    Args:
        df: Factor DataFrame
        sort_by: Column name to sort by
        ascending: Whether to sort in ascending order
    
    Returns:
        Sorted DataFrame
    """
    if sort_by not in df.columns:
        return df
    
    return df.sort_values(by=sort_by, ascending=ascending)


def _get_source_directory(base_dir: str, method: str, instruments: str,
                          task_id: str, period: int, category: str,
                          source: str) -> str:
    """Helper function to build source directory path."""
    from pathlib import Path
    if category == 'p':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    elif category == 'd':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / f'd{source}'
    else:
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    return str(source_dir) if source_dir.exists() else ""


def get_factor_image_and_summary(config: Config, category: str, source: str, formula: str) -> tuple:
    """
    Get image and summary paths for a factor by matching formula with Expression in performance_summary.txt.
    This ensures we get the correct files for the specific factor.
    
    Args:
        config: Configuration object
        category: Factor category
        source: Source identifier
        formula: Factor formula to match
    
    Returns:
        Tuple of (image_path, summary_path)
    """
    # First try to find by formula matching (most accurate)
    image_path, summary_path = find_factor_by_formula_matching(
        base_dir=config.base_path,
        method=config.method,
        instruments=config.instruments,
        task_id=config.task_id,
        period=config.period,
        category=category,
        source=source,
        formula=formula
    )
    
    # If formula matching failed, fall back to scanning (for backward compatibility)
    if not image_path:
        image_path = find_factor_image_by_scanning(
            base_dir=config.base_path,
            method=config.method,
            instruments=config.instruments,
            task_id=config.task_id,
            period=config.period,
            category=category,
            source=source
        )
    
    if not summary_path:
        summary_path = find_factor_summary_by_scanning(
            base_dir=config.base_path,
            method=config.method,
            instruments=config.instruments,
            task_id=config.task_id,
            period=config.period,
            category=category,
            source=source
        )
    
    return image_path, summary_path


def get_performance_summary(summary_path: str) -> Optional[str]:
    """
    Read performance summary from file.
    
    Args:
        summary_path: Path to performance_summary.txt
    
    Returns:
        Summary text or None if file doesn't exist
    """
    if file_exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading summary: {str(e)}"
    return None

