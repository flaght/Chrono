"""Utility functions for factor ID generation and path handling"""
import hashlib
import os
from pathlib import Path
from typing import Optional


def create_id(original: str, digit: int = 16) -> str:
    """
    Create an ID from a hexadecimal string, truncating to specified digits.
    
    Args:
        original: Hexadecimal string (typically MD5 hash)
        digit: Number of digits to return (default 16)
    
    Returns:
        Truncated hexadecimal string
    """
    return original[:digit] if len(original) >= digit else original.ljust(digit, '0')


def create_name_id(expression: str, digit: int = 16) -> str:
    """
    Generate a factor ID from an expression using MD5 hash.
    
    Args:
        expression: Factor expression/formula
        digit: Number of digits for the ID (default 16)
    
    Returns:
        Factor ID as hexadecimal string
    """
    m = hashlib.md5()
    m.update(bytes(expression, encoding='UTF-8'))
    return create_id(original=m.hexdigest(), digit=digit)


def build_factor_image_path(base_dir: str, method: str, instruments: str, 
                            task_id: str, period: int, category: str, 
                            source: str, factor_id: Optional[str] = None) -> str:
    """
    Build the path to a factor's image (comparison_plot.png for category 'p', 
    evaluation_plot.png for category 'd').
    If factor_id is provided, builds the direct path. If factor_id is None or empty,
    falls back to directory scanning to find the first available image.
    
    Args:
        base_dir: Base directory for records
        method: Method name (e.g., 'cicso0')
        instruments: Instruments code (e.g., 'ims')
        task_id: Task ID (e.g., '200037')
        period: Period in hours (e.g., 15)
        category: Factor category ('p' or 'd')
        source: Source identifier (e.g., '202510225')
        factor_id: Factor ID (e.g., '10111666'). If None or empty, uses scanning mode.
    
    Returns:
        Full path to image file, or empty string if not found (in scanning mode)
    """
    # If factor_id is not provided, use scanning mode
    if not factor_id:
        return find_factor_image_by_scanning(
            base_dir=base_dir,
            method=method,
            instruments=instruments,
            task_id=task_id,
            period=period,
            category=category,
            source=source
        )
    
    # Direct path construction (original behavior)
    # Determine image filename based on category
    if category == 'd':
        image_filename = 'evaluation_plot.png'
    else:
        image_filename = 'comparison_plot.png'
    
    if category == 'p':
        # Category 'p': records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}/{source}/{factor_id}/comparison_plot.png
        path = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source) / str(factor_id) / image_filename
    elif category == 'd':
        # Category 'd': records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}/d{source}/{factor_id}/evaluation_plot.png
        path = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / f'd{source}' / str(factor_id) / image_filename
    else:
        # For category 'f' or others, use the same pattern as 'p' for now
        path = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source) / str(factor_id) / image_filename
    
    return str(path)


def build_performance_summary_path(base_dir: str, method: str, instruments: str,
                                   task_id: str, period: int, category: str,
                                   source: str, factor_id: Optional[str] = None) -> str:
    """
    Build the path to a factor's performance summary text file.
    If factor_id is provided, builds the direct path. If factor_id is None or empty,
    falls back to directory scanning to find the first available summary.
    
    Args:
        base_dir: Base directory for records
        method: Method name
        instruments: Instruments code
        task_id: Task ID
        period: Period in hours
        category: Factor category ('p' or 'd')
        source: Source identifier
        factor_id: Factor ID. If None or empty, uses scanning mode.
    
    Returns:
        Full path to performance_summary.txt, or empty string if not found (in scanning mode)
    """
    # If factor_id is not provided, use scanning mode
    if not factor_id:
        return find_factor_summary_by_scanning(
            base_dir=base_dir,
            method=method,
            instruments=instruments,
            task_id=task_id,
            period=period,
            category=category,
            source=source
        )
    
    # Direct path construction (original behavior)
    if category == 'p':
        path = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source) / str(factor_id) / 'performance_summary.txt'
    elif category == 'd':
        path = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / f'd{source}' / str(factor_id) / 'performance_summary.txt'
    else:
        path = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source) / str(factor_id) / 'performance_summary.txt'
    
    return str(path)


def file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(file_path) and os.path.isfile(file_path)


def find_factor_image_by_scanning(base_dir: str, method: str, instruments: str,
                                   task_id: str, period: int, category: str,
                                   source: str) -> str:
    """
    Scan source directory to find factor image (comparison_plot.png for category 'p', 
    evaluation_plot.png for category 'd').
    Since we don't know the exact factor_id (directory name), we scan all subdirectories.
    Returns the first found image (may not be exact match, but allows images to display).
    
    Args:
        base_dir: Base directory for records
        method: Method name
        instruments: Instruments code
        task_id: Task ID
        period: Period in hours
        category: Factor category ('p' or 'd')
        source: Source identifier
    
    Returns:
        Full path to image if found, empty string otherwise
    """
    # Build base source directory path
    if category == 'p':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    elif category == 'd':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / f'd{source}'
    else:
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    
    if not source_dir.exists() or not source_dir.is_dir():
        return ""
    
    # Determine image filename based on category
    if category == 'd':
        image_filename = 'evaluation_plot.png'
    else:
        image_filename = 'comparison_plot.png'
    
    # Scan all subdirectories for the appropriate image file
    # Sort directories to ensure consistent ordering
    subdirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    for subdir in subdirs:
        image_path = subdir / image_filename
        if image_path.exists() and image_path.is_file():
            return str(image_path)
    
    return ""


def normalize_formula(formula: str) -> str:
    """
    Normalize a formula by standardizing quotes and whitespace for comparison.
    
    Args:
        formula: Formula string
    
    Returns:
        Normalized formula string
    """
    import re
    # Remove extra whitespace
    formula = ' '.join(formula.split())
    # Replace single quotes with double quotes for consistency
    formula = formula.replace("'", '"')
    return formula


def find_factor_by_formula_matching(base_dir: str, method: str, instruments: str,
                                     task_id: str, period: int, category: str,
                                     source: str, formula: str) -> tuple:
    """
    Find factor image and summary by matching formula with Expression in performance_summary.txt.
    This is the most accurate way to find the correct factor files.
    
    Args:
        base_dir: Base directory for records
        method: Method name
        instruments: Instruments code
        task_id: Task ID
        period: Period in hours
        category: Factor category ('p' or 'd')
        source: Source identifier
        formula: Factor formula to match
    
    Returns:
        Tuple of (image_path, summary_path), both empty strings if not found
    """
    # Build base source directory path
    if category == 'p':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    elif category == 'd':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / f'd{source}'
    else:
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    
    if not source_dir.exists() or not source_dir.is_dir():
        return "", ""
    
    # Normalize the target formula
    normalized_target = normalize_formula(formula)
    
    # Determine image filename based on category
    if category == 'd':
        image_filename = 'evaluation_plot.png'
    else:
        image_filename = 'comparison_plot.png'
    
    # Scan all subdirectories and match by Expression
    subdirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    for subdir in subdirs:
        summary_path = subdir / 'performance_summary.txt'
        if summary_path.exists() and summary_path.is_file():
            try:
                # Read the summary file to extract Expression
                with open(summary_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('Expression:'):
                            # Extract expression after "Expression:"
                            expression = line.split('Expression:', 1)[1].strip()
                            normalized_expression = normalize_formula(expression)
                            
                            # Check if they match
                            if normalized_expression == normalized_target:
                                # Found matching factor, return paths
                                image_path = subdir / image_filename
                                if image_path.exists() and image_path.is_file():
                                    return str(image_path), str(summary_path)
                                else:
                                    # Summary found but image missing
                                    return "", str(summary_path)
                            break
            except Exception:
                # If reading fails, continue to next directory
                continue
    
    # If no exact match found, return empty strings
    return "", ""


def find_factor_summary_by_scanning(base_dir: str, method: str, instruments: str,
                                     task_id: str, period: int, category: str,
                                     source: str) -> str:
    """
    Scan source directory to find performance_summary.txt.
    Returns the first found summary (may not be exact match, but allows summaries to display).
    
    Args:
        base_dir: Base directory for records
        method: Method name
        instruments: Instruments code
        task_id: Task ID
        period: Period in hours
        category: Factor category ('p' or 'd')
        source: Source identifier
    
    Returns:
        Full path to performance_summary.txt if found, empty string otherwise
    """
    # Build base source directory path
    if category == 'p':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    elif category == 'd':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / f'd{source}'
    else:
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    
    if not source_dir.exists() or not source_dir.is_dir():
        return ""
    
    # Scan all subdirectories for performance_summary.txt
    # Sort directories to ensure consistent ordering
    subdirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    for subdir in subdirs:
        summary_path = subdir / 'performance_summary.txt'
        if summary_path.exists() and summary_path.is_file():
            return str(summary_path)
    
    return ""

