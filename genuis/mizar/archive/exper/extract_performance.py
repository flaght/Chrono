#!/usr/bin/env python3
"""
Extract IM绩效 and IC绩效 from performance_summary.txt files
and merge with final.csv data.
"""

import csv
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List


def normalize_formula(formula: str) -> str:
    """Normalize a formula by standardizing quotes and whitespace for comparison."""
    # Remove extra whitespace
    formula = ' '.join(formula.split())
    # Replace single quotes with double quotes for consistency
    formula = formula.replace("'", '"')
    return formula


def find_performance_summary_by_formula(
    base_dir: str,
    method: str,
    instruments: str,
    task_id: str,
    period: int,
    category: str,
    source: str,
    formula: str
) -> Optional[str]:
    """
    Find performance_summary.txt by matching formula with Expression in the file.
    
    Returns:
        Path to performance_summary.txt if found, None otherwise
    """
    # Build base source directory path
    if category == 'p':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    elif category == 'd':
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / f'd{source}'
    else:
        source_dir = Path(base_dir) / method / instruments / 'rulex' / str(task_id) / f'nxt1_ret_{period}h' / str(source)
    
    if not source_dir.exists() or not source_dir.is_dir():
        return None
    
    # Normalize the target formula
    normalized_target = normalize_formula(formula)
    
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
                                return str(summary_path)
                            break
            except Exception:
                # If reading fails, continue to next directory
                continue
    
    return None


def extract_performance_metrics(summary_path: str, extract_ic: bool = True) -> Tuple[dict, dict]:
    """
    Extract IM绩效 (ims) and IC绩效 (ics) from performance_summary.txt.
    
    Args:
        summary_path: Path to performance_summary.txt
        extract_ic: Whether to extract IC绩效 (ics). If False, only extract IM绩效 (ims).
    
    Returns:
        Tuple of (ims_metrics, ics_metrics) dictionaries
        ics_metrics will be empty dict if extract_ic is False
    """
    ims_metrics = {}
    ics_metrics = {}
    
    if not summary_path or not Path(summary_path).exists():
        return ims_metrics, ics_metrics
    
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the metric rows
        in_metrics = False
        for line in lines:
            line = line.strip()
            
            # Check if this is the header line
            if 'Metric' in line and 'ims' in line.lower() and 'ics' in line.lower():
                in_metrics = True
                continue
            
            # Skip separator lines
            if line.startswith('---') or not line:
                continue
            
            if in_metrics and line:
                # Parse metric line using | as delimiter
                # Format: Metric               | ims             | ics
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    metric_name = parts[0].strip()
                    ims_value = parts[1].strip() if len(parts) > 1 else ''
                    ics_value = parts[2].strip() if len(parts) > 2 else ''
                    
                    # Skip if metric_name is empty or is a separator
                    if not metric_name or metric_name.startswith('-'):
                        continue
                    
                    # Extract numeric values (remove % signs)
                    if ims_value:
                        ims_clean = ims_value.rstrip('%').strip()
                        try:
                            ims_metrics[metric_name] = float(ims_clean)
                        except ValueError:
                            ims_metrics[metric_name] = ims_value
                    
                    if extract_ic and ics_value:
                        ics_clean = ics_value.rstrip('%').strip()
                        try:
                            ics_metrics[metric_name] = float(ics_clean)
                        except ValueError:
                            ics_metrics[metric_name] = ics_value
                else:
                    # If we can't parse, we might have reached the end
                    if in_metrics:
                        break
                        
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
    
    return ims_metrics, ics_metrics


def extract_specific_metrics(summary_path: str, category: str = 'd') -> Tuple[float, float, float, float, float]:
    """
    Extract ic, Sharpe, Calmar, ICIR, and Factor Autocorr metrics from performance_summary.txt.

    Args:
        summary_path: Path to performance_summary.txt
        category: 'p' for ims/ics format, 'd' for single column format

    Returns:
        Tuple of (ic, sharpe, calmar, icir, factor_autocorr) values, or (None, None, None, None, None) if not found
    """
    ic = None
    sharpe = None
    calmar = None
    icir = None
    factor_autocorr = None

    if not summary_path or not Path(summary_path).exists():
        return None, None, None, None, None

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Determine which column to extract based on category
        # 'p' category uses ims column (strategy metrics), 'd' category uses single values
        target_column = 1 if category == 'p' else 0

        for line in lines:
            line = line.strip()

            # Skip header lines and separators
            if '---' in line or 'Metric' in line or not line:
                continue

            # Parse metric line using | as delimiter
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= target_column + 1:
                    metric_name = parts[0].strip()

                    # Extract IC Mean
                    if metric_name == 'IC Mean':
                        try:
                            ic = float(parts[target_column])
                        except (ValueError, IndexError):
                            pass

                    # Extract Sharpe Ratio (use Ann Sharpe Ratio if available, otherwise regular)
                    elif metric_name in ['Sharpe Ratio', 'Ann Sharpe Ratio']:
                        try:
                            # Prefer Ann Sharpe Ratio if it exists
                            if metric_name == 'Ann Sharpe Ratio' or sharpe is None:
                                sharpe = float(parts[target_column])
                        except (ValueError, IndexError):
                            pass

                    # Extract Calmar Ratio
                    elif metric_name == 'Calmar Ratio':
                        try:
                            calmar = float(parts[target_column])
                        except (ValueError, IndexError):
                            pass

                    # Extract ICIR
                    elif metric_name == 'ICIR':
                        try:
                            icir = float(parts[target_column])
                        except (ValueError, IndexError):
                            pass

                    # Extract Factor Autocorr
                    elif metric_name == 'Factor Autocorr':
                        try:
                            factor_autocorr = float(parts[target_column])
                        except (ValueError, IndexError):
                            pass

            # Fallback for single-value format (used in some 'd' category files)
            else:
                # Extract IC Mean
                if line.startswith('IC Mean'):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            ic = float(parts[1].strip())
                        except ValueError:
                            pass

                # Extract Sharpe Ratio
                elif line.startswith('Sharpe Ratio') or line.startswith('Ann Sharpe Ratio'):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            sharpe = float(parts[1].strip())
                        except ValueError:
                            pass

                # Extract Calmar Ratio
                elif line.startswith('Calmar Ratio'):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            calmar = float(parts[1].strip())
                        except ValueError:
                            pass

                # Extract ICIR
                elif line.startswith('ICIR'):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            icir = float(parts[1].strip())
                        except ValueError:
                            pass

                # Extract Factor Autocorr
                elif line.startswith('Factor Autocorr'):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            factor_autocorr = float(parts[1].strip())
                        except ValueError:
                            pass

    except Exception as e:
        print(f"Error reading {summary_path}: {e}")

    return ic, sharpe, calmar, icir, factor_autocorr


def process_draft_csv():
    """Process draft.csv to extract performance metrics."""
    # Configuration
    base_dir = "/workspace/worker/pj/Chrono/genuis/mizar/records"
    method = "cicso0"
    instruments = "ims"
    task_id = "200037"
    period = 15

    # Input and output files
    input_csv = "/workspace/worker/pj/Chrono/genuis/mizar/records/cicso0/ims/rulex/200037/nxt1_ret_15h/draft.csv"
    output_csv = "/workspace/worker/pj/Chrono/genuis/mizar/records/cicso0/ims/rulex/200037/nxt1_ret_15h/draft_with_performance.csv"

    # Read draft.csv
    print(f"Reading {input_csv}...")
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} rows")

    # Process each row
    processed_rows = []
    for idx, row in enumerate(rows):
        if (idx + 1) % 20 == 0:
            print(f"Processing row {idx+1}/{len(rows)}...")

        formula = row.get('formula', '').strip('"')
        source = str(row.get('source', ''))
        category = str(row.get('category', ''))
        direction = str(row.get('direction', ''))

        # Find performance_summary.txt
        summary_path = find_performance_summary_by_formula(
            base_dir=base_dir,
            method=method,
            instruments=instruments,
            task_id=task_id,
            period=period,
            category=category,
            source=source,
            formula=formula
        )


        # Extract specific metrics
        ic, sharpe, calmar, icir, factor_autocorr = extract_specific_metrics(summary_path, category)

        # Create output row
        output_row = {
            'formula': row.get('formula', ''),
            'direction': direction,
            'source': source,
            'category': category,
            'ic': ic,
            'Sharpe': sharpe,
            'Calmar': calmar,
            'ICIR': icir,
            'Factor Autocorr': factor_autocorr
        }

        processed_rows.append(output_row)

    # Write output CSV
    print(f"\nSaving results to {output_csv}...")
    fieldnames = ['formula', 'direction', 'source', 'category', 'ic', 'Sharpe', 'Calmar', 'ICIR', 'Factor Autocorr']

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    print(f"Done! Output saved to {output_csv}")
    print(f"Total rows processed: {len(processed_rows)}")


def main():
    # Process draft.csv instead of final.csv
    process_draft_csv()


if __name__ == "__main__":
    main()
