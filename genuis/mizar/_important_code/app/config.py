"""Configuration management for the factor dashboard"""
import os
import re
from pathlib import Path
from typing import Optional, Tuple


def parse_csv_path(csv_path: str) -> Tuple[str, str, str, int]:
    """
    Parse final.csv path to extract method, instruments, task_id, and period.
    
    Expected path format:
    records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/final.csv
    
    Args:
        csv_path: Path to final.csv file
    
    Returns:
        Tuple of (method, instruments, task_id, period)
    """
    # Normalize path
    csv_path = os.path.abspath(csv_path)
    
    # Try to match the pattern
    pattern = r'records[/\\]([^/\\]+)[/\\]([^/\\]+)[/\\]rulex[/\\]([^/\\]+)[/\\]nxt1_ret_(\d+)h[/\\]final\.csv'
    match = re.search(pattern, csv_path)
    
    if match:
        method = match.group(1)
        instruments = match.group(2)
        task_id = match.group(3)
        period = int(match.group(4))
        return method, instruments, task_id, period
    
    # Fallback: try to infer from directory structure
    path_obj = Path(csv_path)
    if path_obj.name == 'final.csv':
        # Navigate up the directory tree
        period_dir = path_obj.parent
        if period_dir.name.startswith('nxt1_ret_') and period_dir.name.endswith('h'):
            period_str = period_dir.name.replace('nxt1_ret_', '').replace('h', '')
            try:
                period = int(period_str)
                task_id_dir = period_dir.parent
                task_id = task_id_dir.name
                rulex_dir = task_id_dir.parent
                if rulex_dir.name == 'rulex':
                    instruments_dir = rulex_dir.parent
                    instruments = instruments_dir.name
                    method_dir = instruments_dir.parent
                    method = method_dir.name
                    if method_dir.parent.name == 'records':
                        return method, instruments, task_id, period
            except (ValueError, AttributeError):
                pass
    
    # Default values if parsing fails
    return 'cicso0', 'ims', '200037', 15


def get_base_path() -> str:
    """
    Get the base path for records directory.
    
    Tries environment variables first, then falls back to a default.
    
    Returns:
        Base path string
    """
    # Try to get from environment variables (like kdutils.macro2 does)
    base_path_env = os.environ.get('BASE_PATH', '')
    record_path_env = os.environ.get('RECORD_PATH', '')
    
    if base_path_env and record_path_env:
        return os.path.join(base_path_env, record_path_env)
    
    # Fallback: use the workspace directory structure
    workspace_path = os.environ.get('WORKSPACE_PATH', '')
    if workspace_path:
        # Try to find records directory in workspace
        records_path = os.path.join(workspace_path, 'records')
        if os.path.exists(records_path):
            return records_path
    
    # Default: assume records is in the current workspace
    # Based on the example path: /workspace/worker/pj/Chrono/genuis/mizar/records/...
    # We'll extract up to 'mizar' and add 'records'
    current_file = os.path.abspath(__file__)
    mizar_dir = None
    for parent in Path(current_file).parents:
        if parent.name == 'mizar':
            mizar_dir = parent
            break
    
    if mizar_dir:
        records_path = mizar_dir / 'records'
        if records_path.exists():
            return str(records_path)
    
    # Check if we're in the workspace directory
    workspace_path = os.environ.get('WORKSPACE_PATH', '')
    if workspace_path:
        # Extract mizar directory from workspace path
        if 'mizar' in workspace_path:
            parts = workspace_path.split('mizar')
            if len(parts) > 1:
                mizar_path = parts[0] + 'mizar'
                records_path = os.path.join(mizar_path, 'records')
                if os.path.exists(records_path):
                    return records_path
    
    # Last resort: try current working directory
    current_dir = os.getcwd()
    if 'mizar' in current_dir:
        parts = current_dir.split('mizar')
        if len(parts) > 1:
            mizar_path = parts[0] + 'mizar'
            records_path = os.path.join(mizar_path, 'records')
            if os.path.exists(records_path):
                return records_path
    
    # Final fallback: relative path
    return 'records'


class Config:
    """Configuration class for the dashboard"""
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            csv_path: Path to final.csv file. If None, uses default path.
        """
        if csv_path is None:
            # Default path based on example
            csv_path = 'records/cicso0/ims/rulex/200037/nxt1_ret_15h/final.csv'
        
        self.csv_path = csv_path
        self.base_path = get_base_path()
        self.method, self.instruments, self.task_id, self.period = parse_csv_path(csv_path)
    
    def update_csv_path(self, new_csv_path: str):
        """Update CSV path and re-parse parameters."""
        self.csv_path = new_csv_path
        self.base_path = get_base_path()
        self.method, self.instruments, self.task_id, self.period = parse_csv_path(new_csv_path)

