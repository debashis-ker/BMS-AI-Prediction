import pandas as pd
import json
from pathlib import Path  
from typing import List, Dict, Any


def create_hierarchical_stats(df: pd.DataFrame, hierarchy_keys: List[str]) -> Dict[str, Any]:
    if not hierarchy_keys:
        raise ValueError("The hierarchy_keys list cannot be empty.")
    missing_keys = [key for key in hierarchy_keys if key not in df.columns]
    if missing_keys:
        raise ValueError(f"The following keys are not in the DataFrame: {missing_keys}")
    result_json = {}
    grouped = df.groupby(hierarchy_keys)
    for group_keys, group_df in grouped:
        monitoring_status = group_df['monitoring_data']
        # Convert to numeric, replacing non-numeric values with NaN
        monitoring_status_numeric = pd.to_numeric(monitoring_status, errors='coerce').dropna()
        
        # Skip groups with no numeric data
        if len(monitoring_status_numeric) == 0:
            stats = {
                'min': None, 'max': None,
                'lower_quartile': None, 'upper_quartile': None, 'IQR': None,
                'lower_bound': None, 'upper_bound': None,
                'note': 'No numeric data available'
            }
        else:
            q1 = monitoring_status_numeric.quantile(0.25)
            q3 = monitoring_status_numeric.quantile(0.75)
            iqr = q3 - q1
            stats = {
                'min': monitoring_status_numeric.min(), 'max': monitoring_status_numeric.max(),
                'lower_quartile': q1, 'upper_quartile': q3, 'IQR': iqr,
                'lower_bound': q1 - 1.5 * iqr, 'upper_bound': q3 + 1.5 * iqr,
            }
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        current_level = result_json
        for key in group_keys[:-1]:
            current_level = current_level.setdefault(key, {})
        current_level[group_keys[-1]] = {'count': len(group_df), 'statistics': stats}
    def add_counts_to_parents(node: Dict[str, Any]) -> int:
        if 'statistics' in node:
            return node.get('count', 0)
        total_count = sum(add_counts_to_parents(node[key]) for key in node)
        node['count'] = total_count
        return total_count
    add_counts_to_parents(result_json)
    return result_json


def generate_and_save_stats(
    df: pd.DataFrame, 
    hierarchy: List[str], 
    output_filename: str
) -> None:
    """
    Generates statistics and saves them to a 'resources' folder in the project root.
    It robustly finds the project root by assuming it's the parent of the notebook's directory.
    """
    print(f"Generating hierarchical statistics for hierarchy: {hierarchy}...")
    
    def convert_to_native_types(obj):
        """Convert numpy and pandas types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return obj
    
    try:
        hierarchical_data = create_hierarchical_stats(df, hierarchy)
        # Convert all numpy types to native Python types
        hierarchical_data = convert_to_native_types(hierarchical_data)
        
        notebook_dir = Path.cwd()
        project_root = notebook_dir.parent
        resources_dir = project_root / 'resources'
        resources_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = resources_dir / output_filename
        
        with open(file_path, 'w') as json_file:
            json.dump(hierarchical_data, json_file, indent=4)
        
        print(f"Successfully saved analysis to '{file_path}'")

    except (ValueError, IOError) as e:
        print(f"An error occurred: {e}")