"""
特征文件参数化存储工具

提供通用的参数化保存和加载功能，支持任意参数组合。
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import hashlib


def format_param_value(value: Any) -> str:
    """
    格式化参数值为文件名友好的字符串
    
    参数:
        value: 任意类型的参数值
    
    返回:
        格式化后的字符串
    """
    s = str(value)
    # 处理科学计数法：1e-10 -> 1e10
    if 'e-' in s:
        s = s.replace('e-', 'e')
    # 替换特殊字符
    s = s.replace('.', 'p').replace('-', 'n').replace('+', '').replace('/', '_')
    # 移除空格和其他特殊字符
    s = ''.join(c if c.isalnum() or c in ['_', 'p', 'e'] else '_' for c in s)
    return s


def generate_param_filename(params: Dict[str, Any], base_name: str = "features", 
                           suffix: str = "csv", use_hash: bool = False) -> str:
    """
    根据参数字典生成文件名
    
    参数:
        params: 参数字典，键值对会被编码到文件名中
        base_name: 文件名基础
        suffix: 文件后缀
        use_hash: 是否使用哈希值（当参数过多时，使用哈希更简洁）
    
    返回:
        文件名字符串
    """
    if not params:
        return f"{base_name}.{suffix}"
    
    # 按键名排序以确保一致性
    sorted_keys = sorted(params.keys())
    
    if use_hash or len(params) > 5:
        # 参数过多时使用哈希值
        param_str = '_'.join(f"{k}{format_param_value(params[k])}" for k in sorted_keys)
        # 使用MD5哈希的前8位
        hash_value = hashlib.md5(param_str.encode()).hexdigest()[:8]
        filename = f"{base_name}_{hash_value}.{suffix}"
    else:
        # 参数较少时直接编码到文件名
        param_parts = [f"{k}{format_param_value(params[k])}" for k in sorted_keys]
        param_str = '_'.join(param_parts)
        filename = f"{base_name}_{param_str}.{suffix}"
    
    return filename


def save_features_with_params(feature_df: pd.DataFrame, 
                              ic_df: pd.DataFrame,
                              params: Dict[str, Any],
                              metadata: Optional[Dict[str, Any]] = None,
                              output_dir: str = "features_output") -> Tuple[str, str, str]:
    """
    保存特征文件和参数元数据（通用版本，支持任意参数）
    
    参数:
        feature_df: 特征DataFrame
        ic_df: IC结果DataFrame
        params: 参数字典，用于生成文件名和保存元数据
        metadata: 额外的元数据（不会用于文件名生成）
        output_dir: 输出目录
    
    返回:
        (feature_path, ic_path, params_path) 元组
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    feature_filename = generate_param_filename(params, base_name="features", suffix="csv")
    ic_filename = generate_param_filename(params, base_name="ic", suffix="csv")
    params_filename = generate_param_filename(params, base_name="params", suffix="json")
    
    # 保存特征文件
    feature_path = os.path.join(output_dir, feature_filename)
    feature_df.to_csv(feature_path, index=False)
    print(f"✓ 特征文件已保存至: {feature_path}")
    print(f"  特征数量: {len(feature_df)}")
    
    # 保存IC文件
    ic_path = os.path.join(output_dir, ic_filename)
    ic_df.to_csv(ic_path, index=False)
    print(f"✓ IC文件已保存至: {ic_path}")
    
    # 合并所有元数据
    all_metadata = {
        **params,  # 参数本身也是元数据的一部分
        "feature_count": len(feature_df),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "feature_file": feature_filename,
        "ic_file": ic_filename
    }
    
    # 添加额外的元数据
    if metadata:
        all_metadata.update(metadata)
    
    # 保存JSON格式的元数据
    params_path = os.path.join(output_dir, params_filename)
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ 参数元数据已保存至: {params_path}")
    
    # 更新参数索引文件（记录所有参数组合）
    index_file = os.path.join(output_dir, "params_index.csv")
    if os.path.exists(index_file):
        index_df = pd.read_csv(index_file)
    else:
        index_df = pd.DataFrame()
    
    # 准备新记录
    new_record = {
        "timestamp": all_metadata["timestamp"],
        "feature_count": len(feature_df),
        "feature_file": feature_filename,
        "ic_file": ic_filename,
        "params_file": params_filename,
        **{k: str(v) for k, v in params.items()}  # 所有参数都添加到索引中
    }
    
    # 如果metadata中有特定字段，也添加进去
    if metadata:
        for key in ["method", "task_id", "period", "instruments"]:
            if key in metadata:
                new_record[key] = str(metadata[key])
    
    new_record_df = pd.DataFrame([new_record])
    
    # 合并到索引（避免重复）
    # 检查是否已存在相同的参数组合
    if len(index_df) > 0:
        # 比较参数列
        param_cols = [col for col in new_record.keys() if col not in 
                     ["timestamp", "feature_file", "ic_file", "params_file", "feature_count"]]
        existing_cols = [col for col in index_df.columns if col in param_cols]
        
        if existing_cols:
            # 检查是否有完全相同的参数组合
            is_duplicate = False
            for _, row in index_df.iterrows():
                if all(str(new_record.get(col, '')) == str(row.get(col, '')) for col in existing_cols):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                index_df = pd.concat([index_df, new_record_df], ignore_index=True)
        else:
            index_df = pd.concat([index_df, new_record_df], ignore_index=True)
    else:
        index_df = new_record_df
    
    index_df.to_csv(index_file, index=False)
    print(f"✓ 参数索引已更新: {index_file}")
    
    return feature_path, ic_path, params_path


def load_features_by_params(params: Dict[str, Any], 
                           output_dir: str = "features_output") -> Tuple[Optional[pd.DataFrame], 
                                                                         Optional[pd.DataFrame], 
                                                                         Optional[Dict[str, Any]]]:
    """
    根据参数字典加载已保存的特征文件（通用版本）
    
    参数:
        params: 参数字典，用于生成文件名
        output_dir: 输出目录
    
    返回:
        (feature_df, ic_df, params_dict) 元组，如果文件不存在则返回None
    """
    feature_filename = generate_param_filename(params, base_name="features", suffix="csv")
    ic_filename = generate_param_filename(params, base_name="ic", suffix="csv")
    params_filename = generate_param_filename(params, base_name="params", suffix="json")
    
    feature_path = os.path.join(output_dir, feature_filename)
    ic_path = os.path.join(output_dir, ic_filename)
    params_path = os.path.join(output_dir, params_filename)
    
    if not os.path.exists(feature_path):
        print(f"✗ 特征文件不存在: {feature_path}")
        return None, None, None
    
    feature_df = pd.read_csv(feature_path)
    ic_df = pd.read_csv(ic_path)
    
    params_dict = None
    if os.path.exists(params_path):
        with open(params_path, 'r', encoding='utf-8') as f:
            params_dict = json.load(f)
    
    print(f"✓ 已加载特征文件: {feature_path}")
    print(f"  特征数量: {len(feature_df)}")
    
    return feature_df, ic_df, params_dict


def list_all_saved_params(output_dir: str = "features_output") -> pd.DataFrame:
    """
    列出所有已保存的参数组合
    
    参数:
        output_dir: 输出目录
    
    返回:
        参数索引DataFrame
    """
    index_file = os.path.join(output_dir, "params_index.csv")
    if os.path.exists(index_file):
        index_df = pd.read_csv(index_file)
        print(f"✓ 找到 {len(index_df)} 个已保存的参数组合")
        return index_df
    else:
        print(f"✗ 参数索引文件不存在: {index_file}")
        return pd.DataFrame()


def search_params_by_values(output_dir: str = "features_output", **kwargs) -> pd.DataFrame:
    """
    根据参数值搜索已保存的参数组合
    
    参数:
        output_dir: 输出目录
        **kwargs: 要搜索的参数键值对
    
    返回:
        匹配的参数组合DataFrame
    """
    index_df = list_all_saved_params(output_dir)
    
    if len(index_df) == 0:
        return pd.DataFrame()
    
    # 过滤匹配的参数
    mask = pd.Series([True] * len(index_df))
    for key, value in kwargs.items():
        if key in index_df.columns:
            mask = mask & (index_df[key].astype(str) == str(value))
        else:
            # 如果列不存在，返回空DataFrame
            print(f"✗ 参数 '{key}' 不在索引中")
            return pd.DataFrame()
    
    result_df = index_df[mask]
    print(f"✓ 找到 {len(result_df)} 个匹配的参数组合")
    return result_df


# 便捷函数：针对常用的四个阈值参数
def save_features_with_thresholds(feature_df: pd.DataFrame,
                                  ic_df: pd.DataFrame,
                                  nan_threshold: float,
                                  var_threshold: float,
                                  corr_threshold: float,
                                  ic_threshold: float,
                                  metadata: Optional[Dict[str, Any]] = None,
                                  output_dir: str = "features_output") -> Tuple[str, str, str]:
    """
    保存特征文件（便捷函数，针对四个阈值参数）
    
    参数:
        feature_df: 特征DataFrame
        ic_df: IC结果DataFrame
        nan_threshold: NaN阈值
        var_threshold: 方差阈值
        corr_threshold: 相关性阈值
        ic_threshold: IC阈值
        metadata: 额外的元数据（如method, task_id, period等）
        output_dir: 输出目录
    
    返回:
        (feature_path, ic_path, params_path) 元组
    """
    params = {
        "nan_threshold": nan_threshold,
        "var_threshold": var_threshold,
        "corr_threshold": corr_threshold,
        "ic_threshold": ic_threshold
    }
    return save_features_with_params(feature_df, ic_df, params, metadata, output_dir)


def load_features_by_thresholds(nan_threshold: float,
                                var_threshold: float,
                                corr_threshold: float,
                                ic_threshold: float,
                                output_dir: str = "features_output") -> Tuple[Optional[pd.DataFrame],
                                                                              Optional[pd.DataFrame],
                                                                              Optional[Dict[str, Any]]]:
    """
    根据四个阈值参数加载已保存的特征文件（便捷函数）
    
    参数:
        nan_threshold: NaN阈值
        var_threshold: 方差阈值
        corr_threshold: 相关性阈值
        ic_threshold: IC阈值
        output_dir: 输出目录
    
    返回:
        (feature_df, ic_df, params_dict) 元组，如果文件不存在则返回None
    """
    params = {
        "nan_threshold": nan_threshold,
        "var_threshold": var_threshold,
        "corr_threshold": corr_threshold,
        "ic_threshold": ic_threshold
    }
    return load_features_by_params(params, output_dir)
