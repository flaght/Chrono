import os, yaml
import pdb


def load_sirius_params(code: str, task_id: str):
    file_path = os.path.join("config", "sirius_strategy.yaml")
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = config[code]
    if task_id not in config:
        raise KeyError(f"配置文件中必须包含 {task_id}。")

    config = config[task_id]

    features = config['features']
    params = config['params']
    return features, params
