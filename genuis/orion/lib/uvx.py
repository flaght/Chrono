import os,yaml,pdb
import pandas as pd
from kdutils.macro2 import *

def check_config_keys(config, **kwargs):
    missing = []
    for section, uid in kwargs.items():
        if uid not in config.get(section, {}):
            missing.append(f"[{section}][{uid}]")
    
    if missing:
        raise KeyError(f"配置缺失: {' , '.join(missing)}")
    

def load_rl_params(file_dirs: str, trade_id: str, model_id: str, 
                   feature_id:str, 
                   env_id:str,
                   train_id:str,
                   name:str = 'param'):
    file_path = os.path.join(file_dirs, "params", "{}.yaml".format(name))
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config = config['params']
        if 'trade' not in config or 'model' not in config:
            raise KeyError("配置文件中必须包含 'params'顶级键。")
        check_config_keys(config=config, trade=trade_id,
                          model=model_id, feature=feature_id,
                          env=env_id, train=train_id)
        env_param = config['env'][env_id]
        train_param = config['train'][train_id]
        trade_param = config['trade'][trade_id]
        model_param = config['model'][model_id]
        
        feature_param = config['feature'][feature_id].split('|')
        return env_param, trade_param, model_param, train_param, feature_param
            

    except FileNotFoundError:
        print(f"错误：配置文件 '{file_path}' 不存在。")
        return None, None
    except yaml.YAMLError as e:
        print(f"错误：解析YAML文件 '{file_path}' 失败: {e}")
        return None, None
    except KeyError as e:
        print(f"错误：配置文件中缺少必需的键路径: {e}")
        return None, None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None, None