import os,yaml
import pandas as pd
from kdutils.macro2 import *
from lib.utils.params import Params
from lib.lsx001 import fetch_times
from lib import logger

def fetch_research_fetures(method, instruments,task_id,period, name, params):
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
    fparams = Params(base_path=os.path.join(outdirs), experiment_name="feature")
    return  fparams.load_content(params=params, name=name)

def save_clean_data(output, data, params):
    params_id = Params.create_tag(params)
    output = os.path.join(output, "data")
    if not os.path.exists(output):
        os.makedirs(output)
    filename = os.path.join(output, "clean_data_{0}.feather".format(params_id))
    data.to_feather(filename)

def fetch_clean_data2(method, task_id, instruments, output, params, 
                     train_time=['train_time','val_time'], test_time=['test_time']): ## 加载
    params_id = Params.create_tag(params)
    output = os.path.join(output, "data")
    filename = os.path.join(output, "clean_data_{0}.feather".format(params_id))
    final_data =  pd.read_feather(filename)
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    train_data = final_data[(final_data['trade_time'] >= time_array[train_time[0]][0]) & (
        final_data['trade_time'] <= time_array[train_time[-1]][-1])]
    test_data = final_data[(final_data['trade_time'] >= time_array[test_time[0]][0]) & (
        final_data['trade_time'] <= time_array[test_time[-1]][-1])]
    return train_data, test_data

def load_rl_params(file_dirs: str, trade_id: str, model_id: str, 
                   feature_id:str, 
                   env_id:str,
                   train_id:str,
                   regime_id:str,
                   name:str = 'param'):
    file_path = os.path.join(file_dirs, "params", "{}.yaml".format(name))
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config = config['params']
        if 'trade' not in config or 'model' not in config:
            raise KeyError("配置文件中必须包含 'params'顶级键。")
        
        if trade_id not in config['trade'] or model_id not in config['model']\
            or feature_id not in config['feature'] or env_id not in config['env']\
                or train_id not in config['train'] or regime_id not in config['env']:
            raise KeyError(f"配置文件中没有 {trade_id} or {model_id} 数据")
        
        env_param = config['env'][env_id]
        train_param = config['train'][train_id]
        trade_param = config['trade'][trade_id]
        model_param = config['model'][model_id]
        feature_param = config['feature'][feature_id].split()
        regime_min_param = config['regime'][regime_id]['min'].split()
        regime_daily_param = config['regime'][regime_id]['daily'].split()
        return env_param, trade_param, model_param, train_param, feature_param, \
                    regime_min_param,regime_daily_param
            

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
    
    
def load_params2(file_dirs: str, name:str, model_name: str, data_name:str):
    file_path = os.path.join(file_dirs, "params", "{}.yaml".format(name))
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'params' not in config  or 'data' not in config:
            raise KeyError("配置文件中必须包含 'params' 'train' 'data' 两个顶级键。")
        
        model_params = config['params'][model_name]
        data_params = config['data'][data_name]
        
        print(f"成功从 '{file_path}' 加载配置。")
        return model_params, data_params

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


def load_params1(file_dirs: str, name:str, model_name: str, train_name: str, data_name:str):
    file_path = os.path.join(file_dirs, "params", "{}.yaml".format(name))
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'params' not in config or 'train' not in config or 'data' not in config:
            raise KeyError("配置文件中必须包含 'params' 'train' 'data' 两个顶级键。")
        
        model_params = config['params'][model_name]
        train_params = config['train'][train_name]
        data_params = config['data'][data_name]
        
        print(f"成功从 '{file_path}' 加载配置。")
        return model_params, train_params,data_params

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


def load_params(file_dirs: str, name:str, model_name: str, train_name: str):
    file_path = os.path.join(file_dirs, "params", "{}.yaml".format(name))
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'params' not in config or 'train' not in config:
            raise KeyError("配置文件中必须包含 'params' 和 'train' 两个顶级键。")

        model_params = config['params'][model_name]
        train_params = config['train'][train_name]
        
        print(f"成功从 '{file_path}' 加载配置。")
        return model_params, train_params

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
    


def load_params3(file_dirs: str, file_name:str, signal_name: str=None):
    file_path = os.path.join(file_dirs, "params", "{}.yaml".format(file_name))
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = yaml.safe_load(f)
    
    output_list = []
    for method_name , signals in raw_data['signal_configs'].items():
        if method_name != signal_name:
            continue
        for signal_id, param in signals. items():
            item = {
                "param_id":f"{method_name}_{signal_id}",
                "method":f"{method_name}_signal",
                "param": param
            }
            output_list.append(item)
    return output_list
            
        
    
    
    
def create_train_records(method, task_id, instruments, period, category, params):
    '''
    # 同时设置多个参数
    
logger.configure(
    verbose=True,           # 或使用 level=logging.DEBUG
    log_file="debug.log"
)

# 或者
logger.configure(
    level=logging.WARNING,
    log_file="warnings.log"
)
    '''
    name = Params.create_tag(params)
    output_dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research", "experiment", category)

    if not os.path.exists(output_dirs):
        os.makedirs(output_dirs)
    
    filename =  os.path.join(output_dirs, "{0}.log".format(name))
    print(filename)
    logger.configure(log_file=filename)
    return name


def load_sirius_params(code: str, task_id: str):
    file_path = os.path.join("config", "strategy","sirius.yaml")
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = config[code]
    
    if task_id not in config:
        raise KeyError(f"配置文件中必须包含 {task_id}。")

    config = config[task_id]

    features = config['features']
    params = config['params']
    return features, params
