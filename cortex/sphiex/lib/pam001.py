import os, toml


def load_memory_params():
    config_file = os.path.join('config.toml')  # 主要是记忆体
    config = toml.load(config_file)
    memory_params = {}
    if 'short' in config:
        memory_params['short'] = config['short']
    if 'mid' in config:
        memory_params['mid'] = config['mid']
    if 'long' in config:
        memory_params['long'] = config['long']
    if 'reflection' in config:
        memory_params['reflection'] = config['reflection']
    return memory_params
