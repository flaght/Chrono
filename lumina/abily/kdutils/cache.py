import os, pdb
from ultron.kdutils.file import load_pickle, dump_pickle

base_path = os.path.join("cache", "abily")
if not os.path.exists(base_path):
    os.makedirs(base_path)


def exist_cache(code, task_id, method, cache_file):
    """
    Check if the cache file exists.
    """
    return os.path.exists(
        os.path.join(base_path, code, str(task_id), method, cache_file))


def load_cache(code, task_id, method, cache_file):
    """
    Load the cache file if it exists.
    """
    if exist_cache(code=code,
                   task_id=task_id,
                   method=method,
                   cache_file=cache_file):
        return load_pickle(
            os.path.join(base_path, code, str(task_id), method, cache_file))
    else:
        return None


def save_cache(code, task_id, method, cache_file, data):
    """
    Save the data to the cache file.
    """
    dump_pickle(
        data, os.path.join(base_path, code, str(task_id), method, cache_file))
