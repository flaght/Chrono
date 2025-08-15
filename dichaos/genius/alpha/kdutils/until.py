import os


def create_memory_path(base_path, date, method=None):
    path = os.path.join(base_path, method, 'memory', date) if isinstance(
        method, str) else os.path.join(base_path, 'memory', date)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
