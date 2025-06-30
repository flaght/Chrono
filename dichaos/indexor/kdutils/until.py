import os


def create_agent_path(name, method, symbol, date):
    path = os.path.join(os.environ['BASE_PATH'], method, name, 'brain', symbol,
                        date)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
