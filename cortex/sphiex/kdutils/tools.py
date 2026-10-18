import json, os


def expandvars(obj):
    """
    深度递归遍历：自动找出配置字典里所有字符串，遇到包着 ${VAR} 的全部就地展开为真实的 OS 环境变量。
    对数字、布尔值绝对安全放行无视。
    """
    # 场景 1：如果是字典，深入到每一个 Key: Value 去爆破
    if isinstance(obj, dict):
        return {k: expandvars(v) for k, v in obj.items()}

    # 场景 2：如果是列表，遍历列表里所有元素去爆破
    elif isinstance(obj, list):
        return [expandvars(item) for item in obj]

    # 场景 3（终极出口）：如果是字符串，调用 OS 原生魔术方法进行替换！
    elif isinstance(obj, str):
        return os.path.expandvars(obj)

    else:
        return obj
