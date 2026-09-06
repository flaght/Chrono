from html import escape


def neutralize_prompt_braces(value):
    """将动态正文的大括号转成字符实体，避免多层模板重复解释。"""
    return str(value).replace("{", "&#123;").replace("}", "&#125;")
