import json,re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Any, Optional, Literal, Union, get_args, get_origin



def get_fewshot_template(model_class) -> str:
    template = {}
    fields = getattr(model_class, 'model_fields', None) or getattr(
        model_class, '__fields__', {})

    for name, field in fields.items():
        desc = getattr(field, 'description', None)
        if desc is None and hasattr(field, 'field_info'):
            desc = getattr(field.field_info, 'description', "")
        desc = desc or "无描述"

        define_val = None
        extra_v2 = getattr(field, 'json_schema_extra', None)
        if extra_v2 and isinstance(extra_v2, dict):
            define_val = extra_v2.get("define")

        if not define_val and hasattr(field, 'field_info'):
            extra_v1 = getattr(field.field_info, 'extra', {})
            define_val = extra_v1.get("define")

        final_text = define_val or desc

        outer_type = getattr(field, 'annotation', None) or getattr(
            field, 'outer_type_', None)
        origin = getattr(outer_type, '__origin__', None)

        if origin is list or outer_type is list:
            template[name] = [f"{final_text}1", f"{final_text}2"]
        else:
            template[name] = f"{final_text}"

    return json.dumps(template, indent=4, ensure_ascii=False)

def build_dynamic_schema(schema_name: str,
                         fields_config: dict,
                         base_class=None):
    annotations = {}

    # 构建绝对没有任何注入风险的沙盒
    safe_typing_env = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "Any": Any,
        "Optional": Optional,
        "List": List,
        "Dict": Dict,
        "list": list,
        "dict": dict
    }

    for field_name, attr in fields_config.items():
        f_type_str = attr["type"]
        f_desc = attr["description"]

        try:
            real_type = eval(f_type_str, {"__builtins__": None},
                             safe_typing_env)
        except Exception:
            real_type = str

        f_define = attr.get("define", "")
        f_detail = attr.get("detail", "")
        field_kwargs = {"description": f_desc}
        if f_define or f_detail:
            field_kwargs["define"] = f_define
            field_kwargs["detail"] = f_detail
            field_kwargs["json_schema_extra"] = {
                "define": f_define,
                "detail": f_detail
            } 

        annotations[field_name] = (real_type, Field(..., **field_kwargs))

    DynamicModel = create_model(schema_name, **annotations) if isinstance(
        base_class, BaseModel) else create_model(
            schema_name, __base__=base_class, **annotations)
    return DynamicModel



def get_fewshot_template2(model_class) -> str:
    """Build a nested JSON example matching the actual Pydantic model."""
    template = _template_for_model(model_class)
    return json.dumps(template, indent=4, ensure_ascii=False)

def _field_text(field):
    """Return the human-readable example text stored on a Pydantic field."""
    desc = getattr(field, 'description', None)
    if desc is None and hasattr(field, 'field_info'):
        desc = getattr(field.field_info, 'description', "")
    desc = desc or "无描述"

    define_val = None
    extra_v2 = getattr(field, 'json_schema_extra', None)
    if extra_v2 and isinstance(extra_v2, dict):
        define_val = extra_v2.get("define")

    if not define_val and hasattr(field, 'field_info'):
        extra_v1 = getattr(field.field_info, 'extra', {})
        define_val = extra_v1.get("define")

    return define_val or desc

def _unwrap_optional(annotation):
    origin = get_origin(annotation)
    if origin is Union:
        args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
        if len(args) == 1:
            return args[0]
    return annotation

def _template_for_model(model_class):
    template = {}
    fields = getattr(model_class, 'model_fields', None) or getattr(
        model_class, '__fields__', {})

    for name, field in fields.items():
        final_text = _field_text(field)
        annotation = getattr(field, 'annotation', None) or getattr(
            field, 'outer_type_', None)
        annotation = _unwrap_optional(annotation)
        origin = get_origin(annotation)

        if origin in (list, List) or annotation is list:
            args = get_args(annotation)
            item_type = _unwrap_optional(args[0]) if args else Any
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                template[name] = [_template_for_model(item_type)]
            else:
                template[name] = [f"{final_text}1", f"{final_text}2"]
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            template[name] = _template_for_model(annotation)
        else:
            template[name] = f"{final_text}"

    return template


def generate_regex_from_format(format_str: str) -> re.Pattern:
    """
    黑科技：将 Python 的 str.format() 模板字符串，反向编译为正则表达式！
    比如将 "{0} [{1}] 针对 <{2}>" 动态转变为 "^(.*?)\s*\[(.*?)\]\s*针对\s*\<(.*?)\>"
    """
    # 1. 按照 {n} 将字符串拆分成硬编码的文本块
    parts = re.split(r'\{\d+\}', format_str)
    
    regex_parts =[]
    for p in parts:
        # 将文本中所有的空白符/换行符统一替换为单空格，并进行正则安全转义
        p_esc = re.escape(re.sub(r'\s+', ' ', p))
        # 把转义后的空格替换为 \s*，这样能完美兼容大模型输出时多一个空格或少一个换行的问题
        p_esc = p_esc.replace(r'\ ', r'\s*')
        regex_parts.append(p_esc)
        
    # 2. 将文本块与捕获组 (.*?) 拼接
    pattern = "^"
    for i in range(len(regex_parts) - 1):
        pattern += regex_parts[i]
        if i == len(regex_parts) - 2:
            pattern += r"(.*)"  # 最后一个变量匹配结尾的所有内容 (包含换行)
        else:
            pattern += r"(.*?)" # 中间的变量进行非贪婪匹配
    pattern += regex_parts[-1] + "$"
    
    # 3. 返回编译好的正则对象 (开启 re.DOTALL 以匹配多行正文)
    return re.compile(pattern, re.DOTALL)




def enrich_broadcast_header(msg: Optional[str],
                            agents_dict: dict) -> Optional[str]:
    """
    解析大模型输出，并严格使用 DEFAULT_XXX_FORMAT 常量进行装配。
    无论你怎么修改常量里的中文字符，这里的逻辑一行都不需要改！
    """
    if not msg:
        return None
    
    # --- 解析 Speak 动作 ---
    match_speak = REGEX_SPEAK.match(msg)
    if match_speak:
        # args 包含了提取出来的 {0}, {1}, {2}, {3}
        args = list(match_speak.groups()) 
        
        sender = args[0]
        target = args[2]
        
        # 替换发送者 {0}
        if sender in agents_dict:
            args[0] = f"{agents_dict[sender].title}【{sender}】"
            
        # 替换目标对象 {2}，并全角化内部方括号(防 rich 吞字)
        if target in agents_dict:
            args[2] = f"{agents_dict[target].title}【{target}】"
        else:
            args[2] = target.replace("[", "【").replace("]", "】")
            
        # 重点：直接把处理好的参数解包塞回你的原始常量中！
        return DEFAULT_SPEAK_FORMAT.format(*args)

    # --- 解析 Vote 动作 ---
    match_vote = REGEX_VOTE.match(msg)
    if match_vote:
        args = list(match_vote.groups())
        sender = args[0] # {0}
        
        if sender in agents_dict:
            args[0] = f"{agents_dict[sender].title}【{sender}】"
            
        return DEFAULT_VOTE_FORMAT.format(*args)

    # --- 解析 Terminate 动作 ---
    match_term = REGEX_TERMINATE.match(msg)
    if match_term:
        args = list(match_term.groups())
        sender = args[0] # {0}
        
        if sender in agents_dict:
            args[0] = f"{agents_dict[sender].title}【{sender}】"
            
        return DEFAULT_TERMINATE_FORMAT.format(*args)

    # 如果全都不匹配，安全降级原样返回
    return msg
