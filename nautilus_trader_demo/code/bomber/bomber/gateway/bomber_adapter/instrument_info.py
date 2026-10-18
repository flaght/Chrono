"""
合约信息查询工具

从合约代码（如 IC2401、MO2401-C-5000）解析出品种、合约类型、乘数、标的等信息。
不依赖数据库，纯字符串 + 内置常量，可在任何环境使用。

典型用法：
    from bomber_adapter.instrument_info import (
        get_product, is_option, is_future, get_underlying,
        get_multiplier, get_contract_type, parse_contract,
    )

    get_product("IC2401")          # "IC"
    get_product("MO2401-C-5000")   # "MO"
    is_option("MO2401-C-5000")     # True
    is_future("IC2401")            # True
    get_underlying("MO2401-C-5000")  # "IM2401"
    get_multiplier("IF2401")       # 300
"""
import re
from typing import Optional, Tuple


# ================================================================
# 内置常量
# ================================================================

# 期货品种 → 合约乘数
FUTURES_MULTIPLIER = {
    # 中金所股指期货/期权标的
    "IF": 300,   # 沪深300
    "IH": 300,   # 上证50
    "IC": 200,   # 中证500
    "IM": 200,   # 中证1000
    # 中金所国债期货
    "TS": 20000, # 2 年期
    "TF": 10000, # 5 年期
    "T":  10000, # 10 年期
    "TL": 10000, # 30 年期
}

# 期权品种 → 合约乘数（中金所规定：股指期权合约乘数统一为每点 100 元，
# 与标的期货的乘数不同！如 MO 期权乘数=100，而标的 IM 期货乘数=200）
OPTION_MULTIPLIER = {
    "IO": 100,   # 沪深300 期权（标的 IF，IF 乘数 300）
    "HO": 100,   # 上证50 期权（标的 IH，IH 乘数 300）
    "MO": 100,   # 中证1000 期权（标的 IM，IM 乘数 200）
    "OP": 10,    # 50ETF 期权（上交所，标的 SH510050）
}

# 期权品种 → 标的期货品种
# 期权品种 → 标的期货/ETF 品种
OPTION_TO_UNDERLYING = {
    "IO": "IF",        # 沪深300 期权 → 沪深300 期货
    "HO": "IH",        # 上证50 期权 → 上证50 期货
    "MO": "IM",        # 中证1000 期权 → 中证1000 期货
    "OP": "SH510050",  # 50ETF 期权 → 50ETF（上交所）
}

# 合约类型
CONTRACT_TYPE_FUTURE = "FUTURE"
CONTRACT_TYPE_OPTION = "OPTION"
CONTRACT_TYPE_INDEX = "INDEX"
CONTRACT_TYPE_COMBINATION = "COMBINATION"
CONTRACT_TYPE_UNKNOWN = "UNKNOWN"

# 指数代码前缀
INDEX_PREFIXES = ("SH", "SZ", "CSI")


# ================================================================
# 解析函数
# ================================================================

def parse_contract(inst: str) -> Tuple[str, str, str]:
    """
    解析合约代码，返回 (product, expiry_yymm, suffix)

    示例：
        parse_contract("IC2401")          → ("IC", "2401", "")
        parse_contract("MO2401-C-5000")   → ("MO", "2401", "C-5000")
        parse_contract("IC2401.CFFEX")    → ("IC", "2401", "")
        parse_contract("000905.SH")       → ("000905", "", "SH")  # 指数格式
    """
    # 处理 "000905.SH" 格式（数字开头 + 交易所后缀）
    if re.match(r'^\d{6}\.\w+$', inst):
        code, exchange = inst.split('.')
        return (code, "", exchange)

    # 去掉交易所后缀（如 .CFFEX）
    code = inst.split('.')[0] if '.' in inst else inst

    # 提取字母前缀（品种）
    m = re.match(r'^([A-Za-z]+)', code)
    if not m:
        return ("", "", "")
    product = m.group(1).upper()

    # 剩余部分：先取 4 位年月（如果存在），再取其他后缀
    rest = code[m.end():]
    expiry = ""
    suffix = ""
    if len(rest) >= 4 and rest[:4].isdigit():
        expiry = rest[:4]
        suffix = rest[4:]
        if suffix.startswith('-'):
            suffix = suffix[1:]
    else:
        suffix = rest
        if suffix.startswith('-'):
            suffix = suffix[1:]

    return (product, expiry, suffix)


def get_product(inst: str) -> str:
    """
    获取合约品种代码

    示例：
        get_product("IC2401")          → "IC"
        get_product("MO2401-C-5000")   → "MO"
        get_product("IC2401.CFFEX")    → "IC"
    """
    product, _, _ = parse_contract(inst)
    return product


def is_option(inst: str) -> bool:
    """
    判断是否为期权合约

    判断规则（优先级顺序）：
      1. 品种在 OPTION_MULTIPLIER 中（MO/IO/HO）
      2. 合约代码含 '-'（期权代码格式：MO2401-C-5000）
    """
    product = get_product(inst)
    if product in OPTION_MULTIPLIER:
        return True
    # 去掉交易所后缀后再判断
    code = inst.split('.')[0] if '.' in inst else inst
    return '-' in code


def is_future(inst: str) -> bool:
    """判断是否为期货合约（非期权）"""
    return not is_option(inst) and not is_index(inst)


def is_index(inst: str) -> bool:
    """
    判断是否为指数合约（非期货/非期权）

    指数代码格式：
      - SH000XXX  上交所指数（如 SH000905 中证500, SH000300 沪深300）
      - SZ399XXX  深交所指数
      - CSI899XXX / CSI932XXX  中证指数
      - 000905.SH / 000016.SH（数字代码 + 交易所后缀）
      - SH000016.SH / 000016.SH（带交易所后缀）

    判断规则：
      1. 品种前缀在 INDEX_PREFIXES 中（SH/SZ/CSI）
      2. 或格式为 "6位数字.交易所"（如 000905.SH）
      3. 且不是期货/期权格式（不含 '-'，且后面不跟 4 位年月）
    """
    # 处理 "000905.SH" 格式（数字代码 + 交易所后缀）
    if re.match(r'^\d{6}\.\w+$', inst):
        return True

    code = inst.split('.')[0] if '.' in inst else inst
    product = get_product(inst)

    # 指数前缀
    if product in ("SH", "SZ", "CSI"):
        # 排除被误判为期货的情况（SH 开头 + 4 位数字 = 期货？实际上没有 SH 期货）
        # 指数代码通常是 SH + 6 位数字（如 SH000905）
        return True

    # 兜底：看合约代码格式是否像指数
    # 期货格式：IC2401, IF2401（2 字母 + 4 位年月）
    # 指数格式：SH000905（2 字母 + 6 位数字），或 CSI932000（3 字母 + 6 位数字）
    if re.match(r'^(SH|SZ|CSI)\d{6}$', code):
        return True

    return False


def is_stock(inst: str) -> bool:
    """判断是否为股票合约（暂时未支持，预留接口）"""
    return False


def get_contract_type(inst: str) -> str:
    """
    返回合约类型字符串：
      - "FUTURE"
      - "OPTION"
      - "INDEX"
      - "COMBINATION"（组合）
      - "UNKNOWN"
    """
    if is_index(inst):
        return CONTRACT_TYPE_INDEX
    if is_option(inst):
        return CONTRACT_TYPE_OPTION
    product = get_product(inst)
    if product in FUTURES_MULTIPLIER or product in OPTION_MULTIPLIER:
        return CONTRACT_TYPE_FUTURE
    return CONTRACT_TYPE_UNKNOWN


def get_multiplier(inst: str) -> Optional[float]:
    """
    返回合约乘数，未知品种返回 None

    示例：
        get_multiplier("IC2401")          → 200
        get_multiplier("IF2401")          → 300
        get_multiplier("MO2401-C-5000")   → 100
        get_multiplier("IO2401-C-4000")   → 100
    """
    product = get_product(inst)
    if product in OPTION_MULTIPLIER:
        return OPTION_MULTIPLIER[product]
    if product in FUTURES_MULTIPLIER:
        return FUTURES_MULTIPLIER[product]
    return None


def get_underlying(inst: str) -> Optional[str]:
    """
    返回期权对应的标的期货合约代码，期货本身返回 None

    注意：此函数返回的是**品种**层面的标的（如 MO → IM），
    不是具体合约的标的合约（MO2401-C-5000 标的不是 IM2401 的具体映射）。
    如需具体合约标的，请配合 parse_contract 使用。

    示例：
        get_underlying("MO2401-C-5000")   → "IM"
        get_underlying("IO2401-C-4000")   → "IF"
        get_underlying("IC2401")          → None
    """
    product = get_product(inst)
    return OPTION_TO_UNDERLYING.get(product)


def get_underlying_contract(inst: str) -> Optional[str]:
    """
    返回期权对应的标的期货**合约**代码（同月份）

    示例：
        get_underlying_contract("MO2401-C-5000")   → "IM2401"
        get_underlying_contract("IO2401-C-4000")   → "IF2401"
        get_underlying_contract("IC2401")          → None
    """
    if not is_option(inst):
        return None
    product, expiry, _ = parse_contract(inst)
    underlying_product = OPTION_TO_UNDERLYING.get(product)
    if not underlying_product or not expiry:
        return None
    return f"{underlying_product}{expiry}"


def get_option_info(inst: str) -> Optional[dict]:
    """
    解析期权合约的完整信息，期货返回 None

    返回：
        {
            "product": "MO",
            "expiry": "2401",
            "option_type": "C",         # C 或 P
            "strike": 5000.0,
            "underlying": "IM",
            "underlying_contract": "IM2401",
            "multiplier": 200,
        }
    """
    if not is_option(inst):
        return None
    product, expiry, suffix = parse_contract(inst)

    # 解析 option_type 和 strike：格式为 C-5000 或 P-4000
    option_type = None
    strike = None
    if '-' in suffix:
        parts = suffix.split('-')
        if len(parts) >= 2:
            option_type = parts[0].upper()
            try:
                strike = float(parts[1])
            except ValueError:
                pass

    underlying = OPTION_TO_UNDERLYING.get(product)
    underlying_contract = f"{underlying}{expiry}" if underlying and expiry else None

    return {
        "product": product,
        "expiry": expiry,
        "option_type": option_type,
        "strike": strike,
        "underlying": underlying,
        "underlying_contract": underlying_contract,
        "multiplier": OPTION_MULTIPLIER.get(product),
    }
