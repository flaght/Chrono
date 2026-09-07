import toml, os, pdb

contract_file = os.path.join(os.environ['ENTL_CONFIG'], 'config/contract.toml')
contract = toml.load(contract_file)

### 基础合约配置
MAIN_CONTRACT_MAPPING = contract['MAIN_CONTRACT']
CONT_MULTNUM_MAPPING = contract['CONT_MULTNUM']
SYMBOL_CONTRANCT_MAPPING = contract['SYMBOL_CONTRANCT']

### 基础字段配置
SYMBOL_IMPULSE_MAPPING = contract['SYMBOL_IMPULSE']

### 策略配置
CHAOS_PHECDA_MAPPING = contract['CHAOS_PHECDA']

### 期权对应现货
#OPTIONS_SPOT = contract['OPTIONS_SPOT']

### 策略配置
CHAOS_VIRGTOR_MAPPING = contract['CHAOS_VIRGTOR']
#KICHAOS_VIRGTOR_MAPPING = contract['KICHAOS_VIRGTOR']

CHAOS_SIRIUS_MAPPING = contract['CHAOS_SIRIUS']

### 统一表名
MARKET_BAR_TABLE = 'impluse_market_bar'
RAW_FACTORS_TABLE = 'impluse_raw_factors'
DERIV_FACTORS_TABLE = 'impluse_deriv_factors'
NORM_FACTORS_TABLE = 'impluse_norm_factors'
TRADER_BIAS_TABLE = 'impluse_trader_bias'
TRADER_EVENT_TABLE = 'impluse_trader_event'

### 复权因子
ADUJSTED_FACTOR_MAPPING = contract['ADUJSTED_FACTOR']
