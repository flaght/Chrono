import toml, os, pdb
contract_file = os.path.join(os.environ['ENTL_CONFIG'], 'config/contract.toml')
contract = toml.load(contract_file)

### 基础合约配置
MAIN_CONTRACT_MAPPING = contract['MAIN_CONTRACT']
CONT_MULTNUM_MAPPING = contract['CONT_MULTNUM']
SYMBOL_CONTRANCT_MAPPING = contract['SYMBOL_CONTRANCT']


### 策略配置
CHAOS_PHECDA_MAPPING = contract['CHAOS_PHECDA']

