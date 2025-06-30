import toml, os, pdb
contract_file = os.path.join(os.environ['LUMINA_CONFIG'], 'config/contract.toml')
contract = toml.load(contract_file)

CONT_MULTNUM_MAPPING = contract['CONT_MULTNUM']
COST_MAPPING = contract['COST']
SLIPPAGE_MAPPING = contract['SLIPPAGE']
INDEX_MAPPING = contract['GENTIC_INDEX']
PERFORMANCE_MAPPING = contract['PERFORMANCE']
INSTRUMENTS_CODES = contract['INSTRUMENTS']
THRESHOLD_MAPPING = contract['THRESHOLD']
FILTER_YEAR_MAPPING = contract['FILTER_YEAR']