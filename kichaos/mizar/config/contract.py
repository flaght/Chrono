import toml, os, pdb

contract_file = os.path.join(os.environ['MIZAR_CONFIG'], 'config/contract.toml')
contract = toml.load(contract_file)

CONT_MULTNUM_MAPPING = contract['CONT_MULTNUM']
COST_MAPPING = contract['COST']
INIT_CASH_MAPPING = contract['INIT_CASH']
TRADE_TIME_MAPPING = contract['TRADE_TIME']
INSTRUMENTS_CODES = contract['INSTRUMENTS']
RINSTRUMENTS_CODES = dict(zip(INSTRUMENTS_CODES.values(),INSTRUMENTS_CODES.keys()))
