import toml, os

contract_file = os.path.join('config/contract.toml')
contract = toml.load(contract_file)

MAIN_CONTRACT_MAPPING = contract['MAIN_CONTRACT']
CONT_MULTNUM_MAPPING = contract['CONT_MULTNUM']
