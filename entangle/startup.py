import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from connor import Conor
from plugin.atlex.engine import Engine as AtlexEngine


def main():
    code = 'RB'
    atlex_qubit = AtlexEngine(code=code)
    conor = Conor(name='ctp', code=code,qubit=atlex_qubit)
    conor.start(
        account_id=os.environ['CTP_ACCOUNT_ID'],
        password=os.environ['CTP_PASSWORD'],
        broker_id=os.environ['CTP_BROKER_ID'],
        app_id=os.environ['CTP_APP_ID'],
        auth_code=os.environ['CTP_AUTH_CODE'],
        td_address=os.environ['CTP_TD_ADDRESS'],
        md_address=os.environ['CTP_MD_ADDRESS'],
    )

if __name__ == '__main__':
    main()
