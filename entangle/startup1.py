import os, argparse, setproctitle
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from connor1 import Conor
from plugin.virgtor.engine import Engine as VirgtorEngine


def main():
    codes = ['IM']
    setproctitle.setproctitle("entangle2")
    chaos_qubit = VirgtorEngine(codes=codes)
    qubits = [chaos_qubit]
    conor = Conor(name='ctp', codes=codes, qubits=qubits)
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
