import pdb,os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from lib.utils.ttimes import get_dates
from lib.utils.macro import base_path
from lib.data.scan_data import scan_file
from feature.utils.checker import check_factor_anomalies

def check_factors(method, instruments):
    output_path = os.path.join(base_path, method, instruments, 'factors')
    for factor_file in Path(output_path).glob('*.feather'):
        df_lazy = scan_file(factor_file)
        report = check_factor_anomalies(df_lazy)
        report.print_summary()
        bad_rows = report.get_bad_rows(limit=20)
        print(bad_rows)
        print('-' * 80)

if __name__ == '__main__':
    check_factors(method='ricso2', instruments='rbb')