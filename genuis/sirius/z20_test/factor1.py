import polars as pl
import pdb

from feature.tf002.or001 import compute

def calculate_factors(df_lazy):
    pdb.set_trace()
    dt = compute(df_lazy)
    dt1 = dt.collect()
    print(dt1)

if __name__ == "__main__":
    file_path = (
        "/workspace/worker/pj/Chrono/genuis/mizar/records/"
        "ricso2/rbb/basic/train_data.feather"
    )
    df_lazy = pl.scan_ipc(file_path)
    calculate_factors(df_lazy)