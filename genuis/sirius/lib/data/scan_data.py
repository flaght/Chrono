import polars as pl


def scan_file(file_path):
    df_lazy = pl.scan_ipc(file_path)
    return df_lazy
