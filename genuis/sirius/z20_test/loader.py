import polars as pl
import time

file_path = "/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/basic/train_data.feather"

df_lazy = pl.scan_ipc(file_path)
print(df_lazy.collect().head())

time1 = time.time()
print(compute(df_lazy=df_lazy).collect())
print(time.time() - time1)
