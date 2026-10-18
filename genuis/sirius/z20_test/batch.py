import pdb
import polars as pl
import feature.tc001 as tc001
import feature.tc002 as tc002
import feature.tc003 as tc003
import feature.tc004 as tc004
import feature.tc005 as tc005
import feature.tf001 as tf001
import feature.tf002 as tf002


def run_batch(data, i00):
    frames = [getattr(i00, f)(data) for f in i00.__all__]
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="align").sort(list(['trade_time', 'code']))


def calculate_factors(data):
    for i00 in [tc001, tc002, tc003, tc004, tc005, tf001, tf002]:
        factors_data = run_batch(data=data, i00=i00)
        factors_data = factors_data.collect()
        print(factors_data.tail())


if __name__ == "__main__":
    file_path = (
        "/workspace/worker/pj/Chrono/genuis/mizar/records/"
        "ricso2/rbb/basic/train_data.feather"
    )
    df_lazy = pl.scan_ipc(file_path)
    calculate_factors(data=df_lazy)
