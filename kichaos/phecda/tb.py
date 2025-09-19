from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os, pdb

# 指定日志目录
logdir = "../../records/phecda/tensorboard/sac_base_RB_Hedge041TraderEnv_long_2r_mlppolicy_neuro0004_2p_1a_7_1"
pdb.set_trace()
for root, dirs, files in os.walk(logdir):
    for file in files:
        if file.startswith("events.out"):
            event_path = os.path.join(root, file)
            ea = event_accumulator.EventAccumulator(event_path)
            ea.Reload()
            for tag in ea.Tags()["scalars"]:
                events = ea.Scalars(tag)
                df = pd.DataFrame(events)
                out_path = f"{tag.replace('/', '_')}.csv"
                df.to_csv(out_path, index=False)
                print(f"导出：{out_path}")