import mlflow
import json, hashlib, os, pdb
import pandas as pd
from lumina.genetic.util import create_id 

class Params(object):
    def __init__(self, base_path: str, experiment_name: str):
        self.base_path = os.path.join(base_path, experiment_name)

        store_path = os.path.join(self.base_path, "params")
        abs_uri = os.path.abspath(store_path)

        
        self.tracking_uri = f"file://{abs_uri}"
        self.experiment_name = experiment_name
        print(f"initialized: {self.experiment_name} @ {self.tracking_uri}")

    @staticmethod
    def _create_tag(params):
        """内部静态方法：生成参数指纹"""
        param_str = json.dumps(params, sort_keys=True)
        m = hashlib.md5()
        m.update(bytes(param_str, encoding='UTF-8'))
        return create_id(original=m.hexdigest(), digit=16)

    def _activate_context(self):
        """私有方法：激活当前对象的配置环境"""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
    
    @classmethod
    def create_tag(cls, params):
        return cls._create_tag(params=params)

    def save_params_with_content(self, params: dict, artifacts: dict=None):
        # 1. 切换到当前对象的环境 (关键步骤)
        self._activate_context()
        # 2. 生成指纹
        params_tag = self._create_tag(params)
        
        # 3. 执行存储
        with mlflow.start_run(run_name=params_tag):
            mlflow.set_tag("params_tag", params_tag)
            mlflow.log_params(params)
            outputs_dirs = os.path.join(self.base_path, params_tag)
            if not os.path.exists(outputs_dirs):
                os.makedirs(outputs_dirs)
            for filename, df in artifacts.items():
                if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
                    local_path = os.path.join(outputs_dirs, "{0}.feather".format(filename))
                    if isinstance(df, pd.Series):
                        df = df.to_frame(name=filename)
                    df.reset_index(drop=True).to_feather(local_path)
                mlflow.log_artifact(local_path)
                print(f"Artifact stored: {local_path}")

    def load_content(self, params: dict, name: str) -> pd.DataFrame:
        params_tag = self._create_tag(params)
        target_path = os.path.join(self.base_path, params_tag, f"{name}.feather")
        return  pd.read_feather(target_path)