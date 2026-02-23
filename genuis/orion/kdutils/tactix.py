### 策略参数管理
import os, yaml, pdb
import pdb, argparse


def load_config(file_name, config_id):
    if not os.path.exists(file_name):
        print(f"Config file {file_name} does not exist.")
        return None

    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            all_configs = yaml.safe_load(f)

        if not all_configs or 'experiment_setups' not in all_configs:
            print(f"No valid configurations found in {file_name}.")
            return None

        experiment_setups = all_configs.get('experiment_setups', {})

        selected_config = experiment_setups.get(config_id)

        if selected_config is None:
            print(f"Configuration ID {config_id} not found in {file_name}.")
            available_ids = list(experiment_setups.keys())
            if available_ids:
                print(
                    f"Available configuration IDs: {', '.join(available_ids)}")
            else:
                print("No configuration IDs available.")
        return selected_config

    except yaml.YAMLError as e:
        print(f"error: parse YAML file '{file_name}' failed: {e}")
        return None
    except Exception as e:
        print("error: parse YAML file failed:{e} ")
        return None


class Tactix(object):
    ### yaml配置文件
    def yaml(self, args):
        yaml_config = load_config(args.config_file, args.config_id)
        if not yaml_config:
            raise ValueError(
                f"load {args.config_id} config from {args.config_file} failed")
        if yaml_config:
            print(
                f"load {args.config_id} config from {args.config_file} success"
            )

        for key, value in yaml_config.items():
            if key == "description":
                print("description: {0}".format(value))
                continue
            if hasattr(args, key):
                old_value = getattr(args, key)
                setattr(args, key, value)
                print(f"update {key} from {old_value} to {value}")
            else:
                setattr(args, key, value)
                print(f"args has no attribute {key}, add it")

            print("-" * 30)
        #else:
        #    print(f"no config found in {args.config_file}")
        #    print("-" * 30)
        return args

    def params(self):
        parser = argparse.ArgumentParser(description='Train a model')

        parser.add_argument('--config_file',
                            type=str,
                            default=None,  # 或者给一个默认值 'configs.yaml'
                            help='Path to the YAML configuration file.')

        parser.add_argument('--config_id',
                            type=str,
                            default=None,
                            help='ID of the configuration setup to use from the YAML file.')
        

        parser.add_argument('--method',
                            type=str,
                            default='xxxx0',
                            help='data method')

        parser.add_argument('--task_id',
                            type=str,
                            default='1111111',
                            help='task id')

        parser.add_argument('--instruments',
                            type=str,
                            default='xxxx',
                            help='code or instruments')

        parser.add_argument('--period', type=int, default=1, help='period')


        return parser

    def default(self, callback=None):
        parser = callback() if callback else self.params()
        return parser

    def dump(self, variant):
        for arg_name, arg_value in vars(variant).items():
            if arg_name in ['config_file', 'config_id']:
                continue
            print(f"   {arg_name}: {arg_value}")

    def start(self, callback=None, log=True):
        parser = self.default(callback=callback)
        args = parser.parse_args()

        self.yaml(args)
        #variant = vars(args)
        variant = args
        if log:
            self.dump(variant)
        return variant
