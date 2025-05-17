### 策略参数管理
import pdb, argparse
from kichaos.utils.fyaml import *


class Tactix(object):
    ### yaml配置文件
    def yaml(self, args):
        yaml_config = load_config(args.config_file, args.config_id)
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
        else:
            print(f"no config found in {args.config_file}")
            print("-" * 30)
        return args

    def params(self):
        parser = argparse.ArgumentParser(description='Train a model')

        parser.add_argument('--freq',
                            type=int,
                            default=5,
                            help='Frequency of training')  ## 多少个周期训练一次
        parser.add_argument('--train_days',
                            type=int,
                            default=60,
                            help='Training days')  ## 训练天数
        parser.add_argument('--val_days',
                            type=int,
                            default=5,
                            help='Validation days')  ## 验证天数

        parser.add_argument('--method',
                            type=str,
                            default='aicso3',
                            help='Method name')  ## 方法
        parser.add_argument('--categories',
                            type=str,
                            default='o2o',
                            help='Categories')  ## 类别
        parser.add_argument('--horizon',
                            type=int,
                            default=1,
                            help='Prediction horizon')  ## 预测周期

        parser.add_argument('--nc',
                            type=int,
                            default=1,
                            help='Standard method')  ## 标准方式
        parser.add_argument('--swindow',
                            type=int,
                            default=0,
                            help='Rolling window')  ## 滚动窗口

        parser.add_argument('--check_freq',
                            type=int,
                            default=0,
                            help='Model save frequency')  ## 每次模型保存的频率即评估模型
        parser.add_argument('--batch_size',
                            type=int,
                            default=512,
                            help='Batch size during training')  ## 训练时数据大小
        parser.add_argument(
            '--learning_starts',
            type=int,
            default=1000,
            help='Learning start time')  ## 学习开始时间 目前该参数放入到tessla内部
        parser.add_argument('--epchos',
                            type=int,
                            default=10,
                            help='Number of epochs')  ## 迭代多少轮
        parser.add_argument('--window',
                            type=int,
                            default=3,
                            help='Starting period')  ## 开始周期，即多少个周期构建env数据

        parser.add_argument('--direction',
                            type=str,
                            default='long',
                            help='Direction')  ## 方向
        parser.add_argument('--code', type=str, default='RB',
                            help='Code')  ## 代码
        parser.add_argument('--g_instruments',
                            type=str,
                            default='rbb',
                            help='Instruments')  ## 标的

        parser.add_argument('--param_index',
                            type=int,
                            default=1,
                            help='Model parameter index')  ## 模型参数索引
        parser.add_argument('--addition_index',
                            type=int,
                            default=1,
                            help='General parameter index')  ## 通用参数索引
        parser.add_argument('--policy_key',
                            type=str,
                            default='neuro0004_1',
                            help='Policy parameter index')  ## 策略参数索引

        parser.add_argument('--verbosity',
                            type=int,
                            default=40,
                            help='Training output frequency')  ## 训练时的输出频率
        parser.add_argument('--g_start_pos',
                            type=int,
                            default=47,
                            help='Start position')  ## 开始位置
        parser.add_argument('--g_max_pos',
                            type=int,
                            default=1,
                            help='Max position')  ## 最大位置

        parser.add_argument('--config_file',
                            type=str,
                            default='configs.yaml',
                            help="YAML配置文件的路径")
        parser.add_argument('--config_id',
                            type=str,
                            help="要从YAML文件中加载的实验配置ID (可选)")

        return parser

    def default(self, callback=None):
        parser = callback() if callback else self.params()
        return parser

    def dump(self, variant):
        for arg_name, arg_value in variant.items():
            if arg_name in ['config_file', 'config_id']:
                continue
            print(f"   {arg_name}: {arg_value}")

    def start(self, callback=None, log=True):
        parser = self.default(callback=callback)
        args = parser.parse_args()

        self.yaml(args)
        variant = vars(args)
        pdb.set_trace()
        if log:
            self.dump(variant)
        return variant
