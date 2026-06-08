"""
主程序

整合所有模块，执行完整的IM期货收益率预测流程。

使用滚动训练方式（Walk-Forward Validation）进行模型训练。
"""
import pdb
import os
import sys
import warnings
import json
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

# 导入自定义模块
try:
    # 作为模块导入时使用相对导入
    from .data_loader import DataLoader
    from .data_cleaner import DataCleaner
    from .feature_engineering import FeatureEngineer
    from .model_trainer import ModelTrainer
    from .model_evaluator import ModelEvaluator
    from .visualizer import Visualizer
    from . import config
except ImportError:
    # 直接运行时使用绝对导入
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    from feature_engineering import FeatureEngineer
    from model_trainer import ModelTrainer
    from model_evaluator import ModelEvaluator
    from visualizer import Visualizer
    import config

warnings.filterwarnings('ignore')

# ============================================================================
# Session和日志管理
# ============================================================================

class TeeOutput:
    """
    同时输出到控制台和文件的类
    """
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def setup_session(base_output_dir='./temp'):
    """
    设置session目录和日志文件
    
    参数:
        base_output_dir: 基础输出目录
    
    返回:
        tuple: (session_dir, log_file_path, log_file_handle)
    """
    # 创建session目录（使用时间戳）
    session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(base_output_dir, f'session_{session_name}')
    os.makedirs(session_dir, exist_ok=True)
    
    # 创建日志文件
    log_file_path = os.path.join(session_dir, '00_run_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # 重定向stdout到控制台和文件
    original_stdout = sys.stdout
    tee = TeeOutput(original_stdout, log_file)
    sys.stdout = tee
    
    return session_dir, log_file_path, log_file, original_stdout

def cleanup_session(log_file, original_stdout):
    """
    清理session，恢复stdout
    
    参数:
        log_file: 日志文件句柄
        original_stdout: 原始stdout
    """
    sys.stdout = original_stdout
    if log_file:
        log_file.close()

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：执行完整的预测流程
    """
    # 设置session和日志
    base_output_dir = './temp'
    session_dir, log_file_path, log_file, original_stdout = setup_session(base_output_dir)
    
    try:
        # 更新config中的OUTPUT_DIR为session目录
        config.OUTPUT_DIR = session_dir
        
        print("=" * 80)
        print("IM期货收益率预测 - 模块化版本（滚动训练）")
        print("=" * 80)
        print(f"Session目录: {session_dir}")
        print(f"日志文件: {log_file_path}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # ========================================================================
        # 第1步：数据加载
        # ========================================================================
        print("\n" + "=" * 80)
        print("第1步：数据加载")
        print("=" * 80)
        
        loader = DataLoader(use_mock_data=config.USE_MOCK_DATA)
        
        # 根据实际情况选择数据加载方式
        # 方式1: 从文件加载
        # df = loader.load('/path/to/your/data.feather')
        
        # 方式2: 从项目路径加载（如果项目模块可用）
        # df = loader.load({
        #     'method': 'your_method',
        #     'task_id': 123,
        #     'instruments': 'IM',
        #     'period': 15,
        #     'name': 'final'
        # })
        
        # 方式3: 使用模拟数据（演示用）
        if config.USE_MOCK_DATA:
            df = loader.load()
        else:
            # 请在这里修改为您的实际数据加载方式
            print("⚠️  请修改main.py中的数据加载代码为您的实际数据源")
            print("   示例:")
            print("   df = loader.load('/path/to/your/data.feather')")
            df = loader.load("/workspace/worker/kdwk/Chrono/genuis/mizar/records/cicso0/ims/temp/model/200037/15/draft_data.feather")
            #return
        # 验证数据
        if not loader.validate_data(df, config.TARGET_COL):
            print("✗ 数据验证失败，程序退出")
            return
        
        # ========================================================================
        # 第2步：数据清洗
        # ========================================================================
        cleaner = DataCleaner(
            nan_threshold=config.NAN_THRESHOLD,
            var_threshold=config.VAR_THRESHOLD,
            target_col=config.TARGET_COL
        )
        df = cleaner.clean(df)
    
        # ========================================================================
        # 第3步：特征工程
        # ========================================================================
        engineer = FeatureEngineer(
        corr_threshold=config.CORR_THRESHOLD,
        ic_threshold=config.IC_THRESHOLD,
        target_col=config.TARGET_COL
        )
        selected_features, ic_dict = engineer.select_features(df)
    
        # 保存选择的特征
        feature_df = pd.DataFrame({'feature': selected_features})
        feature_path = os.path.join(config.OUTPUT_DIR, '02_selected_features.csv')
        feature_df.to_csv(feature_path, index=False)
        print(f"\n✓ 选择的特征列表已保存至: {feature_path}")
        # 保存IC结果
        ic_df = pd.DataFrame({
        'feature': list(ic_dict.keys()),
        'IC': list(ic_dict.values())
        }).sort_values('IC', ascending=False)
        ic_path = os.path.join(config.OUTPUT_DIR, '02_factor_ic.csv')
        ic_df.to_csv(ic_path, index=False)
        print(f"✓ 因子IC已保存至: {ic_path}")
    
        # ========================================================================
        # 第4步：准备训练数据
        # ========================================================================
        trainer = ModelTrainer()
        X, y, dates = trainer.prepare_data(df, selected_features)
    
        # ========================================================================
        # 第5步：模型训练（滚动训练 - Walk-Forward Validation）
        # ========================================================================
        # 【关键】使用滚动训练方式，这是时间序列预测的正确方法
        _, wf_results = trainer.train_rolling(
        X, y, dates,
        selected_features=selected_features,
        n_splits=config.N_SPLITS
        )
    
        # 保存Walk-Forward验证结果
        wf_path = os.path.join(config.OUTPUT_DIR, '04_walk_forward_results.csv')
        wf_results.to_csv(wf_path, index=False)
        print(f"\n✓ Walk-Forward验证结果已保存至: {wf_path}")
    
        # ========================================================================
        # 第6步：在完整测试集上评估（使用最后一个模型）
        # ========================================================================
        # 为了与原始代码保持一致，我们也进行一次单次训练和评估
        # 但注意：滚动训练的结果更可靠
    
        print("\n" + "=" * 80)
        print("第6步：单次训练（用于完整评估）")
        print("=" * 80)
        print("【说明】为了完整评估，我们也会进行一次单次训练")
        print("但滚动训练的结果更可靠，因为它模拟了真实交易场景")
    
        # 划分数据
        X_train, X_test, y_train, y_test, dates_train, dates_test = trainer.split_data(
        X, y, dates, train_ratio=config.TRAIN_RATIO
        )
    
        # 单次训练
        model = trainer.train_single(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        selected_features=selected_features
        )
    
        # 预测
        y_train_pred = trainer.predict(X_train, model)
        y_test_pred = trainer.predict(X_test, model)
    
        # ========================================================================
        # 第7步：模型评估
        # ========================================================================
        evaluator = ModelEvaluator()
        eval_results = evaluator.evaluate(
        y_train, y_train_pred,
        y_test, y_test_pred,
        dates_test=dates_test
        )
    
        # ========================================================================
        # 第8步：特征重要性分析
        # ========================================================================
        print("\n" + "=" * 80)
        print("第8步：特征重要性分析")
        print("=" * 80)
    
        feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance_gain': model.feature_importance(importance_type='gain'),
        'importance_split': model.feature_importance(importance_type='split')
        }).sort_values('importance_gain', ascending=False)
    
        print(f"\n特征重要性统计:")
        print(f"  总特征数: {len(feature_importance)}")
        print(f"  重要性总和: {feature_importance['importance_gain'].sum():.2f}")
    
        top_n = config.TOP_N_FEATURES
        top_features = feature_importance.head(top_n)
        top_importance_sum = top_features['importance_gain'].sum()
        total_importance = feature_importance['importance_gain'].sum()
    
        print(f"\n  Top {top_n} 特征:")
        print(f"    累计重要性: {top_importance_sum:.2f}")
        print(f"    占比: {top_importance_sum/total_importance*100:.1f}%")
    
        print(f"\n  Top 20 重要特征详情:")
        print(top_features.head(20).to_string(index=False))
    
        # 保存特征重要性
        importance_path = os.path.join(config.OUTPUT_DIR, '03_feature_importance.csv')
        feature_importance.to_csv(importance_path, index=False)
        print(f"\n✓ 特征重要性已保存至: {importance_path}")
    
        # ========================================================================
        # 第9步：可视化
        # ========================================================================
        print("\n" + "=" * 80)
        print("第9步：可视化分析")
        print("=" * 80)
    
        visualizer = Visualizer(output_dir=config.OUTPUT_DIR)
    
        # 生成完整评估图表
        eval_plot_path = os.path.join(config.OUTPUT_DIR, '03_model_evaluation.png')
        visualizer.plot_evaluation_summary(
        y_test, y_test_pred,
        eval_results['test_strategy_returns'],
        eval_results['test_drawdown'],
        eval_results['quantile_stats'],
        eval_results['confusion_matrix'],
        eval_results['test']['ic'],
        eval_results['test']['direction_acc'],
        eval_results['test']['rmse'],
        eval_results['test']['sharpe'],
        eval_results['test']['max_drawdown'],
        eval_results['q5_q1_diff'],
        save_path=eval_plot_path
        )
    
        # 生成特征重要性图表
        importance_plot_path = os.path.join(config.OUTPUT_DIR, '03_feature_importance_plot.png')
        visualizer.plot_feature_importance(
        feature_importance,
        top_n=20,
        save_path=importance_plot_path
        )
    
        # ========================================================================
        # 第10步：保存模型和元数据
        # ========================================================================
        print("\n" + "=" * 80)
        print("第10步：模型保存")
        print("=" * 80)
    
        # 保存LightGBM模型
        model_path = os.path.join(config.OUTPUT_DIR, '05_lgb_model.txt')
        model.save_model(model_path)
        print(f"  ✓ LightGBM模型已保存至: {model_path}")
    
        # 保存模型元数据
        metadata = {
        'model_info': {
            'algorithm': 'LightGBM GBDT',
            'version': lgb.__version__,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_iteration': int(model.best_iteration),
            'training_method': 'rolling_train + single_train'
        },
        'data_info': {
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'n_features_selected': len(selected_features),
            'train_period': f"{dates_train[0]} to {dates_train[-1]}",
            'test_period': f"{dates_test[0]} to {dates_test[-1]}",
        },
        'performance_metrics': {
            'test_ic': float(eval_results['test']['ic']),
            'test_rank_ic': float(eval_results['test']['rank_ic']),
            'test_rmse': float(eval_results['test']['rmse']),
            'test_direction_acc': float(eval_results['test']['direction_acc']),
            'test_sharpe': float(eval_results['test']['sharpe']),
            'test_cum_return': float(eval_results['test']['cum_return']),
            'test_max_drawdown': float(eval_results['test']['max_drawdown']),
            'wf_avg_ic': float(wf_results['IC'].mean()),
            'wf_avg_sharpe': float(wf_results['sharpe'].mean()),
        },
        'model_params': config.LGB_PARAMS,
        'rolling_train_info': {
            'n_splits': config.N_SPLITS,
            'wf_avg_ic': float(wf_results['IC'].mean()),
            'wf_avg_direction_acc': float(wf_results['direction_acc'].mean()),
            'wf_avg_sharpe': float(wf_results['sharpe'].mean()),
        }
        }
        
        metadata_path = os.path.join(config.OUTPUT_DIR, '05_model_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"  ✓ 模型元数据已保存至: {metadata_path}")
    
        # ========================================================================
        # 第11步：加载已保存模型并进行预测（示例）
        # ========================================================================
        print("\n" + "=" * 80)
        print("第11步：加载已保存模型并进行预测（示例）")
        print("=" * 80)
        print("【说明】以下示例展示如何在单独的推理脚本中加载模型并生成预测。")
        print("       实际使用时，请将 inference_samples 替换为最新的实时数据。")
        
        inference_trainer = ModelTrainer()
        loaded_model = inference_trainer.load_model(
            model_path,
            best_iteration=model.best_iteration
        )
        
        inference_samples = df[['trade_time', 'code'] + selected_features].tail(
            config.INFERENCE_SAMPLE_SIZE
        ).copy()
        inference_X = inference_samples[selected_features].values
        inference_preds = inference_trainer.predict(inference_X, model=loaded_model)
        inference_samples['predicted_return'] = inference_preds
        
        inference_output_path = os.path.join(config.OUTPUT_DIR, '06_inference_example.csv')
        inference_samples[['trade_time', 'code', 'predicted_return']].to_csv(
            inference_output_path,
            index=False
        )
        
        print(f"  ✓ 示例完成：已重新加载模型并对最近 {config.INFERENCE_SAMPLE_SIZE} 条样本生成预测")
        print(f"  ✓ 推理结果示例已保存至: {inference_output_path}")
        print("  ⚠️  提示：若在独立环境推理，请读取 02_selected_features.csv 确保特征顺序一致")

        # ========================================================================
        # 完成
        # ========================================================================
        print("\n" + "=" * 80)
        print("✓✓✓ 全部流程完成！✓✓✓")
        print("=" * 80)
    
        print("\n核心评估指标（测试集）:")
        print("=" * 80)
        print(f"  ⭐⭐⭐ 方向准确率: {eval_results['test']['direction_acc']*100:.2f}% {eval_results['rating']}")
        print(f"  ⭐⭐⭐ Sharpe Ratio: {eval_results['test']['sharpe']:.2f} {eval_results['sharpe_rating']}")
        calmar_display = f"{eval_results['test']['calmar']:.2f}" if pd.notna(eval_results['test']['calmar']) else "N/A"
        calmar_rating_display = eval_results['calmar_rating'] if eval_results['calmar_rating'] else ''
        print(f"  ⭐⭐⭐ Calmar Ratio: {calmar_display} {calmar_rating_display}")
        print(f"  ⭐⭐⭐ 策略累计收益: {eval_results['test']['cum_return']:.6f}")
        print(f"       IC: {eval_results['test']['ic']:.4f}")
        print(f"       最大回撤: {eval_results['test']['max_drawdown']:.6f}")
        print(f"       胜率: {eval_results['test']['win_rate']*100:.2f}%")
    
        print("\n滚动训练结果（Walk-Forward Validation）:")
        print("=" * 80)
        print(f"  平均IC: {wf_results['IC'].mean():.4f} ± {wf_results['IC'].std():.4f}")
        print(f"  平均方向准确率: {wf_results['direction_acc'].mean():.2%} ± {wf_results['direction_acc'].std():.2%}")
        print(f"  平均Sharpe: {wf_results['sharpe'].mean():.2f} ± {wf_results['sharpe'].std():.2f}")
    
        print("\n输出文件清单:")
        print("=" * 80)
        print("  [特征工程]")
        print("    1. 02_factor_ic.csv - 因子IC值")
        print("    2. 02_selected_features.csv - 筛选后的特征列表")
        print("  [模型评估]")
        print("    3. 03_model_evaluation.png - 模型评估图表（6张子图）")
        print("    4. 03_feature_importance.csv - 特征重要性")
        print("    5. 03_feature_importance_plot.png - 特征重要性可视化")
        print("  [交叉验证]")
        print("    6. 04_walk_forward_results.csv - Walk-Forward验证结果")
        print("  [模型保存]")
        print("    7. 05_lgb_model.txt - 训练好的LightGBM模型")
        print("    8. 05_model_metadata.json - 模型元数据")
        print("  [日志]")
        print("    9. 00_run_log.txt - 完整运行日志")
    
        print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "=" * 80)
        print(f"\n✓ 所有输出已保存到session目录: {session_dir}")
        print(f"✓ 日志文件: {log_file_path}")
        
    finally:
        # 清理session，恢复stdout
        cleanup_session(log_file, original_stdout)


if __name__ == '__main__':
    main()

