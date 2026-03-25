import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def analyze_horizon_feasibility(data_path: str, fee_rate: float, top_k: int):
    """
    分析不同预测周期的绝对波幅，评估其是否能在扣除手续费后存活。
    """
    print(f"Loading dataset from: {data_path} ...")
    
    # 兼容 Parquet 或 CSV
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # 获取所有的 target return 列 (例如 nxt1_ret, nxt6_ret, nxt24_ret)
    target_cols = [col for col in df.columns if col.startswith('nxt') and col.endswith('_ret')]
    
    # 按照周期长度排序
    try:
        target_cols = sorted(target_cols, key=lambda x: int(x.replace('nxt', '').replace('_ret', '')))
    except ValueError:
        pass # 如果有非数字的后缀就不强行排序了
    
    if not target_cols:
        print("❌ 未在数据集中找到 nxt*_ret 列！请检查您的特征数据表中是否包含未来收益率标签。")
        return
        
    print(f"✅ 找到 {len(target_cols)} 个收益周期标签。开始分析 Top/Bottom {top_k} 的极限波幅...\n")
    
    results = []
    
    # 针对每一个周期
    for target in target_cols:
        print(f"  - 分析 {target} ...")
        valid_df = df[['trade_time', 'code', target]].dropna()
        if valid_df.empty:
            continue
            
        def calc_cross_section(group):
            sorted_rets = group[target].sort_values(ascending=False).values
            n = len(sorted_rets)
            if n < top_k * 2: return pd.Series({'top_mean': np.nan, 'bottom_mean': np.nan, 'market_mean': np.nan, 'std': np.nan})
            
            # 【上帝视角】如果完全预知未来，纯做多 TopK 的毛利下限
            top_mean = np.mean(sorted_rets[:top_k])
            # 【魔鬼视角】做多最差的 BottomK 会亏多少
            bottom_mean = np.mean(sorted_rets[-top_k:])
            # 【大盘 Beta】瞎买的平均收益
            market_mean = np.mean(sorted_rets)
            # 截面离散度
            cs_std = np.std(sorted_rets)
            
            return pd.Series({
                'top_mean': top_mean, 
                'bottom_mean': bottom_mean, 
                'market_mean': market_mean,
                'std': cs_std
            })
            
        # 按照时间截面 groupby 计算
        metrics = valid_df.groupby('trade_time').apply(calc_cross_section).dropna()
        
        avg_top = metrics['top_mean'].mean()
        avg_bottom = metrics['bottom_mean'].mean()
        avg_market = metrics['market_mean'].mean()
        avg_std = metrics['std'].mean()
        
        # 核心假设：一个极品量化模型，顶多能抓到“上帝极端波幅”的 20%
        # 这就是模型在完全不加护城河时的 "预期真实毛利润"
        expected_actual_alpha = avg_top * 0.20
        
        # 为了不亏钱，以这个真实利润去覆盖摩擦费，你每一期最多能承受多少换手率？
        # 单边换手率 * Fee * 2 (一买一卖) = 摩擦成本
        # Breakeven Turnover = Profit / (Fee * 2)
        if fee_rate > 0:
            breakeven_turnover = expected_actual_alpha / (fee_rate * 2)
        else:
            breakeven_turnover = 999
            
        try:
            horizon_hours = int(target.replace('nxt', '').replace('_ret', ''))
        except:
            horizon_hours = target
            
        results.append({
            'Target': target,
            'Horizon(H)': horizon_hours,
            'TopK_Perfect_Ret': avg_top,
            'Market_Beta': avg_market,
            'Est_Model_Gross(20%)': expected_actual_alpha,  # 预估真实能拿到的毛利
            'Friction_Cost(100%TO)': fee_rate * 2,          # 如果全换手的死寂成本
            'Breakeven_Turnover': breakeven_turnover        # 盈亏平衡允许的最高换手率
        })
        
    res_df = pd.DataFrame(results)
    if res_df.empty:
        return
        
    # ======== 打印极其残酷的分析结果 ========
    print("\n" + "="*70)
    print(f"【降维打击分析报告】 目标：寻找能装得下 {fee_rate*100}% 单边费率的海洋")
    print(f"假设前提：极品 RL 模型能捕获上帝最高波幅的 20%、双边滑点与费率合计 {fee_rate*2*100}%")
    print("="*70)
    
    # 格式化百分比便于阅读
    format_df = res_df.copy()
    for col in ['TopK_Perfect_Ret', 'Market_Beta', 'Est_Model_Gross(20%)', 'Friction_Cost(100%TO)']:
        format_df[col] = (format_df[col] * 100).map("{:.3f}%".format)
    format_df['Breakeven_Turnover'] = format_df['Breakeven_Turnover'].map("{:.2f}".format)
    
    print(format_df.to_markdown(index=False))
    
    print("\n====== 💡 终极判决指南 ======")
    print("1. 观察 `Est_Model_Gross(20%)` (预估毛利) 是否大于 `Friction_Cost(100%TO)` (全换手摩擦)。")
    print("2. 观察 `Breakeven_Turnover` (容忍换手率)。")
    print("   👉 如果它 < 0.5：这是极度危险的绞肉机，费率几乎吃掉所有利润。")
    print("   👉 如果它在 1.0 左右：勉强能做，但你的模型需要极高胜率且加重 Turnover Penalty。")
    print("   👉 如果它 > 2.0：这才是你可以大显身手的宽阔海洋！这就是你应该选择的 Target。")
    print("===========================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析不同收益率周期对费率的容忍度')
    parser.add_argument('--data', type=str, required=True, help='包含 nxt*_ret 特征的 parquet/csv 本地路径')
    parser.add_argument('--fee', type=float, default=0.0015, help='单边交易费率 (默认 0.0015 即万 15)')
    parser.add_argument('--topk', type=int, default=20, help='评估的资产数量 Top K (默认 20)')
    
    args = parser.parse_args()
    analyze_horizon_feasibility(args.data, fee_rate=args.fee, top_k=args.topk)
