"""
优化日志和统计工具
"""
import time
from collections import defaultdict
from typing import Dict, List


class OptimizationLogger:
    """优化过程日志记录器"""
    
    def __init__(self):
        self.stats = {
            'total_trials': 0,
            'valid_trials': 0,
            'filter_1_ic_invalid': 0,
            'filter_2_ic_too_small': 0,
            'filter_3_calmar_invalid': 0,
            'filter_4_sharpe_invalid': 0,
            'exceptions': 0,
            'best_ic': 0.0,
            'best_sharpe2': 0.0,
            'best_calmar': 0.0
        }
        self.trial_history = []
        self.start_time = None
        self.failed_patterns = defaultdict(int)
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        print("\n" + "="*80)
        print("🚀 开始因子优化")
        print("="*80 + "\n")
    
    def log_trial(self, trial_num: int, result_type: str, values: List[float], 
                  expression: str = ""):
        """记录单次试验"""
        self.stats['total_trials'] += 1
        
        expr_short = expression[:60] + "..." if len(expression) > 60 else expression
        
        if result_type == 'valid':
            self.stats['valid_trials'] += 1
            self.stats['best_ic'] = max(self.stats['best_ic'], values[0])
            self.stats['best_sharpe2'] = max(self.stats['best_sharpe2'], values[1])
            self.stats['best_calmar'] = max(self.stats['best_calmar'], values[2])
            
        elif result_type == 'filter_1':
            self.stats['filter_1_ic_invalid'] += 1
        elif result_type == 'filter_2':
            self.stats['filter_2_ic_too_small'] += 1
        elif result_type == 'filter_3':
            self.stats['filter_3_calmar_invalid'] += 1
        elif result_type == 'filter_4':
            self.stats['filter_4_sharpe_invalid'] += 1
        elif result_type == 'exception':
            self.stats['exceptions'] += 1
        
        self.trial_history.append({
            'trial': trial_num,
            'type': result_type,
            'values': values,
            'expression': expr_short
        })
    
    def record_failed_pattern(self, pattern: str):
        """记录失败的模式（算子/字段组合）"""
        self.failed_patterns[pattern] += 1
    
    def print_progress(self, trial_num: int, interval: int = 10):
        """每N次试验打印进度"""
        if trial_num % interval == 0 and trial_num > 0:
            valid_rate = self.stats['valid_trials'] / self.stats['total_trials'] * 100
            elapsed = time.time() - self.start_time if self.start_time else 0
            
            print("\n" + "─"*80)
            print(f"📊 进度报告 - Trial {trial_num}/{self.stats['total_trials']}")
            print("─"*80)
            print(f"⏱️  用时: {elapsed:.1f}秒 | 有效率: {valid_rate:.1f}% ({self.stats['valid_trials']}/{self.stats['total_trials']})")
            print(f"🏆 当前最佳: IC={self.stats['best_ic']:.4f}, Sharpe2={self.stats['best_sharpe2']:.4f}, Calmar={self.stats['best_calmar']:.4f}")
            print("─"*80 + "\n")
    
    def print_summary(self):
        """打印最终统计摘要"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        total = self.stats['total_trials']
        
        # 计算百分比（避免除以零）
        valid_rate = (self.stats['valid_trials'] / total * 100) if total > 0 else 0
        filter1_pct = (self.stats['filter_1_ic_invalid'] / total * 100) if total > 0 else 0
        filter2_pct = (self.stats['filter_2_ic_too_small'] / total * 100) if total > 0 else 0
        filter3_pct = (self.stats['filter_3_calmar_invalid'] / total * 100) if total > 0 else 0
        filter4_pct = (self.stats['filter_4_sharpe_invalid'] / total * 100) if total > 0 else 0
        exception_pct = (self.stats['exceptions'] / total * 100) if total > 0 else 0
        
        print("\n" + "="*80)
        print("📈 优化完成 - 统计摘要")
        print("="*80)
        print(f"\n⏱️  总用时: {elapsed:.2f}秒")
        print(f"📊 总试验数: {total}")
        print(f"✅ 有效试验: {self.stats['valid_trials']} ({valid_rate:.1f}%)")
        print(f"\n过滤原因分布:")
        print(f"  ❌ Filter-1 (IC无效):      {self.stats['filter_1_ic_invalid']:3d} ({filter1_pct:.1f}%)")
        print(f"  ❌ Filter-2 (IC太小):      {self.stats['filter_2_ic_too_small']:3d} ({filter2_pct:.1f}%)")
        print(f"  ❌ Filter-3 (Calmar无效):  {self.stats['filter_3_calmar_invalid']:3d} ({filter3_pct:.1f}%)")
        print(f"  ❌ Filter-4 (Sharpe无效):  {self.stats['filter_4_sharpe_invalid']:3d} ({filter4_pct:.1f}%)")
        print(f"  ❌ 计算异常:               {self.stats['exceptions']:3d} ({exception_pct:.1f}%)")
        
        print(f"\n🏆 最佳结果:")
        print(f"  IC Mean:   {self.stats['best_ic']:.6f}")
        print(f"  Sharpe2:   {self.stats['best_sharpe2']:.6f}")
        print(f"  Calmar:    {self.stats['best_calmar']:.6f}")
        
        if self.failed_patterns:
            print(f"\n🔍 失败最多的组合（Top 5）:")
            sorted_patterns = sorted(self.failed_patterns.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
            for pattern, count in sorted_patterns:
                print(f"  - {pattern}: {count}次")
        
        print("\n" + "="*80 + "\n")
    
    def get_stats(self) -> Dict:
        """获取统计数据"""
        return self.stats.copy()
    
    def get_history(self) -> List[Dict]:
        """获取完整历史"""
        return self.trial_history.copy()

