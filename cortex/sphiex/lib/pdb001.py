import pandas as pd

class PromptDataBuilder:
    """
    负责将时序 DataFrame (pdata, rdata, tdata, memories)
    转化为结构化 Prompt 所需的标准 Markdown/Text 格式
    """

    @staticmethod
    def _format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """通用清洗：格式化 trade_date 日期并精简浮点数"""
        res = df.copy()
        
        # 1. 规范 trade_date：去掉 00:00:00，仅保留 YYYY-MM-DD
        if 'trade_date' in res.columns:
            res['trade_date'] = pd.to_datetime(res['trade_date']).dt.strftime('%Y-%m-%d')
            # 确保 trade_date 排在第一列
            cols = ['trade_date'] + [c for c in res.columns if c != 'trade_date']
            res = res[cols]
            
        # 2. 格式化浮点数列
        for col in res.select_dtypes(include=['float', 'float64']).columns:
            # 针对大额成交量转换为亿元显示
            if 'turnover' in col.lower() or 'volume' in col.lower() or res[col].abs().max() > 1e8:
                res[col] = res[col].apply(lambda x: f"{x/1e8:.2f}亿" if pd.notnull(x) else "NaN")
            else:
                res[col] = res[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
                
        return res

    @staticmethod
    def build_predictive_signals(pdata: pd.DataFrame) -> str:
        """1. 格式化候选预测特征时序 (pdata)"""
        if pdata.empty:
            return "【无预测特征数据】"
        df = PromptDataBuilder._format_dataframe(pdata)
        return df.to_markdown(index=False)

    @staticmethod
    def build_regime_features(rdata: pd.DataFrame) -> str:
        """2. 格式化环境状态特征时序 (rdata)"""
        if rdata.empty:
            return "【无环境状态数据】"
        df = PromptDataBuilder._format_dataframe(rdata)
        return df.to_markdown(index=False)

    @staticmethod
    def build_textual_events(tdata: pd.DataFrame) -> str:
        """3. 格式化文本事件脉络 (tdata)"""
        if tdata.empty:
            return "【当前回溯窗口内无重大文本事件】"
            
        df = tdata.copy()
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
            
        lines = []
        for trade_date, group in df.groupby('trade_date', sort=True):
            lines.append(f"### 📅 交易日: {trade_date}")
            for _, row in group.iterrows():
                fid = row.get('feature_id', 'TF-UNKNOWN')
                ftype = row.get('feature_type', 'macro')
                scope = row.get('affected_scope', 'market')
                imp = row.get('importance', 3)
                status = row.get('status', 'confirmed')
                digest = row.get('digest', '')
                
                lines.append(
                    f"- **[{fid}]** `[{ftype} | {scope}]` ★重要度:{imp} 状态:{status}\n"
                    f"  > **事件简要**: {digest}"
                )
            lines.append("")
            
        return "\n".join(lines)
    
    @staticmethod
    def build_retrieved_memories(memories: list) -> str:
        """
        4. 格式化召回的历史案例记忆 (memories -> 历史先验列表)
        """
        if not memories or len(memories) == 0:
            return "【无历史案例记忆 (处于 Zero-Shot 盲测模式)】"
            
        lines = []
        for i, mem in enumerate(memories):
            mem_id = mem.get("memory_id", f"MEM-{i+1:03d}")
            sim = mem.get("similarity", 0.0)
            features = ", ".join(mem.get("key_features", []))
            exp = mem.get("revised_experience", "")
            adj = mem.get("recommended_adjustment", "")
            direction = mem.get("implied_direction", "FLAT")
            status = mem.get("validation_status", "UNVALIDATED")
            
            lines.append(
                f"#### 🔹 [记忆ID: {mem_id}] (相似度: {sim:.2f} | 状态: {status})\n"
                f"- **核心特征标签**: `{features}`\n"
                f"- **历史经验指向**: `{direction}`\n"
                f"- **复盘经验总结**: {exp}\n"
                f"- **策略应对建议**: {adj}\n"
            )
        return "\n".join(lines)