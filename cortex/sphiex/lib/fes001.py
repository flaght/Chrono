import pdb

TEXTUAL_FEATURE_TYPES = {
    "domestic_macro",
    "domestic_policy",
    "global_liquidity",
    "external_shock",
    "industry_trend",
    "market_sentiment",
}

TEXTUAL_FEATURE_LABELS = {
    "domestic_macro": "国内宏观",
    "domestic_policy": "国内政策",
    "global_liquidity": "全球流动性",
    "external_shock": "外部冲击",
    "industry_trend": "行业趋势",
    "market_sentiment": "市场情绪",
}

# 模态层级标准归一化映射 (容错 PREDICT / PREDICTIVE)
LAYER_NORM_MAP = {
    "PREDICT": "PREDICT",
    "PREDICTIVE": "PREDICT",
    "REGIME": "REGIME",
    "TEXTUAL": "TEXTUAL",
    "TEXT": "TEXTUAL"
}


class FeatureScorer:

    def __init__(self,
                 min_samples: int = 2,
                 min_win_rate: float = 0.50,
                 min_contribution: float = 0.0):
        """
        :param min_samples: 入选 active 的最低被引用样本数门槛
        :param min_win_rate: 入选 active 的最低胜率门槛 (如 50%)
        :param min_contribution: 入选 active 的最低累计贡献分 (大于 0)
        """
        self.min_samples = min_samples
        self.min_win_rate = min_win_rate
        self.min_contribution = min_contribution
        # 三模态独立账本结构: quality_ledgers[layer][feature_name] -> 统计指标字典
        self.quality_ledgers = {
            "PREDICT": {},
            "REGIME": {},
            "TEXTUAL": {},
        }

    def _normalize_feature_item(self, item: dict):
        """
        直接提取大模型显式输出的 feature_layer 与 feature，并做标准归一化
        """
        raw_layer = str(item.get("feature_layer",
                                 "PREDICT")).upper().strip()
        layer = LAYER_NORM_MAP.get(raw_layer, "PREDICT")
        feat = str(item.get("feature", "")).strip()

        # 文本特征合法性校验 (若属于文本层但未落入6大类，打印提示)
        if layer == "TEXTUAL" and feat not in TEXTUAL_FEATURE_TYPES:
            print(f"⚠️ [提示] 文本特征 '{feat}' 建议归类到 6 大稳定文本枚举之一。")

        return layer, feat

    def process_record(self, record_data):
        status = record_data.get("status")
        fwd_ret = record_data.get("forward_return")
        trader_pred = record_data.get("trader_prediction", {})

        # 若结算尚未完成，尝试从 reviewer_result 读取
        if status != "SETTLED" and fwd_ret is None:
            rev_res = record_data.get("reviewer_result", {})
            if isinstance(rev_res, dict):
                fwd_ret = rev_res.get("realized_return", fwd_ret)

        if fwd_ret is None:
            # 真实收益尚未揭晓，跳过
            return

        direction = trader_pred.get("predict_direction", "FLAT")
        confidence = float(trader_pred.get("confidence", 0.5))
        predict_logic = trader_pred.get("predict_logic", [])

        # 1. 判定本次预测胜负 (DirectionMatch)
        is_win = False
        if direction == "UP" and fwd_ret > 0:
            is_win = True
        elif direction == "DOWN" and fwd_ret < 0:
            is_win = True
        elif direction == "FLAT":
            return

        direction_sign = 1.0 if is_win else -1.0

        # 2. 样本内去重：提取本次预测中实际引用的三模态特征复合键 (layer, feat)
        cited_keys = set()
        for item in predict_logic:
            layer, feat = self._normalize_feature_item(item)
            if feat:
                cited_keys.add((layer, feat))

        if not cited_keys:
            return

        # 3. 计算本次样本的总收益贡献分 (置信度 * 收益率幅度 * 胜负符号)
        total_sample_contribution = direction_sign * confidence * abs(fwd_ret)

        # 4. 等权分摊到本次引用的所有特征类别 (避免连坐与多特征通胀)
        num_keys = len(cited_keys)
        delta_contribution = total_sample_contribution / num_keys

        # 5. 更新三模态独立账本
        for layer, feat in cited_keys:
            if layer not in self.quality_ledgers:
                self.quality_ledgers[layer] = {}

            ledger = self.quality_ledgers[layer]
            if feat not in ledger:
                ledger[feat] = {
                    "feature": feat,
                    "layer": layer,
                    "referenced_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "win_rate": 0.0,
                    "cumulative_contribution": 0.0,
                    "mean_contribution": 0.0,
                }

            entry = ledger[feat]
            entry["referenced_count"] += 1
            if is_win:
                entry["positive_count"] += 1
            else:
                entry["negative_count"] += 1

            entry["cumulative_contribution"] += delta_contribution
            entry["win_rate"] = round(
                entry["positive_count"] / entry["referenced_count"], 4)
            entry["mean_contribution"] = round(
                entry["cumulative_contribution"] / entry["referenced_count"],
                6)

    def evaluate_generate(self):
        output_result = {
            "PREDICT": {
                "active": [],
                "inactive": [],
                "insufficient_evidence": []
            },
            "REGIME": {
                "active": [],
                "inactive": [],
                "insufficient_evidence": []
            },
            "TEXTUAL": {
                "active": [],
                "inactive": [],
                "insufficient_evidence": []
            },
            "detailed_reports": []
        }

        for layer, ledger in self.quality_ledgers.items():
            for feat, stats in ledger.items():
                ref_cnt = stats["referenced_count"]
                win_rate = stats["win_rate"]
                cum_contrib = stats["cumulative_contribution"]

                # 评级状态分类：
                # 1. 样本数不足 -> insufficient_evidence
                # 2. 样本充足且胜率/贡献达标 -> active
                # 3. 样本充足但表现不佳 -> inactive
                if ref_cnt < self.min_samples:
                    status_label = "INSUFFICIENT_EVIDENCE"
                    output_result[layer]["insufficient_evidence"].append(feat)
                elif win_rate >= self.min_win_rate and cum_contrib > self.min_contribution:
                    status_label = "ACTIVE"
                    output_result[layer]["active"].append(feat)
                else:
                    status_label = "INACTIVE"
                    output_result[layer]["inactive"].append(feat)

                feat_display = f"{feat} ({TEXTUAL_FEATURE_LABELS.get(feat, '')})" if layer == "TEXTUAL" else feat

                output_result["detailed_reports"].append({
                    "layer":
                    layer,
                    "feature":
                    feat,
                    "display_name":
                    feat_display,
                    "status":
                    status_label,
                    "referenced_count":
                    ref_cnt,
                    "win_count":
                    stats["positive_count"],
                    "loss_count":
                    stats["negative_count"],
                    "win_rate":
                    win_rate,
                    "cumulative_contribution":
                    round(cum_contrib, 6),
                    "mean_contribution":
                    stats["mean_contribution"]
                })

        output_result["detailed_reports"].sort(
            key=lambda x: (x["layer"], -x["cumulative_contribution"]))

        return output_result
