import sqlite3, json, pdb, os, uuid
from typing import Optional
import numpy as np
from typing import Set
from lib.bra001 import create_brain, load_brain

MY_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

MODALITY_PREDICT = "PREDICT"
MODALITY_REGIME = "REGIME"
MODALITY_TEXT = "TEXT"

MODALITY_TO_STORAGE_TYPE = {
    MODALITY_PREDICT: 1,
    MODALITY_REGIME: 2,
    MODALITY_TEXT: 3,
}


class MemoryProjector:

    @staticmethod
    def project_and_render_markdown(
            memory_dict: dict,
            active_predictive_whitelist: Set[str]) -> Optional[str]:
        ev_list = memory_dict["evidence_units"]
        dec_rule = memory_dict["decision_rule"]

        valid_ev_map = {}
        removed_ev_ids = []
        for ev in ev_list:
            # Regime 和 Textual 豁免淘汰；Predictive 必须在白名单内
            if not active_predictive_whitelist or ev["feature_layer"] in [
                    "REGIME", "TEXTUAL"
            ] or all(feat in active_predictive_whitelist
                     for feat in ev["feature_refs"]):
                valid_ev_map[ev["evidence_id"]] = ev
            else:
                removed_ev_ids.append(ev["evidence_id"])

        # 检查核心必要证据 (熔断检测)
        required_ev_ids = set(dec_rule["required_evidence"])
        if not required_ev_ids.issubset(valid_ev_map.keys()):
            return None  # 必要证据缺失，触发熔断

        active_evidences = [
            ev for ev_id, ev in valid_ev_map.items()
            if ev_id in dec_rule["required_evidence"]
            or ev_id in dec_rule.get("supporting_evidence", [])
        ]

        lines = [
            f"#### 🔹 [历史经验 {memory_dict.get('case_id', 'N/A')}] (方向指引: `{memory_dict['implied_direction']}`)",
            f"- **有效分层证据推演 (已根据最新白名单动态投影)**:"
        ]
        for ev in active_evidences:
            layer_tag = f"[{ev['feature_layer']}:{ev['evidence_role']}]"
            lines.append(
                f"  * {layer_tag} {ev['interpretation']} *(依赖特征: {', '.join(ev['feature_refs'])})*"
            )

        lines.append(f"- **综合决策结论**: {dec_rule['conclusion']}")
        return "\n".join(lines)


def create_matrix_uuid(trade_time, code, feature_matrix):
    # 使用 formatter 强制保留 8 位小数，消除科学计数法的微小差异
    np.set_printoptions(suppress=True)
    matrix_str = np.array2string(
        feature_matrix,
        formatter={'float_kind': lambda x: "%.8f" % x},
        separator=',')
    unique_string = f"{trade_time}_{code}_{matrix_str}"
    deterministic_uuid = uuid.uuid5(MY_NAMESPACE, unique_string)
    return str(deterministic_uuid), matrix_str


class SQLiteMemoryStorage:
    """
    负责原子证据结构化存储与特征映射关联 (1张主表 + 2张关系子表)
    """

    def __init__(self, db_path: str = None, db_name: str = ":memory:"):
        import sqlite3
        self.db_path = db_path
        self.db_name = db_name
        db_name = os.path.join(self.db_path, "{}.db".format(
            self.db_name)) if self.db_name != ':memory:' else db_name
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        # 1. 经验主表 (显式维护三模态独立引用 UUID)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS case_memories (
            case_id TEXT PRIMARY KEY,
            trade_time TEXT,
            symbol TEXT,
            predict_uuid TEXT,
            regime_uuid TEXT,
            textual_uuid TEXT,
            implied_direction TEXT,
            decision_rule TEXT,
            invalidation_rules TEXT,
            regime_matrix_str TEXT,
            predictive_matrix_str TEXT,
            textual_events_str TEXT,
            validation_status TEXT DEFAULT 'UNVALIDATED_SINGLE_CASE',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        # 为三大模态引用建立快速索引
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predict_uuid ON case_memories(predict_uuid)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_regime_uuid ON case_memories(regime_uuid)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_textual_uuid ON case_memories(textual_uuid)"
        )

        # 2. 证据单元表 (支持记录 feature_layer: REGIME / PREDICTIVE / TEXTUAL)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evidence_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            evidence_id TEXT,
            feature_layer TEXT,
            evidence_role TEXT,
            feature_refs TEXT,
            interpretation TEXT,
            FOREIGN KEY (case_id) REFERENCES case_memories(case_id)
        )
        """)
        # 3. 特征-证据引用映射底表 (用于统计 FeatureQuality_k 与动态淘汰)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_evidence_relation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            evidence_id TEXT,
            feature_name TEXT,
            feature_layer TEXT,
            evidence_role TEXT,
            is_required INTEGER,
            FOREIGN KEY (case_id) REFERENCES case_memories(case_id)
        )
        """)
        self.conn.commit()
        
        # 4. 真实曝光审计表 (防大模型幻觉)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_exposure_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT,
            case_id TEXT,
            retrieval_rank INTEGER,
            matched_modalities TEXT,
            exposed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(sample_id, case_id)
        )
        """)
        
        # 5. 结算幂等审计完整日志表 (事务级幂等)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS settlement_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT,
            case_id TEXT,
            predict_direction TEXT,
            memory_direction TEXT,
            relation_to_decision TEXT,
            usage_weight REAL,
            confidence REAL,
            realized_return REAL,
            delta_score REAL,
            status_before TEXT,
            status_after TEXT,
            quality_before REAL,
            quality_after REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(sample_id, case_id)
        )
        """)
        self.conn.commit()

    def save_review_result(self, case_id: str, symbol: str, trade_time: str,
                           predict_uuid: str, regime_uuid: str,
                           textual_uuid: str, regime_matrix_str: str,
                           predict_matrix_str: str, textual_events: list,
                           review_dict: dict):
        cursor = self.conn.cursor()
        exp = review_dict["experience_summary"]
        dec_rule = exp["decision_rule"]
        req_set = set(dec_rule["required_evidence"])
        # 写入主表 (显式存入 3 个独立引用 UUID)
        cursor.execute(
            """
        INSERT OR REPLACE INTO case_memories (
            case_id, trade_time, symbol,
            predict_uuid, regime_uuid, textual_uuid,
            implied_direction, decision_rule, invalidation_rules,
            regime_matrix_str, predictive_matrix_str, textual_events_str,
            validation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                case_id,
                trade_time,
                symbol,
                predict_uuid,
                regime_uuid,
                textual_uuid,
                exp["implied_direction"],
                json.dumps(dec_rule, ensure_ascii=False),
                json.dumps(exp.get("invalidation_rules", []),
                           ensure_ascii=False),
                regime_matrix_str,
                predict_matrix_str,
                #json.dumps(regime_matrix),
                # json.dumps(predict_matrix),
                json.dumps(textual_events, ensure_ascii=False),
                exp.get("validation_status", "UNVALIDATED_SINGLE_CASE")))

        cursor.execute("DELETE FROM evidence_units WHERE case_id = ?",
                       (case_id, ))
        cursor.execute(
            "DELETE FROM feature_evidence_relation WHERE case_id = ?",
            (case_id, ))

        for ev in exp["evidence_units"]:
            cursor.execute(
                """
            INSERT INTO evidence_units (case_id, evidence_id, feature_layer, evidence_role, feature_refs, interpretation)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (case_id, ev["evidence_id"], ev["feature_layer"],
                  ev["evidence_role"],
                  json.dumps(ev["feature_refs"],
                             ensure_ascii=False), ev["interpretation"]))

            is_req = 1 if ev["evidence_id"] in req_set else 0
            for feat in ev["feature_refs"]:
                cursor.execute(
                    """
                INSERT INTO feature_evidence_relation (case_id, evidence_id, feature_name, feature_layer, evidence_role, is_required)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (case_id, ev["evidence_id"], feat, ev["feature_layer"],
                      ev["evidence_role"], is_req))

        self.conn.commit()

    def verify_exposure(self, sample_id: str, case_id: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM retrieval_exposure_log WHERE sample_id = ? AND case_id = ?", (sample_id, case_id))
        return cursor.fetchone() is not None
    
    def load_atomic_memory(self, matched_uuid: str,
                           types: str) -> Optional[dict]:
        cursor = self.conn.cursor()
        if types == MODALITY_PREDICT:  #predict
            cursor.execute(
                """SELECT case_id, trade_time, symbol, predict_uuid, regime_uuid, textual_uuid,
                    implied_direction, decision_rule, invalidation_rules, validation_status
                    FROM case_memories  WHERE predict_uuid = ?""",
                (matched_uuid, ))
        elif types == MODALITY_REGIME:  #regmie
            cursor.execute(
                """SELECT case_id, trade_time, symbol, predict_uuid, regime_uuid, textual_uuid,
                    implied_direction, decision_rule, invalidation_rules, validation_status
                    FROM case_memories  WHERE regime_uuid = ?""",
                (matched_uuid, ))
        elif types == MODALITY_TEXT:  #textual
            cursor.execute(
                """SELECT case_id, trade_time, symbol, predict_uuid, regime_uuid, textual_uuid,
                    implied_direction, decision_rule, invalidation_rules, validation_status
                    FROM case_memories  WHERE textual_uuid = ?""",
                (matched_uuid, ))

        row = cursor.fetchone()
        if not row:
            return None

        (case_id, trade_time, symbol, p_uid, r_uid, t_uid, imp_dir,
         dec_rule_json, inv_rules_json, val_status) = row

        cursor.execute(
            """
        SELECT evidence_id, feature_layer, feature_refs, interpretation, evidence_role 
        FROM evidence_units 
        WHERE case_id = ?
        """, (case_id, ))
        ev_rows = cursor.fetchall()

        evidence_units = [{
            "evidence_id": r[0],
            "feature_layer": r[1],
            "feature_refs": json.loads(r[2]),
            "interpretation": r[3],
            "evidence_role": r[4]
        } for r in ev_rows]

        return {
            "case_id": case_id,
            "trade_time": trade_time,
            "symbol": symbol,
            "predict_uuid": p_uid,
            "regime_uuid": r_uid,
            "textual_uuid": t_uid,
            "implied_direction": imp_dir,
            "decision_rule": json.loads(dec_rule_json),
            "invalidation_rules": json.loads(inv_rules_json),
            "validation_status": val_status,
            "evidence_units": evidence_units
        }

    def update_memory_atomic(self,
                             sample_id: str,
                             case_id: str,
                             pred_dir: str,
                             mem_dir: str,
                             rel: str,
                             usage_weight: float,
                             conf: float,
                             realized_return: float,
                             delta_score: float,
                             is_win: bool,
                             min_accepted: int = 3,
                             min_win_rate: float = 0.55):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT accepted_count, win_count, loss_count, cumulative_score, validation_status, predict_uuid, regime_uuid, textual_uuid FROM case_memories WHERE case_id = ?",
                (case_id, ))
            row = cursor.fetchone()
            if not row:
                return False, None

            acc_cnt, win_cnt, loss_cnt, cum_score, status_before, p_uid, r_uid, t_uid = row

            quality_before = cum_score
            acc_cnt_after = acc_cnt + 1
            win_cnt_after = win_cnt + (1 if is_win else 0)
            loss_cnt_after = loss_cnt + (0 if is_win else 1)
            cum_score_after = cum_score + delta_score
            win_rate_after = round(win_cnt_after / acc_cnt_after, 4)
            mean_score_after = round(cum_score_after / acc_cnt_after, 6)

            if acc_cnt_after >= min_accepted and win_rate_after >= min_win_rate and cum_score_after > 0.0:
                status_after = "VALIDATED_ACTIVE"
            elif acc_cnt_after >= min_accepted and (win_rate_after < 0.40 or
                                                    cum_score_after < -0.01):
                status_after = "DEPRECATED"
            else:
                status_after = "UNVALIDATED"

            cursor.execute(
                """
            INSERT INTO settlement_audit_log (
                sample_id, case_id, predict_direction, memory_direction,
                relation_to_decision, usage_weight, confidence, realized_return,
                delta_score, status_before, status_after, quality_before, quality_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (sample_id, case_id, pred_dir, mem_dir, rel, usage_weight,
                  conf, realized_return, delta_score, status_before,
                  status_after, quality_before, cum_score_after))

            cursor.execute(
                """
            UPDATE case_memories
            SET accepted_count = ?, win_count = ?, loss_count = ?, win_rate = ?,
                cumulative_score = ?, mean_score = ?, validation_status = ?
            WHERE case_id = ?
            """, (acc_cnt_after, win_cnt_after, loss_cnt_after, win_rate_after,
                  cum_score_after, mean_score_after, status_after, case_id))

            self.conn.commit()
            updated_entity = self.load_memory_by_case_id(case_id)
            return True, updated_entity

        except sqlite3.IntegrityError:
            self.conn.rollback()
            print(f"⚠️ [幂等拦截] 样本 {sample_id} 对经验 {case_id} 已经结算过，拒绝重复累加！")
            return False, None
        except Exception as e:
            self.conn.rollback()
            print(f"❌ [事务异常回滚] {e}")
            return False, None


class MultimodalBrain:
    """
    多模态大脑 (统一封装 Predictive, Regime 与 Textual 三大向量通道)
    """

    def __init__(self, name, vector_provider, embedding_model,
                 embedding_provider, p_dim, r_dim, storage_path):
        self.name = name
        self.storage_path = storage_path

        self.predict_brain = self.create_brain(
            name='predict',
            vector_provider=vector_provider,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            emb_dim=p_dim)
        self.regime_brain = self.create_brain(
            name='regime',
            vector_provider=vector_provider,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            emb_dim=r_dim)
        self.textual_barain = self.create_brain(
            name='textual',
            vector_provider=vector_provider,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            emb_dim=-1)

    def create_best_path(self, name):
        return os.path.join(self.storage_path, "bestpoints", name)

    def create_brain(self, name, vector_provider, embedding_model,
                     embedding_provider, emb_dim):
        checkpoint_dir = self.create_best_path(name=name)
        filename = os.path.join(checkpoint_dir, "state_dict.pkl")
        if os.path.exists(filename):
            brain = load_brain(path=checkpoint_dir)
        else:
            if emb_dim != -1:
                brain = create_brain(vector_provider=vector_provider,
                                     db_name="{0}".format(name),
                                     emb_dim=emb_dim)
            else:
                brain = create_brain(vector_provider=vector_provider,
                                     db_name="{0}".format(name),
                                     embedding_model=embedding_model,
                                     embedding_provider=embedding_provider,
                                     emb_dim=-1)

        return brain

    def add_memory(self, code, date, predict_matrix, regime_matrix,
                   textual_list):
        predict_ids, predict_str = create_matrix_uuid(
            trade_time=date, code=code, feature_matrix=predict_matrix)

        regime_ids, regime_str = create_matrix_uuid(
            trade_time=date, code=code, feature_matrix=regime_matrix)

        predict_uuids = self.predict_brain.add_memory_short_term(
            symbol=code,  ## symbol 是关键索引
            date=date,
            content=predict_matrix,
            unique_ids=predict_ids)

        regime_uuids = self.regime_brain.add_memory_short_term(
            symbol=code,  ## symbol 是关键索引
            date=date,
            content=regime_matrix,
            unique_ids=regime_ids)

        textual_uuids = self.textual_barain.add_memory_short_term(
            symbol=code, date=date, content=textual_list)

        return predict_uuids, regime_uuids, textual_uuids, predict_str, regime_str

    def query_memory(self, code, predict_matrix, regime_matrix, textual_list,
                     top_k):

        predict_matrix1, _, predict_scores, predict_uuids = self.predict_brain.query_memory_short_term(
            query_content=predict_matrix, top_k=top_k, symbol=code)

        regime_matrix1, _, regime_scores, regime_uuids = self.regime_brain.query_memory_short_term(
            query_content=regime_matrix, top_k=top_k, symbol=code)

        textual_list1, _, textual_scores, textual_uuids = self.textual_barain.query_memory_short_term(
            query_content=textual_list, top_k=top_k, symbol=code)

        predict_retrieve = dict(zip(predict_uuids, predict_scores))
        regime_retrieve = dict(zip(regime_uuids, regime_scores))
        textual_retrieve = dict(zip(textual_uuids, textual_scores))

        return predict_retrieve, regime_retrieve, textual_retrieve

    def update_feedback(self, symbol, predict_uuids, regime_uuids,
                        textual_uuids, reward):
        pass

    def save(self, processed_count=None, force=True):
        predict_checkpoint_dir = self.create_best_path(name="predict")
        os.makedirs(predict_checkpoint_dir, exist_ok=True)
        self.predict_brain.save_checkpoint(path=predict_checkpoint_dir,
                                           force=force)

        regime_checkpoint_dir = self.create_best_path(name="regime")
        os.makedirs(regime_checkpoint_dir, exist_ok=True)
        self.regime_brain.save_checkpoint(path=regime_checkpoint_dir,
                                          force=force)

        textual_checkpoint_dir = self.create_best_path(name="textual")
        os.makedirs(textual_checkpoint_dir, exist_ok=True)
        self.textual_barain.save_checkpoint(path=textual_checkpoint_dir,
                                            force=force)


class MARLMemoryCoordinator:

    def __init__(self,
                 name,
                 storage_path,
                 vector_provider,
                 embedding_model,
                 embedding_provider,
                 p_dim,
                 r_dim,
                 min_accepted=3,
                 min_win_rate=0.55):
        self.min_accepted = min_accepted
        self.min_win_rate = min_win_rate
        self.sqlite_store = SQLiteMemoryStorage(db_path=storage_path,
                                                db_name=name)
        self.multimodal_brain = MultimodalBrain(
            name=name,
            storage_path=storage_path,
            vector_provider=vector_provider,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            p_dim=p_dim,
            r_dim=r_dim)

    def collect_modality_candidates(self, modality, retrieval_result,
                                    case_map):
        for rank, (matched_uuid,
                   raw_score) in enumerate(retrieval_result.items(), start=1):
            memory = self.sqlite_store.load_atomic_memory(
                matched_uuid=matched_uuid, types=modality)
            if memory is None:
                continue
            case_id = memory["case_id"]
            if case_id not in case_map:
                case_map[case_id] = {
                    "case_id": case_id,
                    "entity": memory,
                    "matches": {}
                }

            case_map[case_id]["matches"][modality] = {
                "status": "MATCHED",
                "vector_id": matched_uuid,
                "rank": rank,
                "score": float(raw_score),
            }

    def store_experience(self, code: str, trade_time: str, regime_matrix: list,
                         predict_matrix: list, textual_events: list,
                         review_dict: dict) -> str:
        predict_uuids, regime_uuids, textual_uuids, predict_str, regime_str = self.multimodal_brain.add_memory(
            code=code,
            date=trade_time,
            predict_matrix=predict_matrix,
            regime_matrix=regime_matrix,
            textual_list=textual_events)
        self.multimodal_brain.save()
        self.sqlite_store.save_review_result(case_id=review_dict["case_id"],
                                             symbol=code,
                                             trade_time=trade_time,
                                             predict_uuid=predict_uuids[0],
                                             regime_uuid=regime_uuids[0],
                                             textual_uuid=textual_uuids[0],
                                             regime_matrix_str=regime_str,
                                             predict_matrix_str=predict_str,
                                             textual_events=textual_events,
                                             review_dict=review_dict)

    def candidate_sort_key(self, candidate):
        """
        排序优先级：
        1. 命中模态越多越优先 (-matched_modality_count)
        2. 最佳通道排名越靠前越优先 (best_channel_rank)
        3. 命中通道平均相似度越高越优先 (-mean_matched_similarity)
        4. 案例时间越新越优先
        5. case_id 稳定键
        """
        entity = candidate["entity"]
        trade_time_str = str(entity.get("trade_time", "1970-01-01"))
        try:  # 将日期字符串解析为负数值 (实现越新越优先)
            parts = trade_time_str.replace(":", "-").replace(" ",
                                                             "-").split("-")
            time_score = int(parts[0]) * 10000 + int(parts[1]) * 100 + int(
                parts[2])
            neg_time = -time_score  # 2026-07-06 对应 -20260706，比 2022-01-01 的 -20220101 更小，排在前面！
        except Exception:
            neg_time = 0.0
        return (-candidate["matched_modality_count"],
                candidate["best_channel_rank"],
                -candidate["mean_matched_similarity"],
                str(entity.get("trade_time", "")), candidate["case_id"])

    def finalize_candidate_metrics(self, case_map) -> None:
        """
        为候选案例计算方案 A 的多模态命中指标 (只统计实际命中通道)
        """
        for candidate in case_map.values():
            matches = candidate["matches"]
            scores = [item["score"] for item in matches.values()]
            ranks = [item["rank"] for item in matches.values()]

            candidate["matched_modalities"] = sorted(matches.keys())
            candidate["matched_modality_count"] = len(matches)
            candidate["best_channel_rank"] = min(ranks)
            candidate["mean_matched_similarity"] = sum(scores) / len(scores)

    def build_projection_candidate_pool(self, case_map, quota_per_modality=5):
        """
        构建候选池：保留所有多模态共同命中 + 各单模态最多保留 quota_per_modality 条
        """
        all_candidates = sorted(case_map.values(), key=self.candidate_sort_key)
        selected = []
        selected_ids = set()

        # 保留所有多模态共同命中 (>=2)
        for candidate in all_candidates:
            if candidate["matched_modality_count"] < 2:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["case_id"])

        # 各单模态补充
        for modality in [MODALITY_REGIME, MODALITY_PREDICT, MODALITY_TEXT]:
            modality_count = 0
            for candidate in all_candidates:
                if modality_count >= quota_per_modality:
                    break
                if candidate["case_id"] in selected_ids or candidate[
                        "matched_modality_count"] != 1:
                    continue
                if modality not in candidate["matches"]:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate["case_id"])
                modality_count += 1

        return sorted(selected, key=self.candidate_sort_key)

    def select_final_projected_cases(self,
                                     projected_candidates,
                                     final_top_k=3):
        """
        最终配额选择：优先选多模态共同命中，不足时从单模态按顺序补位
        """
        selected = []
        selected_ids = set()

        # 1. 优先选多模态共振案例
        for item in projected_candidates:
            candidate = item["candidate"]
            if candidate["matched_modality_count"] < 2:
                continue
            selected.append(item)
            selected_ids.add(candidate["case_id"])
            if len(selected) >= final_top_k:
                return selected

        # 2. 单模态补位 (REGIME -> PREDICTIVE -> TEXT)
        for modality in [MODALITY_REGIME, MODALITY_PREDICT, MODALITY_TEXT]:
            for item in projected_candidates:
                candidate = item["candidate"]
                if candidate["case_id"] in selected_ids or candidate[
                        "matched_modality_count"] != 1:
                    continue
                if modality not in candidate["matches"]:
                    continue
                selected.append(item)
                selected_ids.add(candidate["case_id"])
                break
            if len(selected) >= final_top_k:
                break

        return selected

    def retrieve_experience(
            self,
            code: str,
            regime_matrix: list,
            predict_matrix: list,
            textual_events: list,
            active_predictive_whitelist: Set[str],
            channel_top_k: int = 10,  #底层向量库宽召回（每通道 10 条）
            quota_per_modality: int = 5,  # 投影候选池缓冲（单通道最多保留 5 条备胎）
            final_top_k: int = 2):  # 最终交付给大模型的精准条数（3 条）

        ## 三路独立召回
        predict_retrieve, regime_retrieve, textual_retrieve = self.multimodal_brain.query_memory(
            code=code,
            predict_matrix=predict_matrix,
            regime_matrix=regime_matrix,
            textual_list=textual_events,
            top_k=channel_top_k)

        # 2. 合并为统一 case_map
        case_map = {}
        self.collect_modality_candidates(MODALITY_PREDICT, predict_retrieve,
                                         case_map)
        self.collect_modality_candidates(MODALITY_REGIME, regime_retrieve,
                                         case_map)
        self.collect_modality_candidates(MODALITY_TEXT, textual_retrieve,
                                         case_map)
        if not case_map:
            return "【无匹配的历史案例记忆（零样本独立推演模式）】"

        # 3. 计算方案 A 的排序指标
        self.finalize_candidate_metrics(case_map=case_map)

        # 4. 构建充分的投影候选池 (防止前序案例熔断后候补不足)
        candidate_pool = self.build_projection_candidate_pool(
            case_map, quota_per_modality=quota_per_modality)

        # 5. 动态投影与熔断过滤
        projected_candidates = []
        for candidate in candidate_pool:
            memory = candidate["entity"]
            rendered_markdown = MemoryProjector.project_and_render_markdown(
                memory_dict=memory,
                active_predictive_whitelist=active_predictive_whitelist)
            if rendered_markdown is None:
                print(
                    f"⚠️ [方案 A 熔断] 案例 {candidate['case_id']} 核心证据依赖缺失，已自动从候选池熔断！"
                )
                continue

            projected_candidates.append({
                "candidate": candidate,
                "rendered_markdown": rendered_markdown
            })

        # 6. 最终配额选择 (优先多模态，不足单模态补足至 final_top_k)
        final_candidates = self.select_final_projected_cases(
            projected_candidates, final_top_k=final_top_k)

        if not final_candidates:
            return "【无匹配的历史案例记忆（零样本独立推演模式）】"

        # 7. 渲染纯净 Markdown (仅展示业务召回依据，不展示浮点相似度)
        reason_labels = {
            MODALITY_REGIME: "市场环境相似",
            MODALITY_PREDICT: "预测信号轨迹相似",
            MODALITY_TEXT: "事件背景相似",
        }
        rendered_blocks = []
        for item in final_candidates:
            candidate = item["candidate"]
            matched_modalities = candidate["matched_modalities"]
            reason_text = "、".join(
                [reason_labels[m] for m in matched_modalities])

            # 多模态共振时添加醒目标签，但不暴露具体分数
            tag = " [多模态共同确认]" if candidate[
                "matched_modality_count"] >= 2 else ""

            rendered_blocks.append(f"{item['rendered_markdown']}\n"
                                   f"*召回依据：{reason_text}{tag}*")
        return "\n\n".join(rendered_blocks)

    def update_memory(self, sample_id: str, symbol: str,
                      trader_prediction: dict, realized_return: float):
        pdb.set_trace()
        used_memories = trader_prediction.get("used_memories", [])
        pred_dir = trader_prediction.get("predict_direction", "FLAT")
        if not used_memories or realized_return is None or pred_dir == "FLAT":
            return []

        # 1. 语义强校验
        num_mem = len(used_memories)
        if num_mem > 1:
            primary_count = sum(1 for m in used_memories
                                if m.get("usage") == "primary")
            if primary_count != 1:
                raise ValueError(
                    f"used_memories 语义校验失败：多条经验时必须且只能指定 1 条 primary，当前包含 {primary_count} 条！"
                )

        # 2. 计算守恒归一化权重
        raw_weights = []
        has_primary = any(m.get("usage") == "primary" for m in used_memories)
        for m in used_memories:
            if num_mem == 1:
                w = 1.0
            elif m.get("usage") == "primary":
                w = 0.7
            else:
                num_aux = max(1, num_mem - (1 if has_primary else 0))
                w = 0.3 / num_aux
            raw_weights.append(w)

        total_w = sum(raw_weights)
        norm_weights = [w / total_w for w in raw_weights]

        scored_records = []
        pdb.set_trace()
        for mem, weight in zip(used_memories, norm_weights):
            mem_id = mem.get("memory_id")
            mem_dir = mem.get("implied_direction", "FLAT")
            rel = mem.get("relation_to_decision", "SUPPORT").upper()
            conf = float(mem.get("confidence", 0.5))
            pdb.set_trace()
            # 需要加上 每次检索加入日志
            # # 3. 真实曝光审计校验 (防幻觉)
            # if not self.sqlite_store.verify_exposure(sample_id, mem_id):
            #     print(
            #         f"⚠️ [幻觉拦截] 经验 {mem_id} 未在样本 {sample_id} 的真实检索曝光列表中，拒绝结算！")
            #     continue

            # 4. 经验自身方向胜负判定 (memory_direction_win)
            if rel == "BACKGROUND":
                delta_score = 0.0
                is_win = True
            else:
                memory_direction_win = (mem_dir == "UP" and realized_return
                                        > 0) or (mem_dir == "DOWN"
                                                 and realized_return < 0)
                is_win = memory_direction_win
                delta_sign = 1.0 if is_win else -1.0
                delta_score = weight * delta_sign * conf * abs(realized_return)

            # 5. 引擎 1: SQLite 事务原子性与完整审计更新
            success, updated_entity = self.sqlite_store.update_memory_atomic(
                sample_id=sample_id,
                case_id=mem_id,
                pred_dir=pred_dir,
                mem_dir=mem_dir,
                rel=rel,
                usage_weight=weight,
                conf=conf,
                realized_return=realized_return,
                delta_score=delta_score,
                is_win=is_win,
                min_accepted=self.min_accepted,
                min_win_rate=self.min_win_rate)
            
            if success and updated_entity:
                # 6. 引擎 2: MultimodalBrain 向量通道同步更新质量分与计数器
                p_uid = updated_entity.get("predict_uuid")
                r_uid = updated_entity.get("regime_uuid")
                t_uid = updated_entity.get("textual_uuid")

                self.multimodal_brain.update_feedback(
                    symbol=symbol,
                    predict_uuids=[p_uid] if p_uid else [],
                    regime_uuids=[r_uid] if r_uid else [],
                    textual_uuids=[t_uid] if t_uid else [],
                    reward=delta_score)

                stats = updated_entity["stats"]
                scored_records.append({
                    "case_id":
                    mem_id,
                    "relation":
                    rel,
                    "mem_dir":
                    mem_dir,
                    "weight":
                    round(weight, 4),
                    "memory_direction_win":
                    is_win,
                    "delta_score":
                    round(delta_score, 6),
                    "total_score":
                    round(stats["cumulative_score"], 6),
                    "acceptance_rate":
                    stats["acceptance_rate"],
                    "new_status":
                    updated_entity["validation_status"]
                })

        return scored_records
