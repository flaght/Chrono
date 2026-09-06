import pdb
from lib.amr001 import SQLiteMemoryStorage, MultimodalBrain

class ExperienceEvaluator:

    def __init__(self, sqldb_path, min_accepted: int = 3, min_win_rate: float = 0.55):
        self.min_accepted = min_accepted
        self.min_win_rate = min_win_rate
        self.sqlite_store = SQLiteMemoryStorage(db_path=sqldb_path)
        self.multimodal_brain = MultimodalBrain(
            name=name,
            storage_path=storage_path,
            vector_provider=vector_provider,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            p_dim=p_dim,
            r_dim=r_dim)

    def update_memory(self, sample_id: str, symbol: str,
                      trader_prediction: dict, realized_return: float):

        pdb.set_trace()
        used_memories = trader_prediction.get("used_memories", [])
        pred_dir = trader_prediction.get("predict_direction", "FLAT")
        
        if not used_memories or realized_return is None or pred_dir == "FLAT":
            return []
        
        num_mem = len(used_memories)
        if num_mem > 1:
            primary_count = sum(1 for m in used_memories if m.get("usage") == "primary")
            if primary_count != 1:
                raise ValueError(f"used_memories 语义校验失败：多条经验时必须且只能指定 1 条 primary，当前包含 {primary_count} 条！")


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
        for mem, weight in zip(used_memories, norm_weights):
            mem_id = mem.get("memory_id")
            mem_dir = mem.get("implied_direction", "FLAT")
            rel = mem.get("relation_to_decision", "SUPPORT").upper()
            conf = float(mem.get("confidence", 0.5))

            # 3. 真实曝光审计校验 (防幻觉)
            if not self.sqlite_store.verify_exposure(sample_id, mem_id):
                print(f"⚠️ [幻觉拦截] 经验 {mem_id} 未在样本 {sample_id} 的真实检索曝光列表中，拒绝结算！")
                continue
