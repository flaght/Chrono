import pdb, os
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from dichaos.brain.hybrid_brain import HybridBrain
from dichaos.services.vector_db import VectorServiceFactory
from dichaos.services.vector_db.functions import *
from lib.pam001 import load_memory_params


def from_brain(vector_provider, db_name, embedding_model, embedding_provider,
               **params):
    """
    装配并实例化 VectorService （即底层各个时段各自的 FAISS/Chroma 数据库引擎）。
    分别将关于衰减系数（decay）、质量分数初始函数（type）和上下跳跃的界限阈值全部透传至向量底层。
    """
    return VectorServiceFactory.create_vector_service(
        vector_provider=vector_provider,
        db_name=db_name,
        model_name=embedding_model,
        model_provider=embedding_provider,
        importance_score_initialization=
        get_importance_score_initialization_func(
            type=params['type'], memory_layer=params['memory_layer']),
        recency_score_initialization=R_ConstantInitialization(),
        compound_score_calculation=LinearCompoundScore(),
        importance_score_change_access_counter=LinearImportanceScoreChange(),
        decay_function=ExponentialDecay(
            recency_factor=params['recency'],
            importance_factor=params['importance']),
        jump_threshold_upper=params['jump_threshold_upper'],
        jump_threshold_lower=params['jump_threshold_lower'],
        clean_up_threshold_dict=dict(params['clean_up_threshold_dict']),
        emb_dim=params['emb_dim'])


def create_brain(vector_provider,
                 db_name,
                 embedding_model='',
                 embedding_provider='',
                 emb_dim=-1):
    memory_params = load_memory_params()
    short_term_memory = from_brain(
        vector_provider=vector_provider,
        db_name=db_name,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        type=memory_params['short']['importance_score_initialization'],
        memory_layer="short",
        recency=memory_params['short']['decay_recency_factor'],
        importance=memory_params['short']['decay_importance_factor'],
        clean_up_threshold_dict=memory_params['short']
        ['clean_up_threshold_dict'],
        jump_threshold_lower=memory_params['short']['jump_threshold_lower'],
        jump_threshold_upper=memory_params['short']['jump_threshold_upper'],
        emb_dim=emb_dim)

    mid_term_memory = from_brain(
        vector_provider=vector_provider,
        db_name=db_name,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        type=memory_params['mid']['importance_score_initialization'],
        memory_layer="mid",
        recency=memory_params['mid']['decay_recency_factor'],
        importance=memory_params['mid']['decay_importance_factor'],
        clean_up_threshold_dict=memory_params['mid']
        ['clean_up_threshold_dict'],
        jump_threshold_lower=memory_params['mid']['jump_threshold_lower'],
        jump_threshold_upper=memory_params['mid']['jump_threshold_upper'],
        emb_dim=emb_dim)

    long_term_memory = from_brain(
        vector_provider=vector_provider,
        db_name=db_name,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        type=memory_params['long']['importance_score_initialization'],
        memory_layer="long",
        recency=memory_params['long']['decay_recency_factor'],
        importance=memory_params['long']['decay_importance_factor'],
        clean_up_threshold_dict=memory_params['long']
        ['clean_up_threshold_dict'],
        jump_threshold_lower=memory_params['long']['jump_threshold_lower'],
        jump_threshold_upper=memory_params['long']['jump_threshold_upper'],
        emb_dim=emb_dim)

    reflection_memory = from_brain(
        vector_provider=vector_provider,
        db_name=db_name,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        type=memory_params['reflection']['importance_score_initialization'],
        memory_layer="reflection",
        recency=memory_params['reflection']['decay_recency_factor'],
        importance=memory_params['reflection']['decay_importance_factor'],
        clean_up_threshold_dict=memory_params['reflection']
        ['clean_up_threshold_dict'],
        jump_threshold_lower=memory_params['reflection']
        ['jump_threshold_lower'],
        jump_threshold_upper=memory_params['reflection']
        ['jump_threshold_upper'],
        emb_dim=emb_dim)

    # 2. 将装配好的四维度数据库组装到 HybridBrain 实例中
    brain = HybridBrain(agent_name="QuantAgent_X",
                        short_term_memory=short_term_memory,
                        mid_term_memory=mid_term_memory,
                        long_term_memory=long_term_memory,
                        reflection_memory=reflection_memory)
    return brain


def load_brain(path):
    return HybridBrain.load_checkpoint(path=path)


def save_checkpoints(brain, data_path, processed_count):
    new_checkpoint_dir1 = os.path.join(data_path, "checkpoints",
                                       str(processed_count), "hybrid_brain")
    new_checkpoint_dir2 = os.path.join(data_path, "bestpoints", "hybrid_brain")
    os.makedirs(new_checkpoint_dir1, exist_ok=True)
    os.makedirs(new_checkpoint_dir2, exist_ok=True)
    brain.save_checkpoint(path=new_checkpoint_dir1, force=True)
    brain.save_checkpoint(path=new_checkpoint_dir2, force=True)


def filter_unvalidated_memory(brain, symbol: str, min_access: int = 1):
    """
    生产启动时调用：从 FAISS 索引和 score_memory 中
    物理删除所有从未经过 RL 验证的记忆（access_counter < min_access）
    """
    result = {}
    for memory_db in [
        brain.short_term_memory,
        brain.mid_term_memory,
        brain.long_term_memory,
        brain.reflection_memory,
    ]:
        if memory_db is None or symbol not in memory_db.universe:
            continue

        temp_record = memory_db.universe[symbol]
        original_count = len(temp_record["score_memory"])
        to_remove_ids = set()
        kept = SortedList(
            key=lambda x: x["important_score_recency_compound_score"])

        for record in temp_record["score_memory"]:
            if record.get("access_counter", 0) < min_access:
                to_remove_ids.add(record["id"])
            else:
                kept.add(record)

        if to_remove_ids:
            temp_record["score_memory"] = kept
            try:
                temp_record["index"].remove_ids(
                    np.array(list(to_remove_ids), dtype=np.int64))
            except Exception as e:
                print(f"[filter] FAISS remove error: {e}")
            for rid in to_remove_ids:
                temp_record["id_to_record"].pop(rid, None)

        kept_count = len(kept)
        removed = len(to_remove_ids)
        result[memory_db.db_name] = {
            "original": original_count,
            "kept": kept_count,
            "removed": removed
        }
    return result