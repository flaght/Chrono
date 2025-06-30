import sys, os
from datetime import date
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath('..'))
load_dotenv()

from dichaos.brain import BrainDB
from dichaos.services.vector_db import VectorServiceFactory
from dichaos.services.vector_db.functions import *

vector_provider = 'fassis'
db_name = 'test_db'
model_name = 'bge-large'
model_provider = 'openai'

short_memory = VectorServiceFactory.create_vector_service(
    vector_provider=vector_provider,
    db_name=db_name,
    model_name=model_name,
    model_provider=model_provider,
    jump_threshold_lower=0.1,
    jump_threshold_upper=0.5,
    importance_score_initialization=get_importance_score_initialization_func(
        type='sample', memory_layer="short"),
    recency_score_initialization=R_ConstantInitialization(),
    compound_score_calculation=LinearCompoundScore(),
    importance_score_change_access_counter=LinearImportanceScoreChange(),
    decay_function=ExponentialDecay(recency_factor=3.0,
                                    importance_factor=0.92))

long_memory = VectorServiceFactory.create_vector_service(
    vector_provider=vector_provider,
    db_name=db_name,
    model_name=model_name,
    model_provider=model_provider,
    jump_threshold_lower=1.7,
    jump_threshold_upper=1.9,
    importance_score_initialization=get_importance_score_initialization_func(
        type='sample', memory_layer="long"),
    recency_score_initialization=R_ConstantInitialization(),
    compound_score_calculation=LinearCompoundScore(),
    importance_score_change_access_counter=LinearImportanceScoreChange(),
    decay_function=ExponentialDecay(recency_factor=365.0,
                                    importance_factor=0.967))

mid_memory = VectorServiceFactory.create_vector_service(
    vector_provider=vector_provider,
    db_name=db_name,
    model_name=model_name,
    model_provider=model_provider,
    jump_threshold_lower=1.3,
    jump_threshold_upper=1.5,
    importance_score_initialization=get_importance_score_initialization_func(
        type='sample', memory_layer="mid"),
    recency_score_initialization=R_ConstantInitialization(),
    compound_score_calculation=LinearCompoundScore(),
    importance_score_change_access_counter=LinearImportanceScoreChange(),
    decay_function=ExponentialDecay(recency_factor=90.0,
                                    importance_factor=0.967))

reflection_memory = VectorServiceFactory.create_vector_service(
    vector_provider=vector_provider,
    db_name=db_name,
    model_name=model_name,
    model_provider=model_provider,
    jump_threshold_lower=13,
    jump_threshold_upper=15,
    importance_score_initialization=get_importance_score_initialization_func(
        type='sample', memory_layer="reflection"),
    recency_score_initialization=R_ConstantInitialization(),
    compound_score_calculation=LinearCompoundScore(),
    importance_score_change_access_counter=LinearImportanceScoreChange(),
    decay_function=ExponentialDecay(recency_factor=365.0,
                                    importance_factor=0.988))

brain_db = BrainDB(
    agent_name='test_agent',
    short_term_memory=short_memory,
    mid_term_memory=mid_memory,
    long_term_memory=long_memory,
    reflection_memory=reflection_memory,
    use_gpu=False)

brain_db.add_memory_reflection(symbol='test_symbol', date=date.today(),
                                 text='This is a test reflection')

brain_db.add_memory_long_term(symbol='test_symbol', date=date.today(),
                                text='This is a test long term memory')

brain_db.add_memory_mid_term(symbol='test_symbol', date=date.today(),
                                text='This is a test mid term memory')

brain_db.add_memory_short_term(symbol='test_symbol', date=date.today(),
                                text='This is a test short term memory')

print(brain_db.query_memory_reflection(query_text='test', top_k=5,
                                        symbol='test_symbol'))