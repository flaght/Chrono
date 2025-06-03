import sys, os, pdb
from datetime import date
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath('..'))
load_dotenv()

from dichaos.services.vector_db import VectorServiceFactory
from dichaos.services.vector_db.functions import *

## short mermory layer
verctor_db = VectorServiceFactory.create_vector_service(
    vector_provider='fassis',
    db_name='test_db',
    model_name='bge-large',
    model_provider='openai',
    jump_threshold_lower=6,
    jump_threshold_upper=10,
    importance_score_initialization=get_importance_score_initialization_func(
        type='sample', memory_layer="short"),
    recency_score_initialization=R_ConstantInitialization(),
    compound_score_calculation=LinearCompoundScore(),
    importance_score_change_access_counter=LinearImportanceScoreChange(),
    decay_function=ExponentialDecay(recency_factor=3.0,
                                    importance_factor=0.92))


pdb.set_trace()
temp_record = verctor_db.create_index()
verctor_db.add_memory(temp_record=temp_record,
                      date='2024-05-01',
                      text="This is a test")
verctor_db.add_memory(temp_record=temp_record,
                        date='2024-05-02',
                        text="This is another test")
verctor_db.add_memory(temp_record=temp_record,
                        date='2024-05-03',
                        text="This is a test2")
verctor_db.add_memory(temp_record=temp_record,
                        date='2024-05-04',
                        text="This is another test2")

results = verctor_db.query_memory(temp_record=temp_record,
                                  query_text="date is 2024-05-01",
                                  top_k=5)
pdb.set_trace()

verctor_db.step()

verctor_db.prepare_jump()
results = verctor_db.query_memory(temp_record=temp_record,  query_text="date is 2024-05-01", top_k=5)
pdb.set_trace()
print(results)
