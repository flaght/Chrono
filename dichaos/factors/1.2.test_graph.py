import sys, os, pdb, json
from typing import List
from datetime import date, timedelta
from dotenv import load_dotenv
from macro import *

sys.path.insert(0, os.path.abspath('../../'))
load_dotenv()

from dichaos.agents.basic.factorence.graph import Graph

graph = Graph()
graph.run(basic_factors=basic_factors,
          example_factors=example_factors,
          expression=expression,
          basic_data=basic_data)
