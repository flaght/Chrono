import sys, os, pdb, json
from typing import List
from datetime import date, timedelta
from dotenv import load_dotenv
from macro import *

sys.path.insert(0, os.path.abspath('../../'))
load_dotenv()
#load_dotenv(os.path.join('../', '.env'))

from dichaos.agents.basic.factorence.agent import Agent

agent = Agent.from_config()

agent.generate_outputs(basic_factors=basic_factors,
                       example_factors=example_factors,
                       expression=expression,
                       history_factors={})
