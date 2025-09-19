############
import pdb, os, argparse, itertools
import pandas as pd
import numpy as np
from pymongo import InsertOne, DeleteOne
from dotenv import load_dotenv
load_dotenv()

from kdutils.mongodb import MongoDBManager
from lib.aux001 import *


codes = ['ims','ics']
method = 'aicso0'

both_compare(codes=codes, expression="MCPS(14,'ixy013_5_10_1')",method=method)