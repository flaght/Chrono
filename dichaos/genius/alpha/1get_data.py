import pdb
import os, pdb, asyncio, time
from dotenv import load_dotenv

load_dotenv()

from factors.calculator import create_indictor

begin_date = '2025-03-01'
end_date = '2025-03-10'
code = '601519'

create_indictor(begin_date=begin_date,
             end_date=end_date,
             codes=['601519'],
             window=10)
