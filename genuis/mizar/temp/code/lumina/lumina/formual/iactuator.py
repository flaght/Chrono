import pdb, itertools
import pandas as pd
import lumina.env as env

env.g_format = 2

import lumina.impulse.i001 as i001
import lumina.impulse.i002 as i002
import lumina.impulse.i003 as i003
import lumina.impulse.i004 as i004
import lumina.impulse.i005 as i005
import lumina.impulse.i006 as i006
import lumina.impulse.i007 as i007
import lumina.impulse.i008 as i008
import lumina.impulse.i009 as i009
import lumina.impulse.i010 as i010
import lumina.impulse.i011 as i011
import lumina.impulse.i012 as i012
import lumina.impulse.i013 as i013
import lumina.impulse.i014 as i014

from lumina.formual.process import *


def create_factors(column, total_data):
    r1 = column.calc_impulse(total_data.copy())
    values = list(r1.values())
    values1 = [v.sort_index() for v in values]
    dt = pd.concat(values1, axis=1).sort_index()
    return dt


@add_process_env_sig
def run_factors(target_column, total_data):
    position_data = run_process(target_column=target_column,
                                callback=create_factors,
                                total_data=total_data)
    return position_data


class Iactuator(object):

    def __init__(self, k_split):
        self.k_split = k_split
        self.impulse = [
            i001, i002, i003, i004, i005, i006, i007, i008, i009, i010, i011,
            i012, i013, i014
        ]
        self.init_factors()

    def init_factors(self):
        self.factors = [
            getattr(i00, i0)() for i00 in self.impulse for i0 in i00.__all__
        ]

    def calculate(self, total_data):
        process_list = split_k(self.k_split, self.factors)

        res = create_parellel(process_list=process_list,
                              callback=run_factors,
                              total_data=total_data)
        res = list(itertools.chain.from_iterable(res))
        factors_data = pd.concat(res, axis=1).sort_index()
        return factors_data
