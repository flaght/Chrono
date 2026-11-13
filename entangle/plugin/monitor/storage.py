# -*- coding: utf-8 -*-
"""监控表存储适配。"""


class Storage(object):

    def __init__(self, adapter=None):
        self.adapter = adapter

    def save(self, dataframe, db_name, table_name, method="increment"):
        if self.adapter is None:
            raise RuntimeError("未配置 DolphinDBAdapter")
        if dataframe.empty:
            return False
        self.adapter.refresh_data(
            db_name=db_name,
            table_name=table_name,
            total_data=dataframe,
            method=method,
        )
        return True

