import os, pdb, asyncio, time
import pandas as pd
from urllib.parse import urlparse
from kdutils.report import ReportGenerator, DetailGenerator
from kdutils.mongo import MongoLoader
from alphacopilot.calendars.api import advanceDateByCalendar


class Report(object):

    def __init__(self, symbol, base_path, resource_path):
        self._base_path = base_path
        self._resource_path = resource_path
        self._symbol = symbol
        self._mongo_client = MongoLoader(
            connection_string=os.environ['MG_URL'],
            db_name=urlparse(os.environ['MG_URL']).path.lstrip('/'),
            collection_name='chat_history1')

    def load_data(self, end_date, table_name):
        results = self._mongo_client.find(query={
            'trade_date': end_date,
            'code': self._symbol
        },
                                          collection_name=table_name)
        return results

    def process(self, end_date):
        end_date1 = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')
        final_prediction = self.load_data(end_date=end_date1,
                                          table_name='alpha_agent_final')
        final_prediction = final_prediction.drop(['_id'],
                                                 axis=1).loc[0].to_dict()
        final_battle_state = self.load_data(end_date=end_date1,
                                            table_name='alpha_agent_debat')
        final_battle_state = final_battle_state.drop(['_id'],
                                                     axis=1).loc[0].to_dict()
        reason_reports = self.load_data(end_date=end_date1,
                                        table_name='alpha_agent_reason')
        reason_reports = reason_reports[[
            'name', 'reasoning'
        ]].set_index('name')['reasoning'].to_dict()
        report_data = {
            "date": end_date,
            "symbol": self._symbol,
            "final_prediction": final_prediction,
            "battle_results": final_battle_state,
            "reason_reports": reason_reports
        }
        return report_data

    def run(self, end_date):
        report = ReportGenerator(
            output_dir=os.path.join(self._base_path, str(end_date), "report",
                                    "html"),
            template_name=os.path.join(self._resource_path,
                                       "report_template.html"))
        report_data = self.process(end_date=end_date)
        report.run(report_data=report_data)

    def process1(self, end_date, codes=None):
        base_url = "http://120.136.158.26:32738/opts/dichaos/stock/{0}/report/html/DiChaos_Report_{1}.html"
        end_date1 = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')
        query = {}
        query['trade_date'] = end_date1
        if isinstance(codes, list):
            query['code'] = {'$in': codes}

        results = self._mongo_client.find(query=query,
                                          collection_name='alpha_agent_final')
        res = []
        for report in results.itertuples():
            res.append({
                'symbol': report.code,
                'trade_date': end_date,
                'signal': report.signal,
                'confidence': report.confidence,
                "address": base_url.format(end_date, report.code)
            })
        result_data = pd.DataFrame(res).sort_values(
            by=['signal', 'confidence'], ascending=[True, False])
        return result_data

    def detail(self, end_date, codes=None):
        result_data = self.process1(end_date=end_date, codes=codes)
        report = DetailGenerator(
            output_dir=os.path.join(self._base_path, str(end_date), "report",
                                    "html"),
            template_name=os.path.join(self._resource_path,
                                       "detail_template.html"))

        report.run(detail_data=result_data)
        



    def output(self, end_date, codes):
        end_date1 = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')
        results = self._mongo_client.find(query={
            'trade_date': end_date1,
            'code': {
                '$in': codes
            }
        },
                                          collection_name='alpha_agent_final')
        base_url = "http://120.136.158.26:32738/opts/dichaos/stock/{0}/report/html/DiChaos_Report_{1}.html"
        res = []
        for report in results.itertuples():
            res.append({
                'symbol': report.code,
                'trade_date': end_date,
                'signal': report.signal,
                'confidence': report.confidence,
                "address": base_url.format(end_date, report.code)
            })

        result_data = pd.DataFrame(res).sort_values(
            by=['signal', 'confidence'], ascending=[True, False])
