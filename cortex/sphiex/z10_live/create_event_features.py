import pdb
import pandas as pd
from features.text.fetch_data import FetchData
from ultron.tradingday import advanceDateByCalendar


def create_features(begin_date, end_date, premarket=True):
    fetcher = FetchData()
    end_date1 = end_date if not premarket else advanceDateByCalendar(
        'china.sse', end_date, '1b').strftime('%Y-%m-%d')
    news_data = fetcher.fetch_news(begin_date, end_date1)
    cctv_data = fetcher.fetch_cctv(begin_date, end_date1)
    monetary_data = fetcher.fetch_monetary_policy(begin_date, end_date1)
    gov_data = fetcher.fetch_gov_policy(begin_date, end_date1)
    total_data = pd.concat([news_data, cctv_data, monetary_data, gov_data],
                           axis=0)
    total_data = total_data.sort_values(by=['date']).reset_index(drop=True)
    return total_data[total_data['date'] <= end_date1]
