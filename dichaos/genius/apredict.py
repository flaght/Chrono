import os, pdb, asyncio, time
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
from agent import IndicatorPredict, CloutoPredict, PosFlowPredict


def agent_predict1():
    end_date = '2025-06-27'
    begin_date = advanceDateByCalendar('china.sse', end_date,
                                       '-{0}b'.format(10)).strftime('%Y-%m-%d')
    symbol = 'IM'

    predict = IndicatorPredict(symbol=symbol,
                               memory_path=os.path.join("records"),
                               date='2025-01-27')

    predict.prepare_data(begin_date=end_date, end_date=end_date)
    predict_data = predict.create_data(date=end_date)

    response = asyncio.run(
        predict.agenerate_prediction(date=end_date, predict_data=predict_data))


async def agent_predict2(symbol):
    model_date = '2025-01-27'
    end_date = '2025-06-27'
    predict_sets = [
        PosFlowPredict(date=model_date,
                       memory_path=os.path.join("records"),
                       symbol=symbol),
        CloutoPredict(date=model_date,
                      memory_path=os.path.join("records"),
                      symbol=symbol),
        IndicatorPredict(date=model_date,
                         memory_path=os.path.join("records"),
                         symbol=symbol),
    ]


    end_date = advanceDateByCalendar('china.sse', end_date,
                                     '-{0}b'.format(1)).strftime('%Y-%m-%d')

    ### 串行数据准备
    for p in predict_sets:
        p.prepare_data(begin_date=end_date, end_date=end_date)

    ### 并发
    tasks = [
        p.agenerate_prediction(date=end_date,
                               predict_data=p.create_data(date=end_date))
        for p in predict_sets
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        print(result)


if __name__ == "__main__":
    #agent_predict1()
    asyncio.run(agent_predict2(symbol='IM'))
