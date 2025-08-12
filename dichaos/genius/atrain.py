import os, pdb, asyncio, time
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
from agent import CloutoTrain, IndicatorTrain, PosFlowTrain, MoneyFlowTrain


def agent_train1():
    #begin_date = '2025-01-02'
    pdb.set_trace()
    end_date = '2025-02-01'
    returns = 0.05
    end_date = advanceDateByCalendar('china.sse', end_date,
                                     '-{0}b'.format(1)).strftime('%Y-%m-%d')
    symbol = 'IM'

    train = MoneyFlowTrain(config_path=os.path.join("agent"),
                           memory_path=os.path.join("records"),
                           symbol=symbol)

    train.prepare_data(begin_date=end_date, end_date=end_date)

    train_data = train.create_data(date=end_date)

    response = asyncio.run(
        train.agenerate_suggestion(date=end_date,
                                   train_data=train_data,
                                   returns=returns))
    train.update_memory(date=end_date, returns=returns, response=response)


async def agent_train2(symbol):

    trainer_sets = [
        PosFlowTrain(config_path=os.path.join("agent"),
                     memory_path=os.path.join("records"),
                     symbol=symbol),
        CloutoTrain(config_path=os.path.join("agent"),
                    memory_path=os.path.join("records"),
                    symbol=symbol),
        IndicatorTrain(config_path=os.path.join("agent"),
                       memory_path=os.path.join("records"),
                       symbol=symbol),
        MoneyFlowTrain(config_path=os.path.join("agent"),
                       memory_path=os.path.join("records"),
                       symbol=symbol)
    ]
    end_date = '2025-02-01'
    returns = 0.05
    end_date = advanceDateByCalendar('china.sse', end_date,
                                     '-{0}b'.format(1)).strftime('%Y-%m-%d')

    ### 串行数据准备
    for trainer in trainer_sets:
        trainer.prepare_data(begin_date=end_date, end_date=end_date)

    ### 并发
    tasks = [
        p.agenerate_suggestion(date=end_date,
                               train_data=p.create_data(date=end_date),
                               returns=returns) for p in trainer_sets
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    _ = [
        trainer.update_memory(date=end_date,
                              returns=returns,
                              response=response) for response in results
        for trainer in trainer_sets if response.name == trainer.name
    ]


if __name__ == "__main__":
    #agent_train1()
    asyncio.run(agent_train2(symbol='IM'))
