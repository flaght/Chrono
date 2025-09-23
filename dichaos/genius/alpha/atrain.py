import os, pdb, asyncio, time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
from agent import IndicatorTrain, MoneyFlowTrain, CloutoTrain, HotMoneyTrain, ChipTrain
from factors.calculator import create_return, create_chg
from factors.extractor import fetch_limitup


def agent_train1():
    #begin_date = '2025-01-02'
    end_date = '2020-01-23'
    returns = 0.05
    end_date = advanceDateByCalendar('china.sse', end_date,
                                     '-{0}b'.format(1)).strftime('%Y-%m-%d')
    symbol = '300805'
    pdb.set_trace()
    returns = create_return(codes=[symbol],
                            begin_date=end_date,
                            end_date=end_date)
    train = CloutoTrain(config_path=os.path.join("agent"),
                        memory_path=os.path.join("records"),
                        symbol=symbol)

    train.prepare_data(begin_date=end_date, end_date=end_date)

    train_data = train.create_data(date=end_date)

    response = asyncio.run(
        train.agenerate_suggestion(date=end_date,
                                   train_data=train_data,
                                   returns=returns[symbol][end_date]))
    train.update_memory(date=end_date,
                        returns=returns[symbol][end_date],
                        response=response)


async def agent_train2(symbol):

    trainer_sets = [
        CloutoTrain(config_path=os.path.join("agent"),
                    memory_path=os.path.join("records"),
                    symbol=symbol),
        IndicatorTrain(config_path=os.path.join("agent"),
                       memory_path=os.path.join("records"),
                       symbol=symbol),
        MoneyFlowTrain(config_path=os.path.join("agent"),
                       memory_path=os.path.join("records"),
                       symbol=symbol),
        HotMoneyTrain(config_path=os.path.join("agent"),
                      memory_path=os.path.join("records"),
                      symbol=symbol),
        ChipTrain(config_path=os.path.join("agent"),
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


async def train_with_semaphore(trainer, semaphore: asyncio.Semaphore,
                               date: str, returns: float):
    async with semaphore:
        print(f"[{trainer.agent.name}] 获取到信号量许可，开始执行训练...")
        try:
            # 执行实际的预测调用
            result = await trainer.agenerate_suggestion(
                date=date,
                train_data=await asyncio.to_thread(trainer.create_data,
                                                   date=date),
                returns=returns)
            return result
        except Exception as e:
            # 捕获并返回异常，这样 gather 就不会中断
            print(f"[{trainer.agent.name}] 预测时发生错误: {e}")
            return e


async def train3(trainer_sets, end_date, code):
    MAX_CONCURRENT_REQUESTS = 2
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    end_date = advanceDateByCalendar('china.sse', end_date,
                                     '-{0}b'.format(0)).strftime('%Y-%m-%d')
    returns = create_chg(codes=[code], begin_date=end_date, end_date=end_date)
    returns = returns[code][end_date]
    if np.isnan(returns):
        return
    for trainer in trainer_sets:
        trainer.symbol = code
        trainer.prepare_data(begin_date=end_date, end_date=end_date)

    tasks = [
        train_with_semaphore(p, semaphore, end_date, returns)
        for p in trainer_sets
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    _ = [
        trainer.update_memory(date=end_date,
                              returns=returns,
                              response=response) for response in results
        for trainer in trainer_sets if response.name == trainer.name
    ]


async def agent_train3(begin_date, end_date):
    symbol = 'all'
    trainer_sets = [
        IndicatorTrain(config_path=os.path.join("agent"),
                       memory_path=os.path.join("records"),
                       symbol=symbol),
        CloutoTrain(config_path=os.path.join("agent"),
                    memory_path=os.path.join("records"),
                    symbol=symbol),
        HotMoneyTrain(config_path=os.path.join("agent"),
                      memory_path=os.path.join("records"),
                      symbol=symbol),
        MoneyFlowTrain(config_path=os.path.join("agent"),
                       memory_path=os.path.join("records"),
                       symbol=symbol),
        ChipTrain(config_path=os.path.join("agent"),
                  memory_path=os.path.join("records"),
                  symbol=symbol),
    ]
    limitup_data = fetch_limitup(begin_date=begin_date, end_date=end_date)
    for date, codes in limitup_data.items():
        #prev_date = advanceDateByCalendar(
        #    'china.sse', date, '-{0}b'.format(2)).strftime('%Y-%m-%d')
        prev_date = date
        for code in codes:
            await train3(trainer_sets, prev_date, code)


if __name__ == "__main__":
    #agent_train1()
    #asyncio.run(agent_train2(symbol='601519'))
    asyncio.run(agent_train3(begin_date='2016-07-14', end_date='2017-12-31'))
