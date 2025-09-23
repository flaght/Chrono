import os, pdb, asyncio, time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar, makeSchedule
from agent import CloutoTrain, IndicatorTrain, PosFlowTrain, MoneyFlowTrain, ChipTrain
from factors.calculator import create_chg


def agent_train1():
    #begin_date = '2025-01-02'
    pdb.set_trace()
    end_date = '2025-02-01'
    returns = 0.05
    end_date = advanceDateByCalendar('china.sse', end_date,
                                     '-{0}b'.format(1)).strftime('%Y-%m-%d')
    symbol = 'IM'

    train = ChipTrain(config_path=os.path.join("agent"),
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


async def train3(trainer_sets, end_date, symbol):
    MAX_CONCURRENT_REQUESTS = 2
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    ## 将收益率向前偏移一天，与特征对齐
    returns = create_chg(codes=[symbol],
                         begin_date=end_date,
                         end_date=end_date,
                         offset=-1)
    returns = returns[symbol][end_date]

    if np.isnan(returns):
        return
    ## 向前推一天 T期特征对应T+1期收益率
    for trainer in trainer_sets:
        trainer.prepare_data(begin_date=end_date, end_date=end_date)

    ### 并发
    tasks = [
        train_with_semaphore(p, semaphore, end_date, returns)
        for p in trainer_sets
    ]
    '''
    tasks = [
        p.agenerate_suggestion(date=end_date,
                               train_data=p.create_data(date=end_date),
                               returns=returns) for p in trainer_sets
    ]
    '''
    results = await asyncio.gather(*tasks, return_exceptions=True)
    _ = [
        trainer.update_memory(date=end_date,
                              returns=returns,
                              response=response) for response in results
        for trainer in trainer_sets if response.name == trainer.name
    ]


async def agent_train3(begin_date, end_date, symbol):
    trainer_sets = [
        IndicatorTrain(config_path=os.path.join("agent"),
                       memory_path=os.path.join("records"),
                       symbol=symbol),
        CloutoTrain(config_path=os.path.join("agent"),
                    memory_path=os.path.join("records"),
                    symbol=symbol),
        PosFlowTrain(config_path=os.path.join("agent"),
                     memory_path=os.path.join("records"),
                     symbol=symbol),
        MoneyFlowTrain(config_path=os.path.join("agent"),
                       memory_path=os.path.join("records"),
                       symbol=symbol),
        ChipTrain(config_path=os.path.join("agent"),
                  memory_path=os.path.join("records"),
                  symbol=symbol),
    ]

    dates = makeSchedule(begin_date,
                         endDate=end_date,
                         tenor='1b',
                         calendar='china.sse')
    for d in dates:
        await train3(trainer_sets=trainer_sets,
                     end_date=d.strftime('%Y-%m-%d'),
                     symbol='IM')


if __name__ == "__main__":
    #agent_train1()
    #asyncio.run(agent_train2(symbol='IM'))
    asyncio.run(
        agent_train3(begin_date='2022-08-08',
                     end_date='2025-08-01',
                     symbol='IM'))
