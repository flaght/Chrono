import os, pdb, asyncio, time
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
from agent import IndicatorPredict, CloutoPredict, MoneyFlowPredict, HotMoneyPredict, ChipPredict


async def predict_with_semaphore(predictor, semaphore: asyncio.Semaphore,
                                 date: str):
    """
    一个异步包装函数，它在使用 predictor 进行预测之前，会先从 semaphore 获取许可。
    
    Args:
        predictor: 一个 Predictor 类的实例 (例如 CloutoPredict)。
        semaphore (asyncio.Semaphore): 用于控制并发的信号量。
        date (str): 预测日期。

    Returns:
        The result of the prediction or the exception if it fails.
    """
    # 3. 在包装协程内获取 Semaphore
    # async with 语句确保了即使在预测过程中发生异常，信号量也总能被正确释放。
    async with semaphore:
        # 当协程执行到这里时，它已经成功获取了一个“令牌”。
        # 如果令牌已满，它会在上一行异步地等待。
        print(f"[{predictor.agent.name}] 获取到信号量许可，开始执行预测...")

        try:
            # 执行实际的预测调用
            result = await predictor.agenerate_prediction(
                date=date, predict_data=predictor.create_data(date=date))
            return result
        except Exception as e:
            # 捕获并返回异常，这样 gather 就不会中断
            print(f"[{predictor.agent.name}] 预测时发生错误: {e}")
            return e


def agent_predict1():
    end_date = '2025-02-14'
    begin_date = advanceDateByCalendar('china.sse', end_date,
                                       '-{0}b'.format(10)).strftime('%Y-%m-%d')
    symbol = '688585'  #'601519'
    predict = ChipPredict(symbol=symbol,
                          memory_path=os.path.join("records"),
                          config_path=os.path.join("agent"),
                          date='2025-01-27')

    predict.prepare_data(begin_date=end_date, end_date=end_date)
    predict_data = predict.create_data(date=end_date)

    response = asyncio.run(
        predict.agenerate_prediction(date=end_date, predict_data=predict_data))
    print(response)


async def agent_predict2(symbol):
    model_date = '2025-01-27'
    end_date = '2025-02-14'
    predict_sets = [
        CloutoPredict(date=model_date,
                      memory_path=os.path.join("records"),
                      symbol=symbol),
        IndicatorPredict(date=model_date,
                         memory_path=os.path.join("records"),
                         symbol=symbol),
        MoneyFlowPredict(date=model_date,
                         memory_path=os.path.join("records"),
                         symbol=symbol),
        HotMoneyPredict(date=model_date,
                        memory_path=os.path.join("records"),
                        symbol=symbol),
        ChipPredict(date=model_date,
                    memory_path=os.path.join("records"),
                    symbol=symbol)
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


async def agent_predict3(symbol):
    model_date = '2025-01-27'
    end_date = '2025-02-14'

    # --- 1. 创建 Semaphore 实例 ---
    # 设置最大并发数为 2。这意味着最多只有 2 个 agenerate_prediction 会同时进行。
    # 你可以根据你的 API 速率限制调整这个值。
    MAX_CONCURRENT_REQUESTS = 2
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    print(f"信号量已创建，最大并发预测数为: {MAX_CONCURRENT_REQUESTS}")

    predict_sets = [
        CloutoPredict(date=model_date,
                      memory_path=os.path.join("records"),
                      symbol=symbol),
        IndicatorPredict(date=model_date,
                         memory_path=os.path.join("records"),
                         symbol=symbol),
        MoneyFlowPredict(date=model_date,
                         memory_path=os.path.join("records"),
                         symbol=symbol),
        HotMoneyPredict(date=model_date,
                        memory_path=os.path.join("records"),
                        symbol=symbol),
        ChipPredict(date=model_date,
                    memory_path=os.path.join("records"),
                    symbol=symbol)
    ]

    end_date = advanceDateByCalendar('china.sse', end_date,
                                     '-{0}b'.format(1)).strftime('%Y-%m-%d')

    for p in predict_sets:
        p.prepare_data(begin_date=end_date, end_date=end_date)

    # 创建一个任务列表，这次调用的是我们的包装函数 predict_with_semaphore
    tasks = [
        predict_with_semaphore(p, semaphore, end_date) for p in predict_sets
    ]
    print("已创建所有并发任务，准备使用 asyncio.gather 运行...")
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)


if __name__ == "__main__":
    agent_predict1()
    #asyncio.run(agent_predict3(symbol='688585'))
