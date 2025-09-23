import asyncio, time, pdb
from dotenv import load_dotenv

load_dotenv()

from payload.forge.actuator import Actuator as ForgeActuator
from payload.rucge.actuator import Actuator as RucgeFactuator


async def main():
    base_path = "records/factors/"
    base_url = "http://0.0.0.0:8001"
    actuator1 = ForgeActuator(url=base_url, base_path=base_path)
    #await actuator1.interpre(
    #    category='basic',
    #    expression=
    #    "MMAX(18,MSmart(10,'oi015_5_10_1',MA(4,MSTD(4,'rv001_5_10_0_2'))))")
    #await actuator1.run1(category='basic')
    await actuator1.evolve(category='basic')


if __name__ == "__main__":
    count = 50
    while count > 0:
        asyncio.run(main())
        time.sleep(2)
        count -= 1
