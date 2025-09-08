import asyncio
from dotenv import load_dotenv

load_dotenv()

from payload.forge.actuator import Actuator as ForgeActuator
from payload.rucge.actuator import Actuator as RucgeFactuator


async def main():
    base_path = "records/factors/"
    base_url = "http://0.0.0.0:8001"
    actuator = ForgeActuator(url=base_url, base_path=base_path)
    await actuator.run1(category='basic')


if __name__ == "__main__":
    count = 1
    while count > 0:
        asyncio.run(main())
        count -= 1
