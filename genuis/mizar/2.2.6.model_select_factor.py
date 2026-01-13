from dotenv import load_dotenv
import pdb
load_dotenv()
from kdutils.tactix import Tactix
from lib.fsr001.lassocv import train_model


if __name__ == '__main__':
    variant = Tactix().start()
    train_model(method=variant.method,
                instruments=variant.instruments,
                task_id=variant.task_id,
                period=variant.period,
                name=variant.name)