# -*- coding: utf-8 -*-

from enum import Enum
from enum import unique


class StrEnum(str, Enum):
    pass


@unique
class Factors(StrEnum):
    CLOSE = 'close'
    PE = 'pe'
    OPEN = 'open'
    VOLUME = 'volume'
    HIGH = 'high'
    LOW = 'low'
