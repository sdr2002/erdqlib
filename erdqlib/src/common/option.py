from dataclasses import dataclass
from enum import StrEnum, auto


class OptionSide(StrEnum):
    CALL = auto()
    PUT = auto()


class OptionType(StrEnum):
    EUROPEAN = auto()
    AMERICAN = auto()
    ASIAN = auto()
    DOWNANDIN = auto()
    UPANDIN = auto()


@dataclass
class OptionInfo:
    type: OptionType
    K: float  # strike price
    side: OptionSide


@dataclass
class BarrierOptionInfo(OptionInfo):
    barrier: float
