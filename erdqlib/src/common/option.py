from dataclasses import dataclass
from enum import StrEnum, auto

from typing import List


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


class OptionDataColumn(StrEnum):
    STRIKE = "Strike"
    MATURITY = "Maturity"
    DATE = "Date"
    CALL = "Call"
    PUT = "Put"

    TENOR = "T"
    RATE = "r"

    MODEL = "Model"

    DAYSTOMATURITY = "DaysToMaturity"

    @staticmethod
    def get_callput_str() -> List[str]:
        return [str(OptionDataColumn.CALL.value), str(OptionDataColumn.PUT.value)]

    @staticmethod
    def get_datetime_cols() -> List[str]:
        """Get the list of columns that should be parsed as datetime."""
        return [str(OptionDataColumn.DATE.value), str(OptionDataColumn.MATURITY.value)]
