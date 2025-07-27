
from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Any, List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from erdqlib.src.common.option import OptionDataColumn, OptionSide
from erdqlib.src.mc.dynamics import DynamicsParameters


MIN_RMSE: float = 100.0
MIN_MSE: float = 500.0

class FtMethod(StrEnum):
    LEWIS = auto()
    CARRMADAN = auto()


"""
Abstract class FtiCalibrator with abstract&static methods for calibration.
- calculate_characteristic_function
- calculate_integral_characteristic_function
- calculate_option_price
- calculate_error_function
- calibrate
"""
class FtiCalibrator(ABC):
    @staticmethod
    @abstractmethod
    def calculate_characteristic(*args, **kwargs) -> complex:
        raise NotImplementedError("Child class must implement")

    @staticmethod
    @abstractmethod
    def calculate_integral_characteristic(*args, **kwargs) -> float:
        raise NotImplementedError("Child class must implement")

    @staticmethod
    @abstractmethod
    def calculate_option_price(*args, **kwargs) -> float:
        raise NotImplementedError("Child class must implement")

    @staticmethod
    @abstractmethod
    def calculate_option_price_batch(df_options: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Child class must implement")

    @staticmethod
    @abstractmethod
    def calculate_error(
        p0: np.ndarray, df_options: pd.DataFrame, print_iter: List[int], min_MSE: List[float], s0: float, side: Any
    ) -> float:
        raise NotImplementedError("Child class must implement")

    @staticmethod
    @abstractmethod
    def calibrate(
        df_options: pd.DataFrame,
        S0: float,
        r: Optional[float],
        side: Any,
        search_grid: Optional[Dict[str, Tuple[float, float, float]]] = None
    ) -> DynamicsParameters:
        raise NotImplementedError("Child class must implement")


def plot_calibration_result(df_options: pd.DataFrame, model_values: np.ndarray, side: OptionSide):
    """Plot market and model prices for each maturity and OptionSide (full calibration)."""
    df_options = df_options.copy()
    df_options[OptionDataColumn.MODEL] = model_values
    for maturity, df_options_per_maturity in df_options.groupby(OptionDataColumn.DAYSTOMATURITY):
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.grid()
        plt.title(f"{side.name} at Maturity {maturity}")
        plt.ylabel("option values")
        plt.plot(df_options_per_maturity[OptionDataColumn.STRIKE], df_options_per_maturity[side.name], "b", label="market")
        plt.plot(df_options_per_maturity[OptionDataColumn.STRIKE], df_options_per_maturity[OptionDataColumn.MODEL], "ro", label="model")
        plt.legend(loc=0)
        axis1=[
            min(df_options_per_maturity[OptionDataColumn.STRIKE]) - 2.5,
            max(df_options_per_maturity[OptionDataColumn.STRIKE]) + 2.5,
            min(df_options_per_maturity[side.name]) - 2.5,
            max(df_options_per_maturity[side.name]) + 2.5,
        ]
        plt.xlabel("Strike")
        plt.axis(axis1)  # type: ignore

        plt.subplot(212)
        plt.grid()
        diffs = df_options_per_maturity[OptionDataColumn.MODEL].values - df_options_per_maturity[side.name].values
        plt.bar(
            x=df_options_per_maturity[OptionDataColumn.STRIKE].values,
            height=diffs
        )
        plt.xlabel("Strike")
        plt.ylabel("difference")
        axis2 = [
            min(df_options_per_maturity[OptionDataColumn.STRIKE]) - 2.5,
            max(df_options_per_maturity[OptionDataColumn.STRIKE]) + 2.5,
            min(diffs) * 1.1,
            max(diffs) * 1.1,
        ]
        plt.axis(axis2)  # type: ignore
        plt.tight_layout()
        plt.show()