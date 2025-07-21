
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

from erdqlib.src.mc.dynamics import DynamicsParameters

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
    