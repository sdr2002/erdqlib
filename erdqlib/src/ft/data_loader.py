from enum import StrEnum
from typing import Optional, List, Any, Callable

import numpy as np
import pandas as pd

from erdqlib.src.common.option import OptionDataColumn
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def load_option_data(
        path_str: str, S0: float, r_provider: Callable[[Any], float],
        strike_tol: Optional[float] = 0.02
) -> pd.DataFrame:
    """Load option data from CSV file.

    :param path_str: Path to the CSV file containing option data.
    :param S0: Underlying asset price at the time of option pricing.
    :param r_provider: Callable that provides the risk-free rate for a given time-to-maturity.
    :param strike_tol: Tolerance level to select ATM options (percent around ITM/OTM options), default is 0.02 (2%).
    """
    LOGGER.info(f"Loading option data from {path_str}")
    df_options: pd.DataFrame = pd.read_csv(path_str)
    for dcol in OptionDataColumn.get_datetime_cols():
        df_options[dcol] = pd.to_datetime(df_options[dcol])

    # Option Selection
    if strike_tol:
        df_options = df_options[(np.abs(df_options[OptionDataColumn.STRIKE] - S0) / S0) < strike_tol]
    df_options = df_options.rename(columns={c: c.upper() for c in OptionDataColumn.get_callput_str()})
    # Adding Time-to-Maturity and constant short-rates
    for row, option in df_options.iterrows():
        T = (option[OptionDataColumn.MATURITY] - option[OptionDataColumn.DATE]).days / 365.0
        df_options.loc[row, OptionDataColumn.TENOR] = T
        df_options.loc[row, OptionDataColumn.RATE] = r_provider(T)

    return df_options