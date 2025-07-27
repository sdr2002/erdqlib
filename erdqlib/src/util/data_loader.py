from typing import Optional, Any, Callable

import numpy as np
import pandas as pd

from erdqlib.src.common.option import OptionDataColumn
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def load_option_data(
        path_str: str,
        S0: float,
        r_provider: Callable[[Any], float] = None,
        strike_tol: Optional[float] = 0.02,
        n_bdays_per_year: float = 365.0,
        days_to_maturity_target: Optional[int] = None
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
        if dcol in df_options.columns:
            df_options[dcol] = pd.to_datetime(df_options[dcol])

    # Option Selection
    if strike_tol:
        df_options = df_options[(np.abs(df_options[OptionDataColumn.STRIKE] - S0) / S0) < strike_tol]
    df_options = df_options.rename(columns={c: c.upper() for c in OptionDataColumn.get_callput_str()})

    # Adding Time-to-Maturity and constant short-rates
    if (OptionDataColumn.MATURITY in df_options.columns) and (OptionDataColumn.DATE in df_options.columns):
        df_options[OptionDataColumn.DAYSTOMATURITY] = (df_options[OptionDataColumn.MATURITY] - df_options[OptionDataColumn.DATE]).dt.days
    elif OptionDataColumn.DAYSTOMATURITY not in df_options.columns:
        raise ValueError("Option data must contain either 'Maturity' and 'Date' columns or 'DaysToMaturity' column.")

    if days_to_maturity_target:
        df_options = df_options[df_options[OptionDataColumn.DAYSTOMATURITY] == days_to_maturity_target]
        if df_options.empty:
            raise ValueError(f"No options found with {days_to_maturity_target} days to maturity.")

    df_options[OptionDataColumn.TENOR] = df_options[OptionDataColumn.DAYSTOMATURITY] / n_bdays_per_year
    if r_provider:
        df_options[OptionDataColumn.RATE] = df_options.apply(
            func=lambda row: r_provider(row[OptionDataColumn.TENOR]),
            axis=1
        )

    return df_options.reset_index(drop=True)
