import numpy as np

from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)

def capitalization_factor(r_year: float, t_year: float) -> float:
    """Capitalization factor: 1 + n_year(convention) * rate_year"""
    return 1. + t_year * r_year


def annualized_continuous_rate(capitalization_factor: float, t_year: float) -> float:
    """Annualized continuous rate from capitalization factor and time in years"""
    if t_year <= 0:
        raise ValueError("Time in years must be positive")
    return np.log(capitalization_factor) / t_year


def implied_yield(maturity_year: float, zcb_price_t: float, zcb_price_0: float = 1.):
    """Implied yield from zero-coupon bond prices at time t and t=0: Y(0,T) = -log(B_0(T)/B_0(0)) / T"""
    return -np.log(zcb_price_t / zcb_price_0) / maturity_year


def instantaneous_rate(zcb_price: float, t_year: float) -> float:
    """Instantaneous rate from Zero-Coupon-Bond price and time in years"""
    if t_year <= 0:
        raise ValueError("Time in years must be positive")
    return -np.log(zcb_price) / t_year


if __name__ == "__main__":
    # Example usage
    LOGGER.info(capitalization_factor(0.245 / 100, 4 / 12))  # 0.245% annual rate, 4 months