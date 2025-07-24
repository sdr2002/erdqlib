from typing import Optional

import numpy as np

from erdqlib.src.common.rate import zero_rate, implied_yield, capitalization_factor
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def _compare_calculated_capitalization_factor(
        r_year: float, t_year: float,
        expected_cap_factor: Optional[float] = None, precision: Optional[int] = None
):
    """Example usage of capitalization factor"""

    cap_factor = capitalization_factor(r_year, t_year)
    LOGGER.info(f"capitalization factor given r_year({r_year:.4g}), t_year({t_year:.4g}): {cap_factor:.9f}")
    if expected_cap_factor and precision:
        np.testing.assert_approx_equal(cap_factor, desired=expected_cap_factor, significant=precision)
        LOGGER.info("Test passed")
    else:
        LOGGER.info("Test skipped, no expected value provided.")


def test_capitalization_factor():
    _compare_calculated_capitalization_factor(
        r_year=0.245 / 100,  # 0.245% annual rate
        t_year=4 / 12,  # 4 months
        expected_cap_factor=1.00082,  # Expected capitalization factor
        precision=6  # Precision for comparison
    )

    _compare_calculated_capitalization_factor(
        r_year=0.679 / 100,  # 0.245% annual rate
        t_year=5 / 12,  # 4 months
        expected_cap_factor=1.00028,  # Expected capitalization factor
        precision=3  # Precision for comparison
    )

    _compare_calculated_capitalization_factor(
        r_year=0.368 / 100,  # 0.245% annual rate
        t_year=1 / 12,  # 4 months
        expected_cap_factor=1.0003067,  # Expected capitalization factor
        precision=8  # Precision for comparison
    )


def test_zero_rate():
    capitalisation_factor: float = 1.0008167
    ytm: float = zero_rate(capitalisation_factor, 4 / 12)
    LOGGER.info(f"capitalisation_factor({capitalisation_factor:.7g}) -> annualized continuous rate: {ytm:.6g}")
    np.testing.assert_approx_equal(ytm, 0.245 / 100., significant=3)

    capitalisation_factor = 1.0002456
    ytm = zero_rate(capitalisation_factor, 60 / 360)
    LOGGER.info(f"capitalisation_factor({capitalisation_factor:.7g}) -> annualized continuous rate: {ytm:.6g}")
    np.testing.assert_approx_equal(ytm, 0.147 / 100., significant=3)


def test_implied_yield():
    """Example usage of implied yield"""
    maturity_year = 2.5
    zcb_price_t = 0.985  # Price at time t

    ytm = implied_yield(t_year=maturity_year, price_0_t=zcb_price_t)
    LOGGER.info(f"Implied yield: {ytm:.6f}")
    np.testing.assert_approx_equal(ytm, 0.006045, significant=4)
