from typing import Optional

import numpy as np

from erdqlib.src.common.rate import annualized_continuous_rate, implied_yield, capitalization_factor
from erdqlib.scripts.sm_bates import B96_eur_option_value_lewis
from erdqlib.scripts.sm_bcc import BCC_eur_option_value_lewis, BCCParameters
from erdqlib.src.common.option import OptionSide
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
        t_year = 4 / 12,  # 4 months
        expected_cap_factor=1.00082,  # Expected capitalization factor
        precision=6  # Precision for comparison
    )

    _compare_calculated_capitalization_factor(
        r_year=0.679/100,  # 0.245% annual rate
        t_year=5/12,  # 4 months
        expected_cap_factor=1.00028,  # Expected capitalization factor
        precision=3  # Precision for comparison
    )

    _compare_calculated_capitalization_factor(
        r_year=0.368 / 100,  # 0.245% annual rate
        t_year=1 / 12,  # 4 months
        expected_cap_factor=1.0003067,  # Expected capitalization factor
        precision=8  # Precision for comparison
    )


def test_annualized_continuous_rate():
    capitalisation_factor: float = 1.0008167
    ytm: float = annualized_continuous_rate(capitalisation_factor, 4/12)
    LOGGER.info(f"capitalisation_factor({capitalisation_factor:.7g}) -> annualized continuous rate: {ytm:.6g}")
    np.testing.assert_approx_equal(ytm, 0.245/100., significant=3)

    capitalisation_factor = 1.0002456
    ytm = annualized_continuous_rate(capitalisation_factor, 60/360)
    LOGGER.info(f"capitalisation_factor({capitalisation_factor:.7g}) -> annualized continuous rate: {ytm:.6g}")
    np.testing.assert_approx_equal(ytm, 0.147 / 100., significant=3)


def test_implied_yield():
    """Example usage of implied yield"""
    maturity_year = 2.5
    zcb_price_t = 0.985    # Price at time t

    ytm = implied_yield(maturity_year, zcb_price_t)
    LOGGER.info(f"Implied yield: {ytm:.6f}")
    np.testing.assert_approx_equal(ytm, 0.006045, significant=4)


def compare_calculated_eur_option_price_on_bcc_via_lewis(
        model_params: BCCParameters, side: OptionSide,
        expected_px: Optional[float] = None, precision: Optional[int] = None
):
    """Example usage of BCC pricing for European put option"""
    px_option_bcc: float = BCC_eur_option_value_lewis(
        *model_params.get_pricing_params(apply_shortrate=True), side
    )
    LOGGER.info(f"{side} value under BCC (1997): {px_option_bcc} from {model_params.to_str(indent=None)}")
    if expected_px and precision:
        np.testing.assert_approx_equal(px_option_bcc, expected_px, significant=precision)
        LOGGER.info("Test passed")
    else:
        LOGGER.info(f"Test skipped, no expected value provided.")


def test_lewis_price_eur_option_on_bcc():
    """Test BCC model pricing via Lewis method for European options"""
    compare_calculated_eur_option_price_on_bcc_via_lewis(
        BCCParameters(
            # Short-rates
            r0=0.75 / 100,
            kappa_r = 0.068,
            theta_r = 0.207,
            sigma_r = 0.112,
            # Underlying and Option
            S0 = 100.,
            K = 90.,
            Ti = 0.,
            Tf = 0.5,
            # SV
            kappa_v = 18.447,
            theta_v = 0.026,
            sigma_v = 0.978,
            rho = -0.821,
            v0 = 0.035,

            # JD
            lamb = 0.008,
            mu = -0.600,
            delta = 0.001
        ),
        OptionSide.CALL,
        expected_px=11.89,
        precision=4
    )

    compare_calculated_eur_option_price_on_bcc_via_lewis(
        BCCParameters(
            S0=100.,
            K=90.,
            Ti=0.,
            Tf=0.5,
            r0=0.75 / 100,
            kappa_r=0.068,
            theta_r=0.207,
            sigma_r=0.112,
            kappa_v=18.447,
            theta_v=0.026,
            sigma_v=0.978,
            rho=-0.821,
            v0=0.035,
            lamb=0.008,
            mu=-0.600,
            delta=0.001
        ),
        side=OptionSide.PUT,
        expected_px=1.41,
        precision=3
    )

    compare_calculated_eur_option_price_on_bcc_via_lewis(
        BCCParameters(
            r0=1.75 / 100.,
            kappa_r=0.85,
            theta_r=0.5,
            sigma_r=0.35,

            S0=60.,
            K=65.,
            Ti=0.,
            Tf=150. / 365.,

            kappa_v=30.,
            theta_v=0.065,
            sigma_v=0.18,
            rho=0.65,
            v0=0.35,

            lamb=2.8,
            mu=0.5,
            delta=0.75
        ),
        side=OptionSide.PUT,
        expected_px=33.37,
        precision=4
    )

    compare_calculated_eur_option_price_on_bcc_via_lewis(
        BCCParameters(
            r0=1 / 100,
            kappa_r=0.065,
            theta_r=0.45,
            sigma_r=0.35,

            S0=159.,
            K=170.,
            Ti=0.,
            Tf=500. / 365.,

            kappa_v=20.,
            theta_v=0.065,
            sigma_v=0.08,
            rho=-0.25,
            v0=0.015,

            lamb=1.8,
            mu=-0.75,
            delta=0.55
        ),
        side=OptionSide.PUT,
        expected_px=70.98,
        precision=4
    )

    compare_calculated_eur_option_price_on_bcc_via_lewis(
        BCCParameters(
            r0=1 / 100,
            kappa_r=0.065,
            theta_r=0.45,
            sigma_r=0.35,

            S0=159.,
            K=170.,
            Ti=0.,
            Tf=500. / 365.,

            kappa_v=20.,
            theta_v=0.065,
            sigma_v=0.08,
            rho=-0.25,
            v0=0.015,

            lamb=1.8,
            mu=-0.75,
            delta=0.55
        ),
        side=OptionSide.CALL,
        expected_px=66.46,
        precision=4
    )


def test_compare_eur_option_price_on_bcc_and_bates_via_lewis():
    """Compare BCC and Bates pricing for European options"""
    bcc_params: BCCParameters = BCCParameters(
        r0=1.75 / 100.,
        kappa_r=0.85,
        theta_r=0.5,
        sigma_r=0.35,

        S0=60.,
        K=65.,
        Ti=0.,
        Tf=150. / 365.,

        kappa_v=30.,
        theta_v=0.065,
        sigma_v=0.18,
        rho=0.65,
        v0=0.35,

        lamb=2.8,
        mu=0.5,
        delta=0.75
    )

    side: OptionSide = OptionSide.CALL

    px_bates: float = B96_eur_option_value_lewis(
        *bcc_params.get_pricing_params(apply_shortrate=False), side
    )
    LOGGER.info(f"{side} value under Bates: {px_bates}")

    px_bcc: float = BCC_eur_option_value_lewis(
        *bcc_params.get_pricing_params(apply_shortrate=True), side
    )
    LOGGER.info(f"{side} value under BCC (1997): {px_bcc}")

    np.testing.assert_approx_equal(px_bates, desired=30.461673779556723, significant=9)
    np.testing.assert_approx_equal(px_bcc, desired=30.803270624793345, significant=9)
    np.testing.assert_approx_equal(px_bcc - px_bates, desired=0.3416, significant=4)
    LOGGER.info("Test passed")
