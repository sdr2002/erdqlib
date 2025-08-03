from typing import Optional

import numpy as np

from erdqlib.src.ft.bcc import BCCParameters, BccFtiCalibrator
from erdqlib.src.common.option import OptionSide
from erdqlib.src.ft.bates import BatesFtiCalibrator
from erdqlib.src.ft.calibrator import FtiCalibrator
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def compare_calculated_eur_option_price_on_bcc_via_lewis(
        model_params: BCCParameters, side: OptionSide, K: float,
        expected_px: Optional[float] = None, precision: Optional[int] = None
):
    """Example usage of BCC pricing for European put option"""
    x0, T, r, kappa_heston, theta_heston, sigma_heston, rho_heston, v0_heston, lambd_merton, mu_merton, delta_merton = model_params.get_pricing_params(apply_shortrate=True)
    px_option_bcc: float = FtiCalibrator.calculate_option_price_lewis(
        x0=x0, T=T, r=r,
        characteristic_integral=lambda u: BccFtiCalibrator.calculate_integral_characteristic(
            u=u, S0=x0, K=K, T=T, r=r,
            kappa_heston=kappa_heston, theta_heston=theta_heston, sigma_heston=sigma_heston, rho_heston=rho_heston, v0_heston=v0_heston,
            lambd_merton=lambd_merton, mu_merton=mu_merton, delta_merton=delta_merton,
        ),
        K=K, side=side
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
        model_params= BCCParameters(
            # Short-rates
            r=0.75 / 100,
            kappa_cir = 0.068,
            theta_cir = 0.207,
            sigma_cir = 0.112,
            # Underlying and Option
            x0= 100.,
            T = 0.5,
            M = None,
            I = None,
            random_seed = None,
            # SV
            kappa_heston = 18.447,
            theta_heston = 0.026,
            sigma_heston = 0.978,
            rho_heston = -0.821,
            v0_heston = 0.035,

            # JD
            lambd_merton = 0.008,
            mu_merton = -0.600,
            delta_merton = 0.001
        ),
        K = 90.,
        side=OptionSide.CALL,
        expected_px=11.89,
        precision=4
    )

    compare_calculated_eur_option_price_on_bcc_via_lewis(
        model_params=BCCParameters(
            # Underlying and Option
            x0=100.,
            T=0.5,
            M=None,
            I=None,
            random_seed=None,
            # Short-rates
            r=0.75 / 100,
            kappa_cir=0.068,
            theta_cir=0.207,
            sigma_cir=0.112,
            # SV
            kappa_heston=18.447,
            theta_heston=0.026,
            sigma_heston=0.978,
            rho_heston=-0.821,
            v0_heston=0.035,
            # JD
            lambd_merton=0.008,
            mu_merton=-0.600,
            delta_merton=0.001
        ),
        K=90.,
        side=OptionSide.PUT,
        expected_px=1.41,
        precision=3
    )

    compare_calculated_eur_option_price_on_bcc_via_lewis(
        model_params=BCCParameters(
            x0=60.,
            T=150. / 365.,
            M=None,
            I=None,
            random_seed=None,

            r=1.75 / 100.,
            kappa_cir=0.85,
            theta_cir=0.5,
            sigma_cir=0.35,

            kappa_heston=30.,
            theta_heston=0.065,
            sigma_heston=0.18,
            rho_heston=0.65,
            v0_heston=0.35,

            lambd_merton=2.8,
            mu_merton=0.5,
            delta_merton=0.75
        ),
        K=65.,
        side=OptionSide.PUT,
        expected_px=33.37,
        precision=4
    )

    compare_calculated_eur_option_price_on_bcc_via_lewis(
        model_params=BCCParameters(
            x0=159.,
            T=500. / 365.,
            M=None,
            I=None,
            random_seed=None,

            r=1 / 100,
            kappa_cir=0.065,
            theta_cir=0.45,
            sigma_cir=0.35,

            kappa_heston=20.,
            theta_heston=0.065,
            sigma_heston=0.08,
            rho_heston=-0.25,
            v0_heston=0.015,

            lambd_merton=1.8,
            mu_merton=-0.75,
            delta_merton=0.55
        ),
        K=170.,
        side=OptionSide.PUT,
        expected_px=70.98,
        precision=4
    )

    compare_calculated_eur_option_price_on_bcc_via_lewis(
        model_params=BCCParameters(
            r=1. / 100.,
            kappa_cir=0.065,
            theta_cir=0.45,
            sigma_cir=0.35,

            x0=159.,
            T=500. / 365.,
            M=None,
            I=None,
            random_seed=None,

            kappa_heston=20.,
            theta_heston=0.065,
            sigma_heston=0.08,
            rho_heston=-0.25,
            v0_heston=0.015,

            lambd_merton=1.8,
            mu_merton=-0.75,
            delta_merton=0.55
        ),
        K=170.,
        side=OptionSide.CALL,
        expected_px=66.46,
        precision=4
    )


def test_compare_eur_option_price_on_bcc_and_bates_via_lewis():
    """Compare BCC and Bates pricing for European options"""
    bcc_params: BCCParameters = BCCParameters(
        r=1.75 / 100.,
        kappa_cir=0.85,
        theta_cir=0.5,
        sigma_cir=0.35,

        x0=60.,
        T=150. / 365.,
        M=None,
        I=None,
        random_seed=None,

        kappa_heston=30.,
        theta_heston=0.065,
        sigma_heston=0.18,
        rho_heston=0.65,
        v0_heston=0.35,

        lambd_merton=2.8,
        mu_merton=0.5,
        delta_merton=0.75
    )

    K: float = 65.0
    side: OptionSide = OptionSide.CALL

    x0, T, r, kappa_heston, theta_heston, sigma_heston, rho_heston, v0_heston, lambd_merton, mu_merton, delta_merton = bcc_params.get_pricing_params(apply_shortrate=False)
    px_bates: float = FtiCalibrator.calculate_option_price_lewis(
        x0=x0, T=T, r=r,
        characteristic_integral=lambda u: BccFtiCalibrator.calculate_integral_characteristic(
            u=u, S0=x0, K=K, T=T, r=r,
            kappa_heston=kappa_heston, theta_heston=theta_heston, sigma_heston=sigma_heston, rho_heston=rho_heston, v0_heston=v0_heston,
            lambd_merton=lambd_merton, mu_merton=mu_merton, delta_merton=delta_merton,
        ),
        K=K, side=side
    )
    LOGGER.info(f"{side} value under Bates: {px_bates}")

    x0, T, r, kappa_heston, theta_heston, sigma_heston, rho_heston, v0_heston, lambd_merton, mu_merton, delta_merton = bcc_params.get_pricing_params(apply_shortrate=True)
    px_bcc: float = FtiCalibrator.calculate_option_price_lewis(
        x0=x0, T=T, r=r,
        characteristic_integral=lambda u: BccFtiCalibrator.calculate_integral_characteristic(
            u=u, S0=x0, K=K, T=T, r=r,
            kappa_heston=kappa_heston, theta_heston=theta_heston, sigma_heston=sigma_heston, rho_heston=rho_heston, v0_heston=v0_heston,
            lambd_merton=lambd_merton, mu_merton=mu_merton, delta_merton=delta_merton,
        ),
        K=K, side=side
    )
    LOGGER.info(f"{side} value under BCC (1997): {px_bcc}")

    np.testing.assert_approx_equal(px_bates, desired=30.461673779556723, significant=9)
    np.testing.assert_approx_equal(px_bcc, desired=30.803270624793345, significant=9)
    np.testing.assert_approx_equal(px_bcc - px_bates, desired=0.3416, significant=4)
    LOGGER.info("Test passed")


def test_bcc_calibration():
    raise NotImplementedError("Test for BCC calibration is not implemented yet.")
