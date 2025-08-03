from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fmin

from erdqlib.src.common.option import OptionDataColumn
from erdqlib.src.common.rate import SplineCurve, ForwardsLadder
from erdqlib.src.mc.cir import CirDynamicsParameters
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)
MIN_MSE = 1.0


class CirCalibrator:
    """Short rate model calibration for CIR (1985) model"""
    @staticmethod
    def calculate_forward_rate(alpha: np.ndarray, maturities_ladder: np.ndarray, r0: float) -> np.array:
        """
        Forward rates in CIR (1985) model
        The set of parameters is called alpha and include Kappa_r, Theta_r and Sigma_r
        """

        kappa_r, theta_r, sigma_r = alpha

        t = maturities_ladder
        g = np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)

        s1: np.ndarray = (kappa_r * theta_r * (np.exp(g * t) - 1)) / (
                2 * g + (kappa_r + g) * (np.exp(g * t) - 1)
        )

        s2: np.ndarray = r0 * (
                (4 * g ** 2 * np.exp(g * t)) / (2 * g + (kappa_r + g) * (np.exp(g * t) - 1) ** 2)
        )

        return s1 + s2

    @staticmethod
    def calculate_error(
            cir_params: np.ndarray,
            curve_forward_rates: np.ndarray, maturities_ladder: np.ndarray, r0: float,
            print_iter: Optional[List[float]] = None, min_MSE: Optional[List[float]] = None
    ):
        """Error function to calibrate CIR (1985) model"""
        kappa_r, theta_r, sigma_r = cir_params

        # Few remarks to avoid problems for certain values of parameters:
        if 2 * kappa_r * theta_r < sigma_r ** 2:
            return 100
        if kappa_r < 0 or theta_r < 0 or sigma_r < 0.001:
            return 100

        forward_rates: np.ndarray = CirCalibrator.calculate_forward_rate(cir_params, maturities_ladder=maturities_ladder, r0=r0)
        mse: float = np.sum((curve_forward_rates - forward_rates) ** 2) / len(curve_forward_rates)
        if min_MSE is not None:
            min_MSE[0] = min(min_MSE[0], mse)
        if print_iter is not None:
            if print_iter[0] % 25 == 0:
                LOGGER.info(f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in cir_params)}] | {mse:7.3g} | {min_MSE[0]:7.3g}")
            print_iter[0] += 1

        return mse

    @staticmethod
    def calibrate(
        r0: float, curve_forward_rates: np.ndarray, maturities_ladder: np.ndarray
    ) -> CirDynamicsParameters:
        """CIR (1985) Calibration via minimizing Forward rate differences"""
        print_iter = [0]
        min_MSE = [MIN_MSE]

        LOGGER.info("CIR calibration begins: Fmin")
        c_opt: np.ndarray = fmin(
            lambda p, f=curve_forward_rates, r=r0: CirCalibrator.calculate_error(
                cir_params=p, curve_forward_rates=f, maturities_ladder=maturities_ladder, r0=r,
                print_iter=print_iter, min_MSE=min_MSE
            ),
            [1.0, 0.02, 0.1],  # Initial kappa_r, theta_r, sigma_r hypothesis
            xtol=0.00001, ftol=0.00001, maxiter=300, maxfun=500,
            full_output= False, retall=False, disp=False
        )
        opt = CirDynamicsParameters.from_calibration_output(opt_arr=c_opt, x0=r0)  # type: ignore
        LOGGER.info(f"CIR calibrated: {opt}")
        return opt


def plot_interpolated_curve(
        maturities: np.ndarray,
        zcb_rates: np.ndarray,
        maturities_ladder: np.ndarray,
        interpolated_rates: np.ndarray,
        first_derivatives: np.ndarray,
        zcb_forward_rates: np.ndarray
):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(maturities, zcb_rates, "r.", markersize=15, label="Market quotes")
    ax.plot(maturities_ladder, interpolated_rates, "--", markersize=10, label="Spot rate")
    ax.plot(maturities_ladder, first_derivatives, "g--", markersize=10, label="Spot rate time derivative")
    ax.set_xlabel("Time Horizon")
    ax.set_ylabel("Zero forward rate")
    ax2 = ax.twinx()
    ax2.plot(maturities_ladder, zcb_forward_rates, "b--", markersize=10, label="Forward rate")
    fig.suptitle("Term Structure Euribor")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.show()


def plot_calibrated_cir(
        model_params: np.ndarray,
        maturities_ladder: np.ndarray,
        market_forward_rates: np.ndarray,
        r0: float
):
    """Plots market and calibrated forward rate curves."""
    model_forward_rates = CirCalibrator.calculate_forward_rate(model_params, maturities_ladder, r0=r0)
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.title("CIR model calibration")
    plt.ylabel("Forward rate $f(0,T)$")
    plt.plot(maturities_ladder, market_forward_rates, "ro", label="market")
    plt.plot(maturities_ladder, model_forward_rates, "b--", label="model")
    plt.legend(loc=0)
    axis_tenor1: List[float] = [
        min(maturities_ladder) - 0.05,
        max(maturities_ladder) + 0.05,
        min(market_forward_rates) - 0.005,
        max(market_forward_rates) * 1.1
    ]
    plt.axis(axis_tenor1) # type: ignore

    plt.subplot(212)
    wi = 0.02
    plt.bar(maturities_ladder - wi / 2, model_forward_rates - market_forward_rates, width=wi)
    plt.xlabel("Time horizon")
    plt.ylabel("Difference")
    axis_tenor2: List[float] = [
        min(maturities_ladder) - 0.05,
        max(maturities_ladder) + 0.05,
        min(model_forward_rates - market_forward_rates) * 1.1,
        max(model_forward_rates - market_forward_rates) * 1.1,
    ]
    plt.axis(axis_tenor2)  # type: ignore
    plt.tight_layout()
    plt.show()


def ex_calibration(
    skip_plot: bool
):
    """Example of CIR (1985) calibration of euribor dynamics."""
    # Euribor Market data
    euribor_df: pd.DataFrame = pd.read_csv(
        # get_path_from_package("erdqlib@src/ft/data/euribor_20140930.csv")
        get_path_from_package("erdqlib@examples/data/sm_gwp1_euribor.csv")
    )
    LOGGER.info(f"Rates:\n{euribor_df.to_markdown(index=False)}")

    maturities: np.ndarray = euribor_df[OptionDataColumn.TENOR].values  # Maturities in years with 30/360 convention
    euribors: np.ndarray = euribor_df[OptionDataColumn.RATE].values  # Euribor rates in rate unit

    # Interpolation and Forward rates via Cubic spline
    scurve: SplineCurve = SplineCurve()
    scurve.update_curve(maturities=maturities, spot_rates=euribors)
    forward_rates: ForwardsLadder = scurve.calculate_forward_rates(t_f=1.0)

    params_cir: CirDynamicsParameters = CirCalibrator.calibrate(
        r0=scurve.get_r0(),
        curve_forward_rates=forward_rates.rates,
        maturities_ladder=forward_rates.maturities
    )  # [0.06190266 0.23359687 0.14892987]
    LOGGER.info(f"CIR params:{params_cir}")

    if not skip_plot:
        scurve.plot_curve()
        plot_calibrated_cir(
            model_params=params_cir.get_value_arr(),
            maturities_ladder=forward_rates.maturities,
            market_forward_rates=forward_rates.rates,
            r0=scurve.get_r0()
        )


if __name__ == "__main__":
    ex_calibration(skip_plot=False)  # Example of CIR calibration