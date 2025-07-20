from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.optimize import fmin

from erdqlib.src.common.rate import capitalization_factor, annualized_continuous_rate
from erdqlib.src.mc.dynamics import ModelParameters, DynamicsParameters
from erdqlib.src.common.option import OptionDataColumn
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)


@dataclass
class CirDynamicsParameters(DynamicsParameters):
    kappa_cir: float
    theta_cir: float
    sigma_cir: float

    def get_value_arr(self) -> np.ndarray:
        return np.array([self.kappa_cir, self.theta_cir, self.sigma_cir])

    @staticmethod
    def from_opt_result(opt: np.ndarray) -> "CirDynamicsParameters":
        return CirDynamicsParameters(
            S0=None, r=None,
            kappa_cir=float(opt[0]),
            theta_cir=float(opt[1]),
            sigma_cir=float(opt[2])
        )


@dataclass
class CirParameters(ModelParameters, CirDynamicsParameters):
    pass


class CirCalibrator:
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
    def calculate_error(alpha: np.ndarray, curve_forward_rates: np.ndarray, maturities_ladder: np.ndarray, r0: float):
        """Error function to calibrate CIR (1985) model"""
        kappa_r, theta_r, sigma_r = alpha

        # Few remarks to avoid problems for certain values of parameters:
        if 2 * kappa_r * theta_r < sigma_r ** 2:
            return 100
        if kappa_r < 0 or theta_r < 0 or sigma_r < 0.001:
            return 100

        forward_rates: np.ndarray = CirCalibrator.calculate_forward_rate(alpha, maturities_ladder=maturities_ladder, r0=r0)
        MSE: float = np.sum((curve_forward_rates - forward_rates) ** 2) / len(curve_forward_rates)

        return MSE

    @staticmethod
    def calibrate(
        r0: float, curve_forward_rates: np.ndarray, maturities_ladder: np.ndarray
    ) -> CirDynamicsParameters:
        """CIR (1985) Calibration via minimizing Forward rate differences"""
        # np array of kappa_r, theta_r, sigma_r
        opt: np.ndarray = fmin(  # type: ignore
            lambda cir_parms, f=curve_forward_rates, r=r0: CirCalibrator.calculate_error(
                alpha=cir_parms, curve_forward_rates=f, maturities_ladder=maturities_ladder, r0=r
            ),
            [1.0, 0.02, 0.1],  # Initial kappa_r, theta_r, sigma_r hypothesis
            xtol=0.00001, ftol=0.00001, maxiter=300, maxfun=500,
            full_output= False, retall=False, disp=False
        )

        return CirDynamicsParameters.from_opt_result(opt)  # type: ignore


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
    """
    Example of CIR (1985) calibration
    """
    # Euribor Market data
    euribor_df: pd.DataFrame = pd.read_csv(
        get_path_from_package("erdqlib@src/ft/data/euribor_20140930.csv")
    )

    maturities: np.ndarray = euribor_df[OptionDataColumn.MATURITY].values  # Maturities in years with 30/360 convention
    rates: np.ndarray = euribor_df[OptionDataColumn.RATE].values  # Euribor rates in rate unit

    # Capitalization factors and Zero-rates
    r0: float = float(rates[0])
    zcb_rates: np.ndarray = annualized_continuous_rate(
        cap_factor=capitalization_factor(r_year=rates, t_year=maturities),
        t_year=maturities
    ) # Euribor is IR product which is a single cash flow, hence is a zero-coupon bond where YTM = spot-rate

    # Interpolation and Forward rates via Cubic spline
    bspline: Tuple[np.ndarray, np.ndarray, int] = splrep(maturities, zcb_rates, k=3)  # type: ignore
    maturities_ladder: np.ndarray = np.linspace(0.0, 1.0, 24)

    # Forward rate given a curve (of interpolated rates and their first derivatives)
    interpolated_rates: np.ndarray = splev(maturities_ladder, bspline, der=0)  # Interpolated rates
    first_derivatives: np.ndarray = splev(maturities_ladder, bspline, der=1)  # First derivative of spline
    zcb_forward_rates: np.ndarray = interpolated_rates + first_derivatives * maturities_ladder

    # Calibration of CIR parameters
    params_cir: CirDynamicsParameters = CirCalibrator.calibrate(
        r0=r0,
        curve_forward_rates=zcb_forward_rates,
        maturities_ladder=maturities_ladder
    )  # [0.06190266 0.23359687 0.14892987]
    LOGGER.info(params_cir)

    if not skip_plot:
        plot_interpolated_curve(
            maturities=maturities,
            zcb_rates=zcb_rates,
            maturities_ladder=maturities_ladder,
            interpolated_rates=interpolated_rates,
            first_derivatives=first_derivatives,
            zcb_forward_rates=zcb_forward_rates
        )
        plot_calibrated_cir(
            model_params=params_cir.get_value_arr(),
            maturities_ladder=maturities_ladder,
            market_forward_rates=zcb_forward_rates,
            r0=r0
        )


if __name__ == "__main__":
    ex_calibration(skip_plot=False)  # Example of CIR calibration