from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brute, fmin

from erdqlib.src.common.option import OptionDataColumn, OptionType
from erdqlib.src.common.option import OptionSide
from erdqlib.src.ft.calibrator import FtiCalibrator
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.src.mc.heston import HestonDynamicsParameters, HestonSearchGridType
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)


MIN_MSE: float = 500.0

class HestonFtiCalibrator(FtiCalibrator):

    @staticmethod
    def calculate_characteristic(
            u: complex, T: float, r: float,
            kappa_v: float, theta_v: float, sigma_v: float, rho: float, v0: float
    ) -> complex:
        r"""
        Heston (1993) characteristic function for Lewis (2001) Fourier pricing.

        φ^H(u, T) = exp( H₁(u,T) + H₂(u,T) · v₀ )

        where

          H₁(u,T) = i·r·u·T
                   + (c₁ / σ_v²)·[ (κ_v − ρ·σ_v·i·u + c₂)·T
                                   − 2·ln( (1 − c₃·e^{c₂·T}) / (1 − c₃) ) ]

          H₂(u,T) = (κ_v − ρ·σ_v·i·u + c₂) / σ_v²
                   · [ (1 − e^{c₂·T}) / (1 − c₃·e^{c₂·T}) ]

          c₁ = κ_v · θ_v
          c₂ = − sqrt[ (ρ·σ_v·i·u − κ_v)² − σ_v²·(−i·u − u²) ]
          c₃ = (κ_v − ρ·σ_v·i·u + c₂)
               / (κ_v − ρ·σ_v·i·u − c₂)
        """
        # constants
        c1: float = kappa_v * theta_v
        c2: complex = -np.sqrt(
            (rho * sigma_v * u * 1j - kappa_v) ** 2
            - sigma_v ** 2 * (-u * 1j - u ** 2)
        )
        c3: complex = (kappa_v - rho * sigma_v * u * 1j + c2) / (
                kappa_v - rho * sigma_v * u * 1j - c2
        )

        # H1 and H2
        H1: complex = (
                1j * r * u * T
                + (c1 / sigma_v ** 2) * (
                        (kappa_v - rho * sigma_v * u * 1j + c2) * T
                        - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))
                )
        )
        H2: complex = (
                (kappa_v - rho * sigma_v * u * 1j + c2)
                / sigma_v ** 2
                * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T)))
        )

        return np.exp(H1 + H2 * v0)

    @staticmethod
    def calculate_integral_characteristic(
            u: float,
            S0: float, K: float, T: float, r: float,
            kappa_v: float, theta_v: float, sigma_v: float, rho: float, v0: float
    ) -> float:
        r"""
        Integrand for Lewis (2001) call pricing under Heston ’93 model:

          C₀ = S₀ − (√(S₀·K)·e^{−r·T} / π)
               ∫₀^∞ Re[ e^{i·z·ln(S₀/K)} · φ^H(z − i/2, T) ]
                    · dz / (z² + 1/4)

        This returns the real part of
          e^{i·u·ln(S₀/K)} · φ^H(u − i/2, T)
        divided by (u² + 1/4).
        """
        psi = HestonFtiCalibrator.calculate_characteristic(u - 0.5j, T, r, kappa_v, theta_v, sigma_v, rho, v0)
        return (np.exp(1j * u * np.log(S0 / K)) * psi).real / (u ** 2 + 0.25)

    @staticmethod
    def calculate_option_price(
            S0: float, K: float, T: float, r: float,
            kappa_v: float, theta_v: float, sigma_v: float, rho: float, v0: float,
            side: OptionSide, type: OptionType = OptionType.EUROPEAN
    ):
        if type is not OptionType.EUROPEAN:
            raise NotImplementedError()

        """Valuation of European call option in H93 model via Lewis (2001)

        Parameter definition:
        ==========
        S0: float
            initial stock/index level
        K: float
            strike price
        T: float
            time-to-maturity (for t=0)
        r: float
            constant risk-free short rate
        kappa_v: float
            mean-reversion factor
        theta_v: float
            long-run mean of variance
        sigma_v: float
            volatility of variance
        rho: float
            correlation between variance and stock/index level
        v0: float
            initial level of variance
        Returns
        =======
        call_value: float
            present value of European call option
        """
        int_value = quad(
            lambda u: HestonFtiCalibrator.calculate_integral_characteristic(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0),
            0,
            np.inf,
            limit=250,
        )[0]
        call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value)
        if side is OptionSide.CALL:
            return call_value
        elif side is OptionSide.PUT:
            return call_value - S0 + K * np.exp(-r * T)
        raise ValueError(f"Invalid side: {side}")

    @staticmethod
    def calculate_error(
            p0: np.ndarray, df_options, print_iter: List[int],
            min_MSE_record: List[float], s0: float, side: OptionSide,
    ) -> float:
        """Error function for parameter calibration via
        Lewis (2001) Fourier approach for Heston (1993).
        Parameters
        ==========
        p0: list[float]
            model parameters (kappa_v, theta_v, sigma_v, rho, v0)
        df_options: pd.DataFrame
            DataFrame with market option quotes
        print_iter: List[int]
            List to keep track of iterations
        min_MSE_record: List[float]
            List to keep track of minimum mean squared error
        s0: float
            initial stock/index level
        side: OptionSide
            Option side (CALL or PUT)
        Returns
        =======
        MSE: float
            mean squared error
        """
        kappa_v, theta_v, sigma_v, rho, v0 = p0

        if HestonDynamicsParameters.do_parameters_offbound(kappa_v, theta_v, sigma_v, rho, v0):
            return MIN_MSE

        se = []
        for row, option in df_options.iterrows():
            model_value = HestonFtiCalibrator.calculate_option_price(
                S0=s0,
                K=option[OptionDataColumn.STRIKE], T=option[OptionDataColumn.TENOR], r=option[OptionDataColumn.RATE],
                kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho=rho, v0=v0,
                side=side
            )
            se.append((model_value - option[side.name]) ** 2)
        MSE = sum(se) / len(se)
        min_MSE_record[0] = min(min_MSE_record[0], MSE)
        if print_iter[0] % 25 == 0:
            LOGGER.info(f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in p0)}] | {MSE:7.3f} | {min_MSE_record[0]:7.3f}")
        print_iter[0] += 1
        return MSE

    @staticmethod
    def calibrate(
            df_options: pd.DataFrame,
            S0: float,
            r: Optional[float],
            side: OptionSide,
            search_grid: HestonSearchGridType
    ) -> HestonDynamicsParameters:
        """Calibrates Heston (1993) stochastic volatility model to market quotes."""
        print_iter = [0]
        min_MSE = [MIN_MSE]

        LOGGER.info("Brute-force begins")
        p0 = brute(
            lambda params, data=df_options, s0=S0, option_side=side: HestonFtiCalibrator.calculate_error(
                params, df_options=data, print_iter=print_iter, min_MSE_record=min_MSE, s0=s0, side=option_side
            ),
            search_grid,
            finish=None,
        )

        # Second run with local, convex minimization
        # (we dig deeper where promising results)
        LOGGER.info("Fmin begins")
        p_opt: np.array = fmin(
            lambda params, data=df_options, s0=S0, option_side=side: HestonFtiCalibrator.calculate_error(
                params, df_options=data, print_iter=print_iter, min_MSE_record=min_MSE, s0=s0, side=option_side
            ),
            p0, xtol=1e-6, ftol=1e-6, maxiter=750, maxfun=900,
            full_output=False, retall=False, disp=True
        )

        return HestonDynamicsParameters.from_calibration_output(opt_arr=p_opt, S0=S0, r=r).get_bounded_parameters()


def ex_pricing():
    # Option Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02

    # Heston(1993) Parameters
    kappa_v = 1.5
    theta_v = 0.02
    sigma_v = 0.15
    rho = 0.1
    v0 = 0.01

    LOGGER.info(
        "Heston (1993) Call Option Value:   $%10.4f "
        % HestonFtiCalibrator.calculate_option_price(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, side=OptionSide.CALL)
    )


def plot_Heston(
    opt_params: HestonDynamicsParameters, df_options: pd.DataFrame, S0: float, side: OptionSide
):
    """Plot market and model prices for each maturity and OptionSide."""
    df_options_plt = df_options.copy()
    df_options_plt[OptionDataColumn.MODEL] = 0.0
    for row, option in df_options_plt.iterrows():
        df_options_plt.loc[row, OptionDataColumn.MODEL] = HestonFtiCalibrator.calculate_option_price(
            side=side,
            S0=S0, K=option[OptionDataColumn.STRIKE], T=option[OptionDataColumn.TENOR], r=option[OptionDataColumn.RATE],
            kappa_v=opt_params.kappa_heston, theta_v=opt_params.theta_heston, sigma_v=opt_params.sigma_heston, rho=opt_params.rho_heston, v0=opt_params.v0_heston
        )

    for maturity, df_options_per_maturity in df_options_plt.groupby(OptionDataColumn.DAYSTOMATURITY):
        df_options_per_maturity[[OptionDataColumn.STRIKE] + [side.name, OptionDataColumn.MODEL]].plot(
            x=OptionDataColumn.STRIKE, y=[side.name, OptionDataColumn.MODEL],
            style=["b-", "ro"], title=f"Maturity {maturity}D on {side.name}"
        )
        plt.axvline(x=S0, color='k', linestyle='--', label="S0")
        plt.ylabel("Option Value")
    plt.show()


def ex_calibration(
        data_path: str,
        side: OptionSide,
        skip_plot: bool = False
):
    # Market Data from www.eurexchange.com
    # as of September 30, 2014
    S0 = 3225.93  # EURO STOXX 50 level September 30, 2014
    r = 0.02

    df_options: pd.DataFrame = load_option_data(
        path_str=data_path,
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r # constant short rate
    )

    params_heston: HestonDynamicsParameters = HestonFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        search_grid=HestonDynamicsParameters.get_default_search_grid()
    )
    LOGGER.info(f"Heston calib: {params_heston}")

    if not skip_plot:
        plot_Bates(params_heston, df_options, S0, side)


if __name__ == "__main__":
    ex_pricing()
    ex_calibration(
        data_path=get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv"),
        side=OptionSide.CALL,
        skip_plot=False
    )
