from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brute, fmin

from erdqlib.src.common.option import OptionDataColumn
from erdqlib.src.common.option import OptionSide
from erdqlib.src.ft.calibrator import FtiCalibrator, FtMethod, MIN_RMSE, MIN_MSE
from erdqlib.src.mc.jump import (
    JumpDynamicsParameters, JumpSearchGridType, JumpOnlySearchGridType, JumpOnlyDynamicsParameters
)
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)


class JumpFtiCalibrator(FtiCalibrator):
    @staticmethod
    def calculate_characteristic(
            u: complex | np.ndarray, T: float, r: float,
            lambd: float, mu: float, delta: float,
            sigma: Optional[float] = None, exclude_diffusion: bool = False
    ) -> complex | np.ndarray:
        """
        Characteristic function for the Merton ’76 jump-diffusion model.

        φ₀^{M76}(u, T) = exp( [i·u·ω
                              − ½·u²·σ²
                              + λ·(exp(i·u·μ − ½·u²·δ²) − 1)
                             ] · T )

        where
          ω = r − ½·σ² − λ·(exp(μ + ½·δ²) − 1)
        """
        char_func_value: complex
        omega: np.float64
        if exclude_diffusion:
            omega = -lambd * (np.exp(mu + 0.5 * delta ** 2) - 1)
            char_func_value = np.exp(
                (
                        1j * u * omega
                        + lambd * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1)
                ) * T
            )
        else:
            assert sigma is not None, "Delta must be provided if diffusion is included."
            omega = r - 0.5 * sigma ** 2 - lambd * (np.exp(mu + 0.5 * delta ** 2) - 1)
            char_func_value = np.exp(
                (
                        1j * u * omega
                        - 0.5 * u ** 2 * sigma ** 2
                        + lambd * (np.exp(1j * u * mu - 0.5 * u ** 2 * delta ** 2) - 1)
                ) * T
            )
        return char_func_value

    @staticmethod
    def calculate_integral_characteristic(
            u: float,
            S0: float, K: float, T: float, r: float,
            sigma: float, lambd: float, mu: float, delta: float
    ) -> float:
        r"""
        Integrand for the Lewis (2001) FFT pricing under Merton ’76 model.
    
        C₀ = S₀ − (√(S₀·K)·e^{−r·T} / π)
               ∫₀^∞ Re[ e^{i·z·ln(S₀/K)} · φ(z − i/2, T) ] · dz / (z² + 1/4)
    
        This function returns
           Re[ e^{i·u·ln(S₀/K)} · φ(u − i/2, T) ] / (u² + 1/4)
        """
        char: complex = JumpFtiCalibrator.calculate_characteristic(
            u=u - 0.5j, T=T, r=r, lambd=lambd, mu=mu, delta=delta, sigma=sigma
        )
        return (np.exp(1j * u * np.log(S0 / K)) * char).real / (u ** 2 + 0.25)

    @staticmethod
    def calculate_option_price_carrmadan(
            S0: float, K: float, T: float, r: float,
            lambd: float, mu: float, delta: float, sigma: float,
            side: OptionSide
    ) -> float:
        """
        Call option price in Bates (1996) under FFT
        """
        k: float = np.log(K / S0)
        g: int = 1  # Factor to increase accuracy
        N: int = g * 4096
        eps: float = (g * 150) ** -1
        eta: float = 2 * np.pi / (N * eps)
        b: float = 0.5 * N * eps - k
        u: np.ndarray = np.arange(1, N + 1, 1)
        vo: np.ndarray = eta * (u - 1)

        # Modifications to ensure integrability
        if S0 >= 0.95 * K:  # ITM Case
            alpha: float = 1.5
            v: np.ndarray = vo - (alpha + 1) * 1j
            modcharFunc: np.ndarray = np.exp(-r * T) * (
                    JumpFtiCalibrator.calculate_characteristic(
                        u=v, T=T, r=r,
                        lambd=lambd, mu=mu, delta=delta, sigma=sigma
                    ) / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)
            )

        else:
            alpha = 1.1
            v = (vo - 1j * alpha) - 1j
            modcharFunc1 = np.exp(-r * T) * (
                1 / (1 + 1j * (vo - 1j * alpha))
                - np.exp(r * T) / (1j * (vo - 1j * alpha))
                - JumpFtiCalibrator.calculate_characteristic(
                    u=v, T=T, r=r,
                    lambd=lambd, mu=mu, delta=delta, sigma=sigma
                ) / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha))
            )

            v = (vo + 1j * alpha) - 1j
            modcharFunc2 = np.exp(-r * T) * (
                1 / (1 + 1j * (vo + 1j * alpha))
                - np.exp(r * T) / (1j * (vo + 1j * alpha))
                - JumpFtiCalibrator.calculate_characteristic(
                    u=v, T=T, r=r,
                    lambd=lambd, mu=mu, delta=delta, sigma=sigma
                )/ ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha))
            )

        # Numerical FFT Routine
        delt = np.zeros(N)
        delt[0] = 1
        j = np.arange(1, N + 1, 1)
        SimpsonW = (3 + (-1) ** j - delt) / 3
        if S0 >= 0.95 * K:
            FFTFunc = np.exp(1j * b * vo) * modcharFunc * eta * SimpsonW
            payoff = (np.fft.fft(FFTFunc)).real
            CallValueM = np.exp(-alpha * k) / np.pi * payoff
        else:
            FFTFunc = (
                    np.exp(1j * b * vo) * (modcharFunc1 - modcharFunc2) * 0.5 * eta * SimpsonW
            )
            payoff = (np.fft.fft(FFTFunc)).real
            CallValueM = payoff / (np.sinh(alpha * k) * np.pi)

        pos = int((k + b) / eps)
        call_value = CallValueM[pos] * S0
        if side is OptionSide.CALL:
            return call_value
        elif side is OptionSide.PUT:
            return call_value - S0 + K * np.exp(-r * T)
        raise ValueError(f"Invalid side: {side}")

    @staticmethod
    def calculate_option_price(
            S0: float, K: float, T: float, r: float,
            sigma: float, lambd: float, mu: float, delta: float,
            side: OptionSide, ft_method: FtMethod = FtMethod.LEWIS
    ) -> float:
        if ft_method is FtMethod.LEWIS:
            return FtiCalibrator.calculate_option_price_lewis(
                x0=S0, K=K, T=T, r=r,
                characteristic_integral=lambda u: JumpFtiCalibrator.calculate_integral_characteristic(
                    u=u, S0=S0, K=K, T=T, r=r,
                    lambd=lambd, mu=mu, delta=delta, sigma=sigma,
                ),
                side=side
            )
        elif ft_method is FtMethod.CARRMADAN:
            return JumpFtiCalibrator.calculate_option_price_carrmadan(
                S0=S0, K=K, T=T, r=r, lambd=lambd, mu=mu, delta=delta, sigma=sigma, side=side
            )
        raise ValueError(f"Invalid FtMethod method: {ft_method}")

    @staticmethod
    def calculate_error(
            jump_params: np.array, df_options: pd.DataFrame, s0: float, side: OptionSide,
            print_iter: Optional[List[float]] = None, min_MSE: Optional[List[float]] = None,
            ft_method: FtMethod = FtMethod.LEWIS
    ) -> float:
        """
        Error function for parameter calibration in Merton'76 model.
        Now supports both CALL and PUT via OptionSide.
        """
        lambd, mu, delta, sigma = jump_params
        if JumpDynamicsParameters.do_parameters_offbound(sigma=sigma, lambd=lambd, mu=mu, delta=delta):
            return MIN_RMSE
        se = []
        for _, option in df_options.iterrows():
            model_value: float = JumpFtiCalibrator.calculate_option_price(
                S0=s0,
                K=option[OptionDataColumn.STRIKE], T=option[OptionDataColumn.TENOR], r=option[OptionDataColumn.RATE],
                sigma=sigma, lambd=lambd, mu=mu, delta=delta,
                side=side, ft_method=ft_method
            )
            se.append((model_value - option[side.name]) ** 2)
        rmse: float = sum(se) / len(se)
        if min_MSE is not None:
            min_MSE[0] = min(min_MSE[0], rmse)
        if print_iter is not None:
            if print_iter[0] % 50 == 0:
                LOGGER.info(
                    f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in jump_params)}] | {rmse:7.3f} | {min_MSE[0]:7.3f}"
                )
            print_iter[0] += 1
        return rmse

    @staticmethod
    def calibrate(
            df_options: pd.DataFrame,
            S0: float,
            r: Optional[float],
            side: OptionSide,
            search_grid: JumpOnlySearchGridType | JumpSearchGridType,
            ft_method: FtMethod = FtMethod.LEWIS
    ) -> JumpOnlyDynamicsParameters | JumpDynamicsParameters:
        """Calibrates Merton (1976) model to market quotes for given OptionSide."""
        print_iter = [0]
        min_MSE = [MIN_MSE]

        LOGGER.info("Brute-force begins")
        p0 = brute(
            lambda p, data=df_options, s0=S0, option_side=side: JumpFtiCalibrator.calculate_error(
                p, df_options=data, s0=s0, side=option_side, print_iter=print_iter, min_MSE=min_MSE, ft_method=ft_method
            ),
            search_grid,
            finish=None,
        )

        LOGGER.info("Fmin begins")
        p_opt: np.array = fmin(
            lambda p, data=df_options, s0=S0, option_side=side: JumpFtiCalibrator.calculate_error(
                p, df_options=data, s0=s0, side=option_side, print_iter=print_iter, min_MSE=min_MSE, ft_method=ft_method
            ),
            p0, xtol=1e-4, ftol=1e-4, maxiter=550, maxfun=1050
        )
        return JumpDynamicsParameters.from_calibration_output(opt_arr=p_opt, s0=S0, r=r)


def plot_Jump(
        opt_params: JumpDynamicsParameters, df_options: pd.DataFrame, S0: float,
        side: OptionSide, ft_method: FtMethod = FtMethod.LEWIS
):
    """Plot market and model prices for each maturity and OptionSide."""
    lambd, mu, delta, sigma = opt_params.get_values()
    df_options_plt = df_options.copy()
    df_options_plt[OptionDataColumn.MODEL] = 0.0
    for row, option in df_options_plt.iterrows():
        df_options_plt.loc[row, OptionDataColumn.MODEL] = JumpFtiCalibrator.calculate_option_price(
            S0=S0,
            K=option[OptionDataColumn.STRIKE], T=option[OptionDataColumn.TENOR], r=option[OptionDataColumn.RATE],
            sigma=sigma, lambd=lambd, mu=mu, delta=delta,
            side=side, ft_method=ft_method
        )

    for maturity, df_options_per_maturity in df_options_plt.groupby(OptionDataColumn.DAYSTOMATURITY):
        df_options_per_maturity[[side.name, OptionDataColumn.MODEL]].plot(
            style=["b-", "ro"], title=f"Maturity {maturity} {side.name}"
        )
        plt.ylabel("Option Value")
    plt.show()


def ex_pricing():
    """Example: price a call and put option under Merton (1976) model."""
    ft_method: FtMethod = FtMethod.CARRMADAN  # or FtMethod.LEWIS

    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.4
    lambd = 1
    mu = -0.2
    delta = 0.1
    for side in [OptionSide.CALL, OptionSide.PUT]:
        value = JumpFtiCalibrator.calculate_option_price(
            S0=S0, K=K, T=T, r=r, sigma=sigma, lambd=lambd, mu=mu, delta=delta,
            side=side, ft_method=ft_method
        )
        LOGGER.info(f"Value of the {side.name} option under Merton (1976) is:  ${value}")


def ex_calibration(
        data_path: str,
        side: OptionSide,
        skip_plot: bool = False
):
    """Example: calibrate Merton (1976) model to market data and plot results for given OptionSide."""
    S0 = 3225.93  # EURO STOXX 50 level September 30, 2014
    r = 0.005
    ft_method: FtMethod = FtMethod.LEWIS

    df_options: pd.DataFrame = load_option_data(
        path_str=data_path,
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r,  # constant short rate
        days_to_maturity_target=17  # 17 days to maturity
    )

    params_jump: JumpDynamicsParameters = JumpFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, side=side, r=r,
        search_grid=JumpDynamicsParameters.get_default_search_grid(),
        ft_method=ft_method
    )
    LOGGER.info(f"Jump calib: {params_jump}")
    if not skip_plot:
        plot_Jump(params_jump, df_options=df_options, S0=S0, side=side, ft_method=ft_method)


if __name__ == "__main__":
    ex_pricing()
    ex_calibration(
        data_path=get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv"),
        side=OptionSide.CALL
    )
