from typing import Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fmin

from erdqlib.src.common.option import OptionSide, OptionDataColumn
from erdqlib.src.ft.calibrator import FtiCalibrator, FtMethod, MIN_MSE
from erdqlib.src.mc.gbm import GbmDynamicsParameters
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)


class GbmFtiCalibrator(FtiCalibrator):
    @staticmethod
    def calculate_characteristic(
            u: complex | np.ndarray, x0:float, T: float, r: float, sigma: float
    ) -> complex | np.ndarray:
        """
        Computes general Black-Scholes model characteristic function
        to be used in Fourier pricing methods like Lewis (2001) and Carr-Madan (1999):
        exp( i * u * log(S_0) + i * u*(r - 0.5 * sigma**2)* T - 0.5 * u^2 * sigma^2 * T )

        :param x0: log(S_0)
        :param u: u
        """
        return np.exp(((x0 / T + r - 0.5 * sigma ** 2) * 1j * u - 0.5 * sigma ** 2 * u ** 2) * T)

    @staticmethod
    def calculate_integral_characteristic(
            u: float, S0: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """
        Calculate the integral characteristic for the Geometric Brownian Motion (GBM) model.
        For GBM, this is not applicable as GBM does not use characteristic functions in this context.
        """
        cf_val: complex = GbmFtiCalibrator.calculate_characteristic(
            u=u - 0.5j, x0=np.log(S0/S0), T=T, r=r, sigma=sigma
        )
        phase = np.exp(1j * u * np.log(S0 / K))
        return (phase * cf_val).real / (u ** 2 + 0.25)

    @staticmethod
    def calculate_option_price_lewis(
            S0: float, K: float, T: float, r: float, sigma: float, side: OptionSide
    ) -> float:
        """
        Calculate the option price using the Lewis formula for the Geometric Brownian Motion (GBM) model.
        This is a simplified version and does not use characteristic functions.
        """
        int_value = quad(
            lambda u: GbmFtiCalibrator.calculate_integral_characteristic(
                u=u, S0=S0, K=K, T=T, r=r, sigma=sigma
            ),
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
    def calculate_option_price_carrmadan(
            S0: float, K: float, T: float, r: float, sigma: float, side: OptionSide
    ) -> float:
        k = np.log(K / S0)
        x0 = np.log(S0 / S0)
        g = 1  # Factor to increase accuracy
        N = g * 4096
        eps = (g * 150) ** -1
        eta = 2 * np.pi / (N * eps)
        b = 0.5 * N * eps - k
        u = np.arange(1, N + 1, 1)
        vo = eta * (u - 1)

        # Modifications to ensure integrability
        if S0 >= 0.95 * K:  # ITM Case
            alpha = 1.5
            v = vo - (alpha + 1) * 1j
            modcharFunc = np.exp(-r * T) * (
                    GbmFtiCalibrator.calculate_characteristic(u=v, x0=x0, T=T, r=r, sigma=sigma)
                    / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)
            )

        else:
            alpha = 1.1
            v = (vo - 1j * alpha) - 1j
            modcharFunc1 = np.exp(-r * T) * (
                    1 / (1 + 1j * (vo - 1j * alpha))
                    - np.exp(r * T) / (1j * (vo - 1j * alpha))
                    - GbmFtiCalibrator.calculate_characteristic(u=v, x0=x0, T=T, r=r, sigma=sigma)
                    / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha))
            )

            v = (vo + 1j * alpha) - 1j
            modcharFunc2 = np.exp(-r * T) * (
                    1 / (1 + 1j * (vo + 1j * alpha))
                    - np.exp(r * T) / (1j * (vo + 1j * alpha))
                    - GbmFtiCalibrator.calculate_characteristic(u=v, x0=x0, T=T, r=r, sigma=sigma)
                    / ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha))
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

        pos: int = int((k + b) / eps)
        call_value: float = CallValueM[pos] * S0
        if side is OptionSide.CALL:
            return call_value
        elif side is OptionSide.PUT:
            return call_value - S0 + K * np.exp(-r * T)
        raise ValueError(f"Invalid side: {side}")

    @staticmethod
    def calculate_option_price(
            S0: float, K: float, T: float, r: float, sigma: float,
            side: OptionSide, ft_method: FtMethod = FtMethod.LEWIS
    ):
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
        sigma: float
            volatility of variance
        Returns
        =======
        call_value: float
            present value of European call option
        """
        if ft_method is FtMethod.LEWIS:
            return GbmFtiCalibrator.calculate_option_price_lewis(
                S0=S0, K=K, T=T, r=r, sigma=sigma, side=side
            )
        elif ft_method is FtMethod.CARRMADAN:
            return GbmFtiCalibrator.calculate_option_price_carrmadan(
                S0=S0, K=K, T=T, r=r, sigma=sigma, side=side
            )
        raise ValueError(f"Invalid FtMethod method: {ft_method}")

    @staticmethod
    def calculate_error(
            gbm_params: np.array, df_options: pd.DataFrame, s0: float, side: OptionSide,
            print_iter: Optional[List[float]] = None, min_MSE: Optional[List[float]] = None,
            ft_method: FtMethod = FtMethod.LEWIS
    ) -> float:
        sigma = gbm_params
        if GbmDynamicsParameters.do_parameters_offbound(sigma=sigma):
            return MIN_MSE
        se = []
        for _, option in df_options.iterrows():
            model_value: float = GbmFtiCalibrator.calculate_option_price(
                S0=s0,
                K=option[OptionDataColumn.STRIKE], T=option[OptionDataColumn.TENOR], r=option[OptionDataColumn.RATE],
                sigma=sigma,
                side=side, ft_method=ft_method
            )
            se.append((model_value - option[side.name]) ** 2)
        mse: float = sum(se) / len(se)
        if min_MSE is not None:
            min_MSE[0] = min(min_MSE[0], mse)
        if print_iter is not None:
            if print_iter[0] % 25 == 0:
                LOGGER.info(
                    f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in gbm_params)}] | {mse:7.3f} | {min_MSE[0]:7.3f}"
                )
            print_iter[0] += 1
        return mse

    @staticmethod
    def calibrate(
            df_options: pd.DataFrame, S0: float, r: float, side: OptionSide,
            ft_method: FtMethod = FtMethod.LEWIS, **kwargs
    ) -> GbmDynamicsParameters:
        """
        Calibrate the Geometric Brownian Motion (GBM) model parameters using Fourier Transform methods.
        """
        print_iter = [0]
        min_MSE = [MIN_MSE]

        LOGGER.info("Fmin begins")
        p_opt: np.array = fmin(
            lambda p, data=df_options, s0=S0, option_side=side: GbmFtiCalibrator.calculate_error(
                p, df_options=data, s0=s0, side=option_side, print_iter=print_iter, min_MSE=min_MSE,
                ft_method=ft_method
            ),
            np.array([0.1], dtype=float), # Initial guess for sigma
            xtol=1e-6, ftol=1e-6, maxiter=500, maxfun=700
        )
        return GbmDynamicsParameters.from_calibration_output(opt_arr=p_opt, s0=S0, r=r)


def plot_Jump(
        opt_params: GbmDynamicsParameters, df_options: pd.DataFrame, S0: float,
        side: OptionSide, ft_method: FtMethod = FtMethod.LEWIS
):
    """Plot market and model prices for each maturity and OptionSide."""
    sigma, = opt_params.get_values()
    df_options_plt = df_options.copy()
    df_options_plt[OptionDataColumn.MODEL] = 0.0
    for row, option in df_options_plt.iterrows():
        df_options_plt.loc[row, OptionDataColumn.MODEL] = GbmFtiCalibrator.calculate_option_price(
            S0=S0,
            K=option[OptionDataColumn.STRIKE], T=option[OptionDataColumn.TENOR], r=option[OptionDataColumn.RATE],
            sigma=sigma,
            side=side, ft_method=ft_method
        )

    for maturity, df_options_per_maturity in df_options_plt.groupby(OptionDataColumn.DAYSTOMATURITY):
        df_options_per_maturity[[side.name, OptionDataColumn.MODEL]].plot(
            style=["b-", "ro"], title=f"Maturity {maturity} {side.name}"
        )
        plt.ylabel("Option Value")
    plt.show()


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

    params_jump: GbmDynamicsParameters = GbmFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, side=side, r=r,
        ft_method=ft_method
    )
    LOGGER.info(f"Gbm calib: {params_jump}")
    if not skip_plot:
        plot_Jump(params_jump, df_options=df_options, S0=S0, side=side, ft_method=ft_method)


if __name__ == "__main__":
    ex_calibration(
        data_path=get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv"),
        side=OptionSide.CALL
    )