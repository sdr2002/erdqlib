from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brute, fmin

from erdqlib.src.common.option import OptionSide, OptionDataColumn
from erdqlib.src.ft.calibrator import FtiCalibrator, plot_calibration_result, FtMethod, MIN_MSE
from erdqlib.src.ft.heston import HestonFtiCalibrator
from erdqlib.src.ft.jump import JumpFtiCalibrator
from erdqlib.src.mc.bates import BatesDynamicsParameters
from erdqlib.src.mc.heston import HestonDynamicsParameters, HestonSearchGridType
from erdqlib.src.mc.jump import JumpOnlyDynamicsParameters, JumpOnlySearchGridType
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)


class BatesFtiCalibrator(FtiCalibrator):
    @staticmethod
    def calculate_characteristic(
            u: complex | np.ndarray, T: float, r: float,
            kappa_v: float, theta_v: float, sigma_v: float, rho: float, v0: float,
            lambd: float, mu: float, delta: float
    ) -> complex | np.ndarray:
        """Bates (1996) characteristic function."""
        H93: complex = HestonFtiCalibrator.calculate_characteristic(u, T, r, kappa_v, theta_v, sigma_v, rho, v0)
        M76J: complex = JumpFtiCalibrator.calculate_characteristic(
            u=u, T=T, r=r, lambd=lambd, mu=mu, delta=delta, sigma=None, exclude_diffusion=True
        )
        return H93 * M76J

    @staticmethod
    def calculate_integral_characteristic(
            u: float,
            S0: float, K: float, T: float, r: float,
            kappa_v: float, theta_v: float, sigma_v: float, rho: float, v0: float,
            lambd: float, mu: float, delta: float
    ) -> float:
        """Lewis (2001) integral for Bates (1996) characteristic function."""
        char_func_value = BatesFtiCalibrator.calculate_characteristic(
            u - 0.5j, T, r, kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta
        )
        return (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real / (u**2 + 0.25)

    @staticmethod
    def calculate_option_price_lewis(
            x0: float, K: float, T: float, r: float,
            kappa_heston: float, theta_heston: float, sigma_heston: float, rho_heston: float, v0_heston: float,
            lambd_merton: float, mu_merton: float, delta_merton: float,
            side: OptionSide
    ) -> float:
        """European option value in Bates (1996) model via Lewis (2001)."""
        int_value = quad(
            lambda u: BatesFtiCalibrator.calculate_integral_characteristic(
                u=u, S0=x0, K=K, T=T, r=r,
                kappa_v=kappa_heston, theta_v=theta_heston, sigma_v=sigma_heston, rho=rho_heston, v0=v0_heston,
                lambd=lambd_merton, mu=mu_merton, delta=delta_merton
            ),
            0,
            np.inf,
            limit=250,
        )[0]
        call_value = max(0, x0 - np.exp(-r * T) * np.sqrt(x0 * K) / np.pi * int_value)
        if side is OptionSide.CALL:
            return call_value
        elif side is OptionSide.PUT:
            return call_value - x0 + K * np.exp(-r * T)
        raise ValueError(f"Invalid side: {side}")

    @staticmethod
    def calculate_option_price_carrmadan(
            x0: float, K: float, T: float, r: float,
            kappa_heston: float, theta_heston: float, sigma_heston: float, rho_heston: float, v0_heston: float,
            lambd_merton: float, mu_merton: float, delta_merton: float,
            side: OptionSide
    ) -> float:
        """
        Call option price in Bates (1996) under FFT
        """
        k: float = np.log(K / x0)
        g: int = 1  # Factor to increase accuracy
        N: int = g * 4096
        eps: float = (g * 150) ** -1
        eta: float = 2 * np.pi / (N * eps)
        b: float = 0.5 * N * eps - k
        u: np.ndarray = np.arange(1, N + 1, 1)
        vo: np.ndarray = eta * (u - 1)

        # Modifications to ensure integrability
        if x0 >= 0.95 * K:  # ITM Case
            alpha = 1.5
            v = vo - (alpha + 1) * 1j
            modcharFunc = np.exp(-r * T) * (
                BatesFtiCalibrator.calculate_characteristic(v, T, r, kappa_heston, theta_heston, sigma_heston, rho_heston, v0_heston, lambd_merton, mu_merton, delta_merton)
                / (alpha**2 + alpha - vo**2 + 1j * (2 * alpha + 1) * vo)
            )

        else:
            alpha = 1.1
            v = (vo - 1j * alpha) - 1j
            modcharFunc1 = np.exp(-r * T) * (
                1 / (1 + 1j * (vo - 1j * alpha))
                - np.exp(r * T) / (1j * (vo - 1j * alpha))
                - BatesFtiCalibrator.calculate_characteristic(
                    v, T, r, kappa_heston, theta_heston, sigma_heston, rho_heston, v0_heston, lambd_merton, mu_merton, delta_merton
                )
                / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha))
            )

            v = (vo + 1j * alpha) - 1j
            modcharFunc2 = np.exp(-r * T) * (
                1 / (1 + 1j * (vo + 1j * alpha))
                - np.exp(r * T) / (1j * (vo + 1j * alpha))
                - BatesFtiCalibrator.calculate_characteristic(
                    v, T, r, kappa_heston, theta_heston, sigma_heston, rho_heston, v0_heston, lambd_merton, mu_merton, delta_merton
                )
                / ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha))
            )

        # Numerical FFT Routine
        delt = np.zeros(N)
        delt[0] = 1
        j = np.arange(1, N + 1, 1)
        SimpsonW = (3 + (-1) ** j - delt) / 3
        if x0 >= 0.95 * K:
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
        call_value = CallValueM[pos] * x0
        if side is OptionSide.CALL:
            return call_value
        elif side is OptionSide.PUT:
            return call_value - x0 + K * np.exp(-r * T)
        raise ValueError(f"Invalid side: {side}")

    @staticmethod
    def calculate_option_price(
        S0, K, T, r, kappa_v, theta_v, sigma_v, rho_v, v0, lambd, mu, delta,
        side: OptionSide, ft_method: FtMethod = FtMethod.LEWIS
    ) -> float:
        if ft_method is FtMethod.LEWIS:
            return float(BatesFtiCalibrator.calculate_option_price_lewis(
                x0=S0, K=K, T=T, r=r,
                kappa_heston=kappa_v, theta_heston=theta_v, sigma_heston=sigma_v, rho_heston=rho_v, v0_heston=v0,
                lambd_merton=lambd, mu_merton=mu, delta_merton=delta, side=side
            ))
        elif ft_method is FtMethod.CARRMADAN:
            return float(BatesFtiCalibrator.calculate_option_price_carrmadan(
                x0=S0, K=K, T=T, r=r,
                kappa_heston=kappa_v, theta_heston=theta_v, sigma_heston=sigma_v, rho_heston=rho_v, v0_heston=v0,
                lambd_merton=lambd, mu_merton=mu, delta_merton=delta, side=side
            ))
        raise ValueError(f"Invalid FtMethod method: {ft_method}")

    @staticmethod
    def calculate_option_price_batch(
            df_options: pd.DataFrame, S0: float,
            kappa_v: float, theta_v: float, sigma_v: float, rho: float, v0: float,
            lambd: float, mu: float, delta: float,
            side: OptionSide, ft_method: FtMethod = FtMethod.LEWIS
    ) -> np.ndarray:
        """Calculates all model values given parameter vector p0."""
        values = []
        for _, option in df_options.iterrows():
            model_value = BatesFtiCalibrator.calculate_option_price(
                S0=S0,
                K=option[OptionDataColumn.STRIKE],
                T=option[OptionDataColumn.TENOR],
                r=option[OptionDataColumn.RATE],
                kappa_v=kappa_v,
                theta_v=theta_v,
                sigma_v=sigma_v,
                rho_v=rho,
                v0=v0,
                lambd=lambd,
                mu=mu,
                delta=delta,
                side=side,
                ft_method=ft_method
            )
            values.append(model_value)
        return np.array(values)

    @staticmethod
    def calculate_error_jump(
            task_params: np.ndarray, df_options: pd.DataFrame, s0: float,
            kappa_v: float, theta_v: float, sigma_v: float, rho: float, v0: float,
            side: OptionSide,
            ft_method: FtMethod = FtMethod.LEWIS,
            print_iter: Optional[List[int]] = None,
            min_MSE: Optional[List[float]] = None,
            initial_params: Optional[np.ndarray] = None
    ) -> float:
        """Error function for Bates (1996) model calibration."""
        lambd, mu, delta = task_params  # type: float, float, float
        if JumpOnlyDynamicsParameters.do_parameters_offbound(lambd=lambd, mu=mu, delta=delta):
            return MIN_MSE

        se = []
        for _, option in df_options.iterrows():
            model_value = BatesFtiCalibrator.calculate_option_price(
                S0=s0,
                K=option[OptionDataColumn.STRIKE],
                T=option[OptionDataColumn.TENOR],
                r=option[OptionDataColumn.RATE],
                kappa_v=kappa_v,
                theta_v=theta_v,
                sigma_v=sigma_v,
                rho_v=rho,
                v0=v0,
                lambd=lambd,
                mu=mu,
                delta=delta,
                side=side,
                ft_method=ft_method
            )
            se.append((model_value - option[side.name]) ** 2)
        MSE = sum(se) / len(se)
        if min_MSE is not None:
            min_MSE[0] = min(min_MSE[0], MSE)
        if print_iter is not None:
            if print_iter[0] % 25 == 0:
                LOGGER.info(f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in task_params)}] | {MSE:7.3g} | {min_MSE[0]:7.3g}")
            print_iter[0] += 1
        if initial_params is not None:
            penalty = np.sqrt(np.sum((task_params - initial_params) ** 2)) * 1
            return MSE + penalty
        return MSE

    @staticmethod
    def calibrate_jump(
        df_options: pd.DataFrame,
        S0: float,
        kappa_v: float,
        theta_v: float,
        sigma_v: float,
        rho: float,
        v0: float,
        side: OptionSide,
        jump_search_grid: JumpOnlySearchGridType,
        ft_method: FtMethod = FtMethod.LEWIS,
    ) -> JumpOnlyDynamicsParameters:
        """Calibrates jump component of Bates (1996) model to market prices."""
        print_iter = [0]
        min_MSE = [5000.0]
        opt1 = brute(
            lambda p: BatesFtiCalibrator.calculate_error_jump(
                p, df_options=df_options, s0=S0,
                kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho=rho, v0=v0,
                side=side, print_iter=print_iter, min_MSE=min_MSE, ft_method=ft_method
            ),
            jump_search_grid,
            finish=None,
        )
        opt2 = fmin(
            lambda p: BatesFtiCalibrator.calculate_error_jump(
                p, df_options=df_options, s0=S0,
                kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho=rho, v0=v0,
                side=side, print_iter=print_iter, min_MSE=min_MSE, ft_method=ft_method
            ),
            opt1,
            xtol=1e-7,
            ftol=1e-7,
            maxiter=550,
            maxfun=750,
        )
        return JumpOnlyDynamicsParameters.from_calibration_output(opt_arr=opt2, s0=S0)

    @staticmethod
    def calculate_error_full(
            p0: np.ndarray, options: pd.DataFrame, s0: float,
            print_iter: List[int], min_MSE: List[float], side: OptionSide,
            ft_method: FtMethod = FtMethod.LEWIS
    ) -> float:
        """Error function for full Bates (1996) model calibration."""
        kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta = p0
        # Parameter bounds
        if BatesDynamicsParameters.do_parameters_offbound(
                kappa_v=kappa_v,
                theta_v=theta_v,
                sigma_v=sigma_v,
                rho=rho,
                v0=v0,
                lambd=lambd,
                mu=mu,
                delta=delta,
        ):
            return 5000.0

        se = []
        for _, option in options.iterrows():
            model_value = BatesFtiCalibrator.calculate_option_price(
                S0=s0,
                K=option[OptionDataColumn.STRIKE],
                T=option[OptionDataColumn.TENOR],
                r=option[OptionDataColumn.RATE],
                kappa_v=kappa_v,
                theta_v=theta_v,
                sigma_v=sigma_v,
                rho_v=rho,
                v0=v0,
                lambd=lambd,
                mu=mu,
                delta=delta,
                side=side,
                ft_method=ft_method
            )
            se.append((model_value - option[side.name]) ** 2)
        MSE = sum(se) / len(se)
        min_MSE[0] = min(min_MSE[0], MSE)
        if print_iter[0] % 25 == 0:
            LOGGER.info(f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in p0)}] | {MSE:7.3g} | {min_MSE[0]:7.3g}")
        print_iter[0] += 1
        return MSE

    @staticmethod
    def calibrate_full(
            df_options: pd.DataFrame,
            S0: float,
            r:float,
            initial_params: BatesDynamicsParameters,
            side: OptionSide,
            ft_method: FtMethod = FtMethod.LEWIS
    ) -> BatesDynamicsParameters:
        """Calibrates all Bates (1996) model parameters to market prices."""
        print_iter = [0]
        min_MSE = [5000.0]

        LOGGER.info("fmin optimization begins")
        opt: np.ndarray = fmin(
            lambda p: BatesFtiCalibrator.calculate_error_full(
                p0=p, options=df_options, s0=S0,
                print_iter=print_iter, min_MSE=min_MSE, side=side, ft_method=ft_method
            ),
            np.array(initial_params.get_values()),
            xtol=1e-7, ftol=1e-7, maxiter=500, maxfun=700,
            disp=False,
        )
        LOGGER.info(f"Full optimisation result: {print_iter[0]} | [{', '.join(f'{x:.2f}' for x in opt)}] | {min_MSE[0]:7.3f}")
        return BatesDynamicsParameters.from_calibration_output(opt_arr=opt, s0=S0, r=r)

    @staticmethod
    def calibrate(
            df_options: pd.DataFrame, S0: float, r: float, side: OptionSide,
            heston_search_grid: HestonSearchGridType = HestonDynamicsParameters.get_default_search_grid(),
            jumponly_search_grid: JumpOnlySearchGridType = JumpOnlyDynamicsParameters.get_default_search_grid(),
            ft_method: FtMethod = FtMethod.LEWIS,
            h_params_path: Optional[str] = None,
            skip_plot: bool = True,
    ) -> BatesDynamicsParameters:
        """
        Calibrate Bates (1996) model parameters to market data.
        Returns a BatesDynamicsParameters object with calibrated parameters.
        """
        assert not df_options.empty or h_params_path, "Either df_options or h_params_path must be provided for h_params."
        # Load Heston parameters
        h_params: HestonDynamicsParameters
        if h_params_path:
            h_params = get_calibrated_heston_params(h_params_path=h_params_path, S0=S0, r=r)
        else:
            h_params = HestonFtiCalibrator.calibrate(
                df_options=df_options, S0=S0, r=r, side=side,
                search_grid=heston_search_grid, ft_method=ft_method
            )
        LOGGER.info(f"Bates.Heston calib: {h_params}")

        # Calibrate jump parameters
        jump_params: JumpOnlyDynamicsParameters = BatesFtiCalibrator.calibrate_jump(
            df_options=df_options, S0=S0,
            kappa_v=h_params.kappa_heston, theta_v=h_params.theta_heston, sigma_v=h_params.sigma_heston, rho=h_params.rho_heston, v0=h_params.v0_heston,
            jump_search_grid=jumponly_search_grid,
            side=side, ft_method=ft_method
        )
        LOGGER.info(f"Bates.Jump calib: {jump_params}")

        initial_bates_params: BatesDynamicsParameters = BatesDynamicsParameters.from_dynamic_parameters(
            h_params=h_params, j_params=jump_params
        )
        if not skip_plot:
            plot_calibration_result(
                df_options=df_options.copy(),
                model_values=BatesFtiCalibrator.calculate_option_price_batch(
                    df_options, S0, *initial_bates_params.get_values(),
                    side=side, ft_method=ft_method
                ),
                side=side
            )

        # Calibrate full Bates parameters if search grid is provided
        bates_params: BatesDynamicsParameters = BatesFtiCalibrator.calibrate_full(
            df_options=df_options,
            S0=S0,
            r=r,
            initial_params=initial_bates_params,
            side=side,
            ft_method=ft_method
        )

        return bates_params.get_bounded_parameters()


def get_calibrated_heston_params(h_params_path: str, S0: float, r: float) -> HestonDynamicsParameters:
    calib_result: HestonDynamicsParameters
    assert h_params_path is not None
    #Load pre-calibrated Heston model parameters
    heston_calibrated_params_df: pd.DataFrame = pd.read_csv(h_params_path)
    calib_result = HestonDynamicsParameters(
        x0=S0, r=r,
        kappa_heston=heston_calibrated_params_df['kappa_v'].iloc[0],
        sigma_heston=heston_calibrated_params_df['sigma_v'].iloc[0],
        theta_heston=heston_calibrated_params_df['theta_v'].iloc[0],
        rho_heston=heston_calibrated_params_df['rho'].iloc[0],
        v0_heston=heston_calibrated_params_df['v0'].iloc[0],
    ).get_bounded_parameters()
    LOGGER.info(f"Heston model parameters loaded: {calib_result}")

    return calib_result


def ex_pricing():
    ft_method: FtMethod = FtMethod.LEWIS

    """Example: price a call and put option under Bates (1996) model."""
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    kappa_v = 1.5
    theta_v = 0.02
    sigma_v = 0.15
    rho = 0.1
    v0 = 0.01
    lambd = 0.25
    mu = -0.2
    delta = 0.1
    for side in [OptionSide.CALL, OptionSide.PUT]:
        value = BatesFtiCalibrator.calculate_option_price(
            S0=S0, K=K, T=T, r=r,
            kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho_v=rho, v0=v0,
            lambd=lambd, mu=mu, delta=delta,
            side=side, ft_method=ft_method
        )
        LOGGER.info(f"B96 {side.name} option price via Lewis(2001): ${value:10.4f}")


def ex_calibration(
        data_path: str,
        side: OptionSide,
        params_path = None,
        skip_plot: bool = False
):
    """Example: calibrate Bates (1996) model to market data and plot results for given OptionSide.
    Runs both short (jump-only) and full calibration.
    """
    S0: float = 3225.93  # EURO STOXX 50 level 30.09.2014
    r: float = 0.02
    ft_method: FtMethod = FtMethod.LEWIS  # or FtMethod.CARRMADAN

    df_options: pd.DataFrame = load_option_data(
        path_str=data_path, S0=S0,
        r_provider=lambda *_: r # constant short rate
    )

    params_bates: BatesDynamicsParameters  = BatesFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        heston_search_grid=BatesDynamicsParameters.get_default_heston_search_grid(),
        jumponly_search_grid=BatesDynamicsParameters.get_default_jumponly_search_grid(),
        h_params_path=params_path,
        ft_method=ft_method
    )
    LOGGER.info(f"Bates calib: {params_bates}")

    if not skip_plot:
        kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta = params_bates.get_values()
        plot_calibration_result(
            df_options=df_options,
            model_values=BatesFtiCalibrator.calculate_option_price_batch(
                df_options=df_options, S0=S0,
                kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho=rho, v0=v0,
                lambd=lambd, mu=mu, delta=delta,
                side=side, ft_method=ft_method
            ),
            side=side
        )


if __name__ == "__main__":
    ex_pricing()
    ex_calibration(
        # params_path=get_path_from_package("erdqlib@src/ft/data/opt_sv_M2.csv"),
        data_path=get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv"),
        side=OptionSide.CALL,
        skip_plot=False,
    )
    # ex_calibration(side=OptionSide.PUT)
