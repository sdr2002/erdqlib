from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import splev, splrep
from scipy.optimize import fmin, brute

from erdqlib.src.ft.bates import BatesFtiCalibrator
from erdqlib.src.common.option import OptionSide, OptionDataColumn
from erdqlib.src.common.rate import instantaneous_rate, annualized_continuous_rate, capitalization_factor
from erdqlib.src.ft.cir import CirCalibrator
from erdqlib.src.ft.heston import HestonFtiCalibrator
from erdqlib.src.ft.jump import JumpFtiCalibrator
from erdqlib.src.mc.bcc import BCCParameters, BccDynamicsParameters, B
from erdqlib.src.mc.cir import CirDynamicsParameters
from erdqlib.src.mc.heston import HestonDynamicsParameters
from erdqlib.src.mc.jump import JumpOnlyDynamicsParameters
from erdqlib.tests.src.mc.test_heston import heston_params
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)


# BCC (1997) characteristic function (H93+M76)
def BCC_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta):
    """
    BCC (1997) characteristic function
    """
    H93 = HestonFtiCalibrator.calculate_characteristic(
        u=u, T=T, r=r,
        kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho=rho, v0=v0
    )
    M76J = JumpFtiCalibrator.calculate_characteristic(
        u=u, T=T, r=None,
        lambd=lambd, mu=mu, delta=delta,
        exclude_diffusion=True
    )
    return H93 * M76J


# Lewis (2001) integral value of BCC (1997)
def BCC_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta):
    """
    Lewis (2001) integral value for BCC (1997) characteristic function
    """
    char_func_value = BCC_char_func(
        u - 1j * 0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta
    )
    int_func_value = (
            1 / (u ** 2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    )
    return int_func_value


def BCC_eur_option_value_lewis(
        S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta, side: OptionSide
) -> float:
    """
    Valuation of European call option in B96 Model via Lewis (2001)
    Parameters:
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
    lambd: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    ==========
    """
    int_value = quad(
        lambda u: BCC_int_func(
            u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta
        ),
        0,
        np.inf,
        limit=250,
    )[0]
    call_value = float(max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value))
    if side is OptionSide.CALL:
        return call_value
    elif side is OptionSide.PUT:
        return call_value - S0 + K * np.exp(-r * T)
    raise ValueError(f"Invalid side: {side}")


def BCC_error_function_short(
        task_params: np.ndarray, df_options: pd.DataFrame, S0: float, heston_params: HestonDynamicsParameters,
        print_iter: List[int], min_MSE: List[float], side: OptionSide,
        regularise: bool = False, initial_params: Optional[np.ndarray] = None
) -> float:
    """
    Error function for BCC (1997) model
    """
    lambd, mu, delta = task_params
    kappa_v, theta_v, sigma_v, rho, v0 = heston_params.get_values()
    if JumpOnlyDynamicsParameters.do_parameters_offbound(lambd=lambd, mu=mu, delta=delta):
        return 5000.0
    se = []
    for row, option in df_options.iterrows():
        model_value = BCC_eur_option_value_lewis(
            S0,
            option[OptionDataColumn.STRIKE], option[OptionDataColumn.TENOR], option[OptionDataColumn.RATE],
            kappa_v, theta_v, sigma_v, rho, v0,
            lambd, mu, delta,
            side=side
        )
        se.append((model_value - option[side.name]) ** 2)
    MSE = sum(se) / len(se)
    min_MSE[0] = min(min_MSE[0], MSE)
    if print_iter[0] % 25 == 0:
        LOGGER.info(
            f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in task_params)}] | {MSE:7.3f} | {min_MSE[0]:7.3f}")
    print_iter[0] += 1
    if regularise:
        penalty = np.sqrt(np.sum((task_params - initial_params) ** 2)) * 1
        return MSE + penalty
    return MSE


def BCC_error_function_full(
        task_params: np.ndarray, options: pd.DataFrame, S0: float,
        print_iter: List[int], min_MSE: List[float], side: OptionSide,
        regularise: bool = False, initial_params: Optional[np.array] = None
) -> float:
    """
    Error function for full parameter calibration of BCC model
    """
    kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta = task_params
    # Parameter bounds
    if HestonDynamicsParameters.do_parameters_offbound(kappa_v, theta_v, sigma_v, rho, v0):
        return 5000.0

    se = []
    for row, option in options.iterrows():
        model_value = BCC_eur_option_value_lewis(
            S0,
            option[OptionDataColumn.STRIKE], option[OptionDataColumn.TENOR], option[OptionDataColumn.RATE],
            kappa_v, theta_v, sigma_v, rho, v0,
            lambd, mu, delta,
            side=side
        )
        se.append((model_value - option[side.name]) ** 2)
    MSE = sum(se) / len(se)
    min_MSE[0] = min(min_MSE[0], MSE)
    if print_iter[0] % 25 == 0:
        LOGGER.info(
            f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in task_params)}] | {MSE:7.3f} | {min_MSE[0]:7.3f}")
    print_iter[0] += 1
    if regularise:
        penalty = np.sqrt(np.sum((task_params - initial_params) ** 2)) * 1
        return MSE + penalty
    return MSE


def BCC_calibration_short(
        df_options: pd.DataFrame, heston_params: HestonDynamicsParameters,
        S0: float, side: OptionSide
) -> JumpOnlyDynamicsParameters:
    """Calibrates jump component of BCC97 model to market quotes."""
    # We first run with brute force
    # (scan sensible regions)
    print_iter = [0]
    min_MSE = [5000.0]
    LOGGER.info("Brute force optimization begins")
    opt1: np.ndarray = brute(  # type: ignore
        lambda p: BCC_error_function_short(
            task_params=p, heston_params=heston_params,
            df_options=df_options, S0=S0,
            print_iter=print_iter, min_MSE=min_MSE, side=side
        ),
        (
            (0.0, 0.51, 0.1),  # lambda
            (-0.5, -0.11, 0.1),  # mu
            (0.0, 0.51, 0.25),
        ),  # delta
        finish=None,
    )
    LOGGER.info(
        f"Brute-force result: {print_iter[0]} | [{', '.join(f'{x:.2f}' for x in opt1)}] | {min_MSE[0]:7.3f}"
    )

    # second run with local, convex minimization
    # (dig deeper where promising)
    LOGGER.info("fmin optimization begins")
    opt2: np.ndarray = fmin(
        lambda p: BCC_error_function_short(
            task_params=p, heston_params=heston_params,
            df_options=df_options, S0=S0,
            print_iter=print_iter, min_MSE=min_MSE, side=side
        ),
        opt1,
        xtol=0.0000001,
        ftol=0.0000001,
        maxiter=550,
        maxfun=750,
    )[0]
    LOGGER.info(
        f"Fmin result: {print_iter[0]} | [{', '.join(f'{x:.2f}' for x in opt2)}] | {min_MSE[0]:7.3f}"
    )
    return JumpOnlyDynamicsParameters(
        S0=S0, r=None,
        lambd_merton=float(opt2[0]), mu_merton=float(opt2[1]), delta_merton=float(opt2[2])
    )


def BCC_jump_calculate_model_values(
        mertonj_params: JumpOnlyDynamicsParameters, heston_params: HestonDynamicsParameters,
        df_options: pd.DataFrame, S0: float, side: OptionSide
) -> np.ndarray:
    """Calculates all model values given parameter vector p0."""
    lambd, mu, delta = mertonj_params.get_values()
    kappa_v, theta_v, sigma_v, rho, v0 = heston_params.get_values()
    values = []
    for row, option in df_options.iterrows():
        model_value = BCC_eur_option_value_lewis(
            S0,
            option[OptionDataColumn.STRIKE], option[OptionDataColumn.TENOR], option[OptionDataColumn.RATE],
            kappa_v, theta_v, sigma_v, rho, v0,
            lambd, mu, delta,
            side=side
        )
        values.append(model_value)
    return np.array(values)


def plot_BCC_short(
        mertonj_params: JumpOnlyDynamicsParameters,
        heston_params: HestonDynamicsParameters,
        df_options: pd.DataFrame,
        S0: float,
        side: OptionSide
):
    df_options_to_plot = df_options.copy()
    df_options_to_plot[OptionDataColumn.MODEL] = BCC_jump_calculate_model_values(
        mertonj_params=mertonj_params, heston_params=heston_params, df_options=df_options_to_plot, S0=S0, side=side
    )
    for maturity, df_options_per_maturity in df_options_to_plot.groupby(OptionDataColumn.MATURITY):
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.grid()
        plt.title(f"(Short-calib) {side.name} at Maturity {maturity}")
        plt.ylabel("option values")
        plt.plot(df_options_per_maturity[OptionDataColumn.STRIKE], df_options_per_maturity[side.name], "b", label="market")
        plt.plot(df_options_per_maturity[OptionDataColumn.STRIKE], df_options_per_maturity[OptionDataColumn.MODEL], "ro", label="model")
        plt.legend(loc=0)
        axis1: List[float] = [
            min(df_options_per_maturity[OptionDataColumn.STRIKE]) - 10,
            max(df_options_per_maturity[OptionDataColumn.STRIKE]) + 10,
            min(df_options_per_maturity[side.name]) - 10,
            max(df_options_per_maturity[side.name]) + 10,
        ]
        plt.axis(axis1)  # type: ignore

        plt.subplot(212)
        plt.grid()
        wi = 5.0
        diffs = df_options_per_maturity[OptionDataColumn.MODEL].values - df_options_per_maturity[side.name].values
        plt.bar(df_options_per_maturity[OptionDataColumn.STRIKE].values - wi / 2, diffs, width=wi)
        plt.ylabel("difference")
        axis2: List[float] = [
            min(df_options_per_maturity[OptionDataColumn.STRIKE]) - 10,
            max(df_options_per_maturity[OptionDataColumn.STRIKE]) + 10,
            min(diffs) * 1.1,
            max(diffs) * 1.1,
        ]
        plt.axis(axis2)  # type: ignore
        plt.tight_layout()
        plt.show()


def BCC_calibration_full(
        df_options: pd.DataFrame, S0: float, initial_params: np.ndarray, cir_params: CirDynamicsParameters,
        side: OptionSide
) -> BccDynamicsParameters:
    """Full calibration of BCC (1997)"""
    print_iter = [0]
    min_MSE = [5000.0]

    LOGGER.info("fmin optimization begins")
    opt: np.ndarray = fmin(
        lambda p: BCC_error_function_full(p, df_options, S0, print_iter, min_MSE, side),
        initial_params,
        xtol=0.000001, ftol=0.000001, maxiter=450, maxfun=750
    )
    LOGGER.info(
        f"Full optimisation result: {print_iter[0]} | [{', '.join(f'{x:.2f}' for x in opt)}] | {min_MSE[0]:7.3f}")
    return BccDynamicsParameters(
        S0=S0,

        r=None,
        kappa_cir=cir_params.kappa_cir,
        theta_cir=cir_params.theta_cir,
        sigma_cir=cir_params.sigma_cir,

        kappa_heston=float(opt[0]),
        theta_heston=float(opt[1]),
        sigma_heston=float(opt[2]),
        rho_heston=float(opt[3]),
        v0_heston=float(opt[4]),

        lambd_merton=float(opt[5]),
        mu_merton=float(opt[6]),
        delta_merton=float(opt[7]),
    )


def BCC_calculate_model_values(p0: BccDynamicsParameters, df_options: pd.DataFrame, side: OptionSide) -> np.array:
    """Calculates all model values given parameter vector p0."""
    kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta = p0.get_bates_parameters().get_values()
    values = []
    for row, option in df_options.iterrows():
        model_value = BCC_eur_option_value_lewis(
            p0.S0,
            option[OptionDataColumn.STRIKE], option[OptionDataColumn.TENOR], option[OptionDataColumn.RATE],
            kappa_v, theta_v, sigma_v, rho, v0,
            lambd, mu, delta,
            side=side
        )
        values.append(model_value)
    return np.array(values)


def plot_BCC_full(bcc_params: BccDynamicsParameters, df_options: pd.DataFrame, side: OptionSide):
    df_options[OptionDataColumn.MODEL] = BCC_calculate_model_values(p0=bcc_params, df_options=df_options, side=side)
    for maturity, df_options_per_maturity in df_options.groupby(OptionDataColumn.MATURITY):
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.grid()
        plt.title(f"(Full-calib) {side.name} at Maturity {maturity}")
        plt.ylabel("option values")
        plt.plot(df_options_per_maturity[OptionDataColumn.STRIKE], df_options_per_maturity[side.name], "b", label="market")
        plt.plot(df_options_per_maturity[OptionDataColumn.STRIKE], df_options_per_maturity[OptionDataColumn.MODEL], "ro", label="model")
        plt.legend(loc=0)
        axis1 = [
            min(df_options_per_maturity[OptionDataColumn.STRIKE]) - 10,
            max(df_options_per_maturity[OptionDataColumn.STRIKE]) + 10,
            min(df_options_per_maturity[side.name]) - 10,
            max(df_options_per_maturity[side.name]) + 10,
        ]
        plt.axis(axis1)  # type: ignore

        plt.subplot(212)
        plt.grid()
        wi = 5.0
        diffs = df_options_per_maturity[OptionDataColumn.MODEL].values - df_options_per_maturity[side.name].values
        plt.bar(df_options_per_maturity[OptionDataColumn.STRIKE].values - wi / 2, diffs, width=wi)
        plt.ylabel("difference")
        axis2 = [
            min(df_options_per_maturity[OptionDataColumn.STRIKE]) - 10,
            max(df_options_per_maturity[OptionDataColumn.STRIKE]) + 10,
            min(diffs) * 1.1,
            max(diffs) * 1.1,
        ]
        plt.axis(axis2)  # type: ignore
        plt.tight_layout()
        plt.show()

def ex_pricing():
    bcc_params = BCCParameters(
        S0=100.,
        K=90.,
        Ti=0.,
        Tf=1.,
        r0=-0.032 / 100,
        kappa_r=0.068,
        theta_r=0.207,
        sigma_r=0.112,
        v0=0.035,
        kappa_v=18.447,
        theta_v=0.026,
        sigma_v=0.978,
        rho=-0.821,
        lambd=0.008,
        mu=-0.600,
        delta=0.001
    )

    side: OptionSide = OptionSide.CALL
    bates_price: float = BatesFtiCalibrator.calculate_option_price_lewis(
        *bcc_params.get_pricing_params(apply_shortrate=False), side
    )
    LOGGER.info(f"Option value under Bates: {bates_price}")

    bcc_price: float = BCC_eur_option_value_lewis(
        *bcc_params.get_pricing_params(apply_shortrate=True), side  # type: ignore
    )
    LOGGER.info(f"{side} Option value under BCC (1997): {bcc_price}")


def ex_calibration(data_path: str, side: OptionSide, skip_plot: bool):
    ### Rates process
    # Euribor Market data
    S0 = 3225.93
    t_i = 0
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
    )  # Euribor is IR product which is a single cash flow, hence is a zero-coupon bond where YTM = spot-rate

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
    LOGGER.info(f"CIR calib: {params_cir}")

    ### Load data on September 30, 2014 with r from the CIR model instead of constant overnight rate 0.005      # EURO STOXX 50 level
    df_options: pd.DataFrame = load_option_data(
        path_str=data_path, S0=S0,
        r_provider=lambda maturity, r_overnight=r0, ti=t_i,
                          kappa_r=params_cir.kappa_cir, theta_r=params_cir.theta_cir, sigma_r=params_cir.sigma_cir: instantaneous_rate(
            zcb_price=B([r_overnight, kappa_r, theta_r, sigma_r, ti, maturity]),
            t_year=maturity
        )
    )

    ### Short calibration
    # Heston component calib
    LOGGER.info("\n--- Short (Heston) calibration ---")
    params_heston: HestonDynamicsParameters = HestonFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=None, side=side,
        search_grid=HestonDynamicsParameters.get_default_search_grid()
    )
    LOGGER.info(f"Heston calib: {params_heston}")

    # Merton jump component calibration
    LOGGER.info("\n--- Short (Jump-only) calibration ---")
    params_jump: JumpOnlyDynamicsParameters = BCC_calibration_short(
        df_options=df_options, heston_params=params_heston, S0=S0, side=side
    )
    if not skip_plot:
        plot_BCC_short(
            mertonj_params=params_jump,
            heston_params=params_heston,
            df_options=df_options,
            S0=S0,
            side=side
        )

    ### Full calibration
    LOGGER.info("\n--- Full calibration ---")
    params_bcc: BccDynamicsParameters = BCC_calibration_full(
        df_options=df_options, S0=S0,
        initial_params=np.array(
            params_heston.get_values() + params_jump.get_values()
        ),
        cir_params=params_cir,
        side=side
    )
    if not skip_plot:
        plot_BCC_full(
            bcc_params=params_bcc,
            df_options=df_options,
            side=side
        )


if __name__ == "__main__":
    ex_pricing()
    ex_calibration(
        data_path=get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv"),
        side=OptionSide.CALL,
        skip_plot=False
    )
