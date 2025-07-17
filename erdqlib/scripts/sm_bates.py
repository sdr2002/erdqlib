from typing import Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brute, fmin

from erdqlib.scripts.caculator import FtMethod
from erdqlib.scripts.sm_heston import H93_char_func, H93_calibration_full, load_option_data
from erdqlib.src.common.option import OptionSide
from erdqlib.src.mc.heston import HestonDynamicsParameters
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def M76J_char_func(u, T, lamb, mu, delta):
    """Jump component characteristic function for Merton '76 model."""
    omega = -lamb * (np.exp(mu + 0.5 * delta**2) - 1)
    return np.exp(
        (1j * u * omega + lamb * (np.exp(1j * u * mu - 0.5 * u**2 * delta**2) - 1)) * T
    )

def B96_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """Bates (1996) characteristic function."""
    H93 = H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    M76J = M76J_char_func(u, T, lamb, mu, delta)
    return H93 * M76J

def B96_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """Lewis (2001) integral for Bates (1996) characteristic function."""
    char_func_value = B96_char_func(
        u - 0.5j, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
    )
    return (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real / (u**2 + 0.25)

def B96_eur_option_value_lewis(
    S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, side: OptionSide
):
    """European option value in Bates (1996) model via Lewis (2001)."""
    int_value = quad(
        lambda u: B96_int_func(
            u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
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


def B96_eur_option_value_carrmadan(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, side: OptionSide):
    """
    Call option price in Bates (1996) under FFT

    """

    k = np.log(K / S0)
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
            B96_char_func(v, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
            / (alpha**2 + alpha - vo**2 + 1j * (2 * alpha + 1) * vo)
        )

    else:
        alpha = 1.1
        v = (vo - 1j * alpha) - 1j
        modcharFunc1 = np.exp(-r * T) * (
            1 / (1 + 1j * (vo - 1j * alpha))
            - np.exp(r * T) / (1j * (vo - 1j * alpha))
            - B96_char_func(
                v, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
            )
            / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha))
        )

        v = (vo + 1j * alpha) - 1j

        modcharFunc2 = np.exp(-r * T) * (
            1 / (1 + 1j * (vo + 1j * alpha))
            - np.exp(r * T) / (1j * (vo + 1j * alpha))
            - B96_char_func(
                v, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
            )
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

    pos = int((k + b) / eps)
    CallValue = CallValueM[pos] * S0

    if side is OptionSide.CALL:
        return CallValue
    elif side is OptionSide.PUT:
        return CallValue - S0 + K * np.exp(-r * T)
    raise ValueError(f"Invalid side: {side}")


def B96_eur_option_value(
    S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta,
    side: OptionSide, method: FtMethod = FtMethod.LEWIS
):
    if method is FtMethod.LEWIS:
        return B96_eur_option_value_lewis(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, side)
    elif method is FtMethod.CARRMADAN:
        return B96_eur_option_value_carrmadan(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, side)
    raise ValueError(f"Invalid FtMethod method: {method}")


def B96_error_function(
    p0, options, S0, kappa_v, theta_v, sigma_v, rho, v0, side: OptionSide, print_iter=None, min_MSE=None, opt1=None
):
    """Error function for Bates (1996) model calibration."""
    lamb, mu, delta = p0
    if lamb < 0.0 or mu < -0.6 or mu > 0.0 or delta < 0.0:
        return 5000.0
    se = []
    for _, option in options.iterrows():
        model_value = B96_eur_option_value(
            S0,
            option["Strike"],
            option["T"],
            option["r"],
            kappa_v,
            theta_v,
            sigma_v,
            rho,
            v0,
            lamb,
            mu,
            delta,
            side,
        )
        se.append((model_value - option[side.name]) ** 2)
    MSE = sum(se) / len(se)
    if min_MSE is not None:
        min_MSE[0] = min(min_MSE[0], MSE)
    if print_iter is not None:
        if print_iter[0] % 25 == 0:
            LOGGER.info(f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in p0)}] | {MSE:7.3f} | {min_MSE[0]:7.3f}")
        print_iter[0] += 1
    if opt1 is not None:
        penalty = np.sqrt(np.sum((p0 - opt1) ** 2)) * 1
        return MSE + penalty
    return MSE

def B96_calibration_short(options, S0, kappa_v, theta_v, sigma_v, rho, v0, side: OptionSide):
    """Calibrates jump component of Bates (1996) model to market prices."""
    print_iter = [0]
    min_MSE = [5000.0]
    opt1 = brute(
        lambda p: B96_error_function(
            p, options, S0, kappa_v, theta_v, sigma_v, rho, v0, side, print_iter=print_iter, min_MSE=min_MSE
        ),
        (
            (0.0, 0.51, 0.1),  # lambda
            (-0.5, -0.11, 0.1),  # mu
            (0.0, 0.51, 0.25),   # delta
        ),
        finish=None,
    )
    opt2 = fmin(
        lambda p: B96_error_function(
            p, options, S0, kappa_v, theta_v, sigma_v, rho, v0, side, print_iter=print_iter, min_MSE=min_MSE
        ),
        opt1,
        xtol=1e-7,
        ftol=1e-7,
        maxiter=550,
        maxfun=750,
    )
    return opt2

def B96_jump_calculate_model_values(options, S0, kappa_v, theta_v, sigma_v, rho, v0, p0, side: OptionSide):
    """Calculates all model values given parameter vector p0."""
    lamb, mu, delta = p0
    values = []
    for _, option in options.iterrows():
        model_value = B96_eur_option_value(
            S0,
            option["Strike"],
            option["T"],
            option["r"],
            kappa_v,
            theta_v,
            sigma_v,
            rho,
            v0,
            lamb,
            mu,
            delta,
            side,
        )
        values.append(model_value)
    return np.array(values)

def plot_calibration_results(options, model_values, side: OptionSide):
    """Plot market and model prices for each maturity and OptionSide."""
    options = options.copy()
    options["Model"] = model_values
    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.grid()
    plt.title("Maturity %s %s" % (str(options["Maturity"].iloc[0])[:10], side.name))
    plt.ylabel("option values")
    plt.plot(options.Strike, options[side.name], "b", label="market")
    plt.plot(options.Strike, options.Model, "ro", label="model")
    plt.legend(loc=0)
    plt.axis([
        min(options.Strike) - 10,
        max(options.Strike) + 10,
        min(options[side.name]) - 10,
        max(options[side.name]) + 10,
    ])
    plt.subplot(212)
    plt.grid()
    wi = 5.0
    diffs = options.Model.values - options[side.name].values
    plt.bar(options.Strike.values - wi / 2, diffs, width=wi)
    plt.ylabel("difference")
    plt.axis([
        min(options.Strike) - 10,
        max(options.Strike) + 10,
        min(diffs) * 1.1,
        max(diffs) * 1.1,
    ])
    plt.tight_layout()
    plt.show()


def B96_full_error_function(
    p0: np.ndarray, options: pd.DataFrame, S0: float,
    print_iter: List[int], min_MSE: List[float], side: OptionSide
):
    """Error function for full Bates (1996) model calibration."""
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = p0
    # Parameter bounds
    if HestonDynamicsParameters.do_parameters_offbound(*p0):
        return 5000.0
    se = []
    for _, option in options.iterrows():
        model_value = B96_eur_option_value(
            S0,
            option["Strike"],
            option["T"],
            option["r"],
            kappa_v,
            theta_v,
            sigma_v,
            rho,
            v0,
            lamb,
            mu,
            delta,
            side,
        )
        se.append((model_value - option[side.name]) ** 2)
    MSE = sum(se) / len(se)
    min_MSE[0] = min(min_MSE[0], MSE)
    if print_iter[0] % 25 == 0:
        LOGGER.info(f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in p0)}] | {MSE:7.3f} | {min_MSE[0]:7.3f}")
    print_iter[0] += 1
    return MSE

def B96_calibration_full(options, S0, p0, side: OptionSide):
    """Calibrates all Bates (1996) model parameters to market prices."""
    print_iter = [0]
    min_MSE = [5000.0]

    LOGGER.info("fmin optimization")
    opt = fmin(
        lambda p: B96_full_error_function(p, options, S0, print_iter, min_MSE, side),
        p0,
        xtol=1e-7,
        ftol=1e-7,
        maxiter=500,
        maxfun=700,
    )
    LOGGER.info(f"Optimisation result: {print_iter[0]} | [{', '.join(f'{x:.2f}' for x in opt)}] | {min_MSE[0]:7.3f}")
    return opt

def B96_calculate_model_values(options, S0, p0, side: OptionSide):
    """Calculates all model values for full Bates model given parameter vector p0."""
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = p0
    values = []
    for _, option in options.iterrows():
        model_value = B96_eur_option_value(
            S0,
            option["Strike"],
            option["T"],
            option["r"],
            kappa_v,
            theta_v,
            sigma_v,
            rho,
            v0,
            lamb,
            mu,
            delta,
            side,
        )
        values.append(model_value)
    return np.array(values)

def plot_full_calibration_results(options, model_values, side: OptionSide):
    """Plot market and model prices for each maturity and OptionSide (full calibration)."""
    options = options.copy()
    options["Model"] = model_values
    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.grid()
    plt.title("Maturity %s %s (Full Bates)" % (str(options["Maturity"].iloc[0])[:10], side.name))
    plt.ylabel("option values")
    plt.plot(options.Strike, options[side.name], "b", label="market")
    plt.plot(options.Strike, options.Model, "ro", label="model")
    plt.legend(loc=0)
    plt.axis([
        min(options.Strike) - 10,
        max(options.Strike) + 10,
        min(options[side.name]) - 10,
        max(options[side.name]) + 10,
    ])
    plt.subplot(212)
    plt.grid()
    wi = 5.0
    diffs = options.Model.values - options[side.name].values
    plt.bar(options.Strike.values - wi / 2, diffs, width=wi)
    plt.ylabel("difference")
    plt.axis([
        min(options.Strike) - 10,
        max(options.Strike) + 10,
        min(diffs) * 1.1,
        max(diffs) * 1.1,
    ])
    plt.tight_layout()
    plt.show()


def ex_pricing():
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
    lamb = 0.25
    mu = -0.2
    delta = 0.1
    for side in [OptionSide.CALL, OptionSide.PUT]:
        value = B96_eur_option_value(
            S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, side
        )
        LOGGER.info(f"B96 {side.name} option price via Lewis(2001): ${value:10.4f}")


def get_calibrated_heston_params(
    params_path: Optional[str] = None,  # Path to pre-calibrated parameters
    data_path: Optional[str] = None, S0: float = None, r: float = None, side: OptionSide = None
) -> HestonDynamicsParameters:
    calib_result: HestonDynamicsParameters
    if params_path:
        #Load pre-calibrated Heston model parameters
        kappa_v, theta_v, sigma_v, rho, v0 = np.load(params_path)
        calib_result = HestonDynamicsParameters(
            S0=S0,
            r=r,
            v0=v0,
            kappa=kappa_v,
            sigma=sigma_v,
            theta=theta_v,
            rho=rho
        ).get_bounded_parameters()
        LOGGER.info(f"Heston model parameters loaded: {kappa_v:.3g}, {theta_v:.3g}, {sigma_v:.3g}, {rho:.3g}, {v0:.3g}")
    elif data_path:
        #Calibrate Heston model parameters to market data first
        df_options = load_option_data(path_str=data_path, S0=S0, r=r)
        calib_result: HestonDynamicsParameters = H93_calibration_full(
            df_options=df_options, S0=S0, r=r, side=side,
            search_grid=HestonDynamicsParameters.get_default_search_grid()
        )
        LOGGER.info(f"Heston model parameters calibrated: {calib_result}")
    else:
        raise ValueError("Either params_path or data_path must be provided for Heston model calibration.")

    return calib_result

def ex_calibration(
        data_path=None,
        params_path=None,
        side: OptionSide = OptionSide.CALL,
        skip_plot: bool = False
):
    """Example: calibrate Bates (1996) model to market data and plot results for given OptionSide.
    Runs both short (jump-only) and full calibration.
    """

    S0 = 3225.93  # EURO STOXX 50 level 30.09.2014
    r = 0.02
    options: pd.DataFrame = load_option_data(path_str=data_path, S0=S0, r=r)  # Load option data from HDF5 file
    heston_result: HestonDynamicsParameters = get_calibrated_heston_params(
        data_path=data_path, S0=S0, r=r, side=side,
        params_path=params_path
    )

    # Select closest maturity
    mats = sorted(set(options["Maturity"]))
    options = options[options["Maturity"] == mats[0]]

    # Short (jump-only) calibration
    LOGGER.info("\n--- Short (jump-only) calibration ---")
    mertonj_result = B96_calibration_short(options, S0, *heston_result.get_values(), side) # lamb, mu, delta
    model_values_short = B96_jump_calculate_model_values(options, S0, *heston_result.get_values(), mertonj_result, side)
    if not skip_plot:
        plot_calibration_results(options, model_values_short, side)

    # Full Bates calibration
    LOGGER.info("\n--- Full Bates calibration ---")
    params_full = list(x for x in heston_result.get_values())
    params_full.extend(mertonj_result)
    params_full = B96_calibration_full(options, S0, p0=params_full, side=side)
    model_values_full = B96_calculate_model_values(options, S0, params_full, side)
    if not skip_plot:
        plot_full_calibration_results(options, model_values_full, side)


if __name__ == "__main__":
    ex_pricing()
    ex_calibration(side=OptionSide.CALL, data_path="./option_dataset.h5", skip_plot=True) # params_path="./opt_sv_M2.npy"
    # ex_calibration(side=OptionSide.PUT)