import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brute, fmin
from erdqlib.src.common.option import OptionSide
from erdqlib.tool.logger_util import create_logger
from scripts.sm_heston import H93_char_func

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

def B96_eur_option_value(
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

def B96_error_function(
    p0, options, S0, kappa_v, theta_v, sigma_v, rho, v0, side: OptionSide, print_iter=None, min_MSE=None
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
            LOGGER.info(f"{int(print_iter[0])} | {p0} | {MSE:.3f} | {min_MSE[0]:.3f} |")
        print_iter[0] += 1
    return MSE

def B96_calibration_short(options, S0, kappa_v, theta_v, sigma_v, rho, v0, side: OptionSide):
    """Calibrates jump component of Bates (1996) model to market prices."""
    print_iter = [0]
    min_MSE = [5000.0]
    opt1 = brute(
        lambda p: B96_error_function(
            p, options, S0, kappa_v, theta_v, sigma_v, rho, v0, side, print_iter, min_MSE
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
            p, options, S0, kappa_v, theta_v, sigma_v, rho, v0, side, print_iter, min_MSE
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
    p0, options, S0, print_iter, min_MSE, side: OptionSide
):
    """Error function for full Bates (1996) model calibration."""
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = p0
    # Parameter bounds
    if (
        kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0
        or rho < -1.0 or rho > 1.0 or v0 < 0.0
        or lamb < 0.0 or delta < 0.0 or mu < -0.6 or mu > 0.0
        or 2 * kappa_v * theta_v < sigma_v**2
    ):
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
        LOGGER.info(f"{int(print_iter[0])} | {p0} | {MSE:.3f} | {min_MSE[0]:.3f} |")
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
        maxiter=1000,
        maxfun=2000,
    )
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

def ex_calibration(path_str="option_data_M2.h5", side: OptionSide = OptionSide.CALL):
    """Example: calibrate Bates (1996) model to market data and plot results for given OptionSide.
    Runs both short (jump-only) and full calibration.
    """
    h5 = pd.HDFStore(path_str, "r")
    data = h5["data"]
    h5.close()
    S0 = 3225.93  # EURO STOXX 50 level 30.09.2014
    kappa_v, theta_v, sigma_v, rho, v0 = np.load("opt_sv_M2.npy")
    tol = 0.02
    options = data[(np.abs(data["Strike"] - S0) / S0) < tol]
    options = options.assign(**{dt_key: pd.to_datetime(options[dt_key]) for dt_key in ["Date", "Maturity"]})
    options = options.rename(columns={c: c.upper() for c in ["Call", "Put"]})
    for row, option in options.iterrows():
        T = (option["Maturity"] - option["Date"]).days / 365.0
        options.loc[row, "T"] = T
        options.loc[row, "r"] = 0.02
    # Select closest maturity
    mats = sorted(set(options["Maturity"]))
    options = options[options["Maturity"] == mats[0]]

    # Short (jump-only) calibration
    LOGGER.info("\n--- Short (jump-only) calibration ---")
    params_short = B96_calibration_short(options, S0, kappa_v, theta_v, sigma_v, rho, v0, side) # lamb, mu, delta
    model_values_short = B96_jump_calculate_model_values(options, S0, kappa_v, theta_v, sigma_v, rho, v0, params_short, side)
    plot_calibration_results(options, model_values_short, side)

    # Full Bates calibration
    LOGGER.info("\n--- Full Bates calibration ---")
    params_full = [kappa_v, theta_v, sigma_v, rho, v0]
    params_full = params_full.extend(params_short)
    params_full = B96_calibration_full(options, S0, p0=params_full, side=side)
    model_values_full = B96_calculate_model_values(options, S0, params_full, side)
    plot_full_calibration_results(options, model_values_full, side)


if __name__ == "__main__":
    ex_pricing()
    ex_calibration(side=OptionSide.CALL)
    # ex_calibration(side=OptionSide.PUT)