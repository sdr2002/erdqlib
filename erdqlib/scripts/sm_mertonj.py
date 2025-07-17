import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brute, fmin

from erdqlib.src.common.option import OptionSide
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def M76_char_func(u, T, r, sigma, lamb, mu, delta):
    r"""
    Characteristic function for the Merton ’76 jump-diffusion model.

    φ₀^{M76}(u, T) = exp( [i·u·ω
                          − ½·u²·σ²
                          + λ·(exp(i·u·μ − ½·u²·δ²) − 1)
                         ] · T )

    where
      ω = r − ½·σ² − λ·(exp(μ + ½·δ²) − 1)
    """
    omega = r - 0.5 * sigma**2 - lamb * (np.exp(mu + 0.5 * delta**2) - 1)
    return np.exp(
        (
            1j * u * omega
            - 0.5 * u**2 * sigma**2
            + lamb * (np.exp(1j * u * mu - 0.5 * u**2 * delta**2) - 1)
        )
        * T
    )

def M76_integration_function(u, S0, K, T, r, sigma, lamb, mu, delta):
    r"""
    Integrand for the Lewis (2001) FFT pricing under Merton ’76 model.

    C₀ = S₀ − (√(S₀·K)·e^{−r·T} / π)
           ∫₀^∞ Re[ e^{i·z·ln(S₀/K)} · φ(z − i/2, T) ] · dz / (z² + 1/4)

    This function returns
       Re[ e^{i·u·ln(S₀/K)} · φ(u − i/2, T) ] / (u² + 1/4)
    """
    char = M76_char_func(u - 0.5j, T, r, sigma, lamb, mu, delta)
    return (np.exp(1j * u * np.log(S0 / K)) * char).real / (u**2 + 0.25)


def M76_eur_option_value(S0, K, T, r, sigma, lamb, mu, delta, side: OptionSide):
    """
    Value of the European option under Lewis (2001) for Merton'76 jump diffusion model.
    Supports both CALL and PUT via OptionSide.
    """
    int_value = quad(
        lambda u: M76_integration_function(u, S0, K, T, r, sigma, lamb, mu, delta),
        0,
        50,
        limit=250,
    )[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value)
    if side is OptionSide.CALL:
        return call_value
    elif side is OptionSide.PUT:
        # Put-Call parity: P = C - S0 + K*exp(-rT)
        return call_value - S0 + K * np.exp(-r * T)
    raise ValueError(f"Invalid side: {side}")

def M76_error_function(p0, options, S0, side: OptionSide, print_iter=None, min_RMSE=None):
    """
    Error function for parameter calibration in Merton'76 model.
    Now supports both CALL and PUT via OptionSide.
    """
    sigma, lamb, mu, delta = p0
    if sigma < 0.0 or delta < 0.0 or lamb < 0.0:
        return 500.0
    se = []
    for _, option in options.iterrows():
        model_value = M76_eur_option_value(
            S0, option["Strike"], option["T"], option["r"], sigma, lamb, mu, delta, side
        )
        se.append((model_value - option[side.name]) ** 2)
    RMSE = np.sqrt(sum(se) / len(se))
    if min_RMSE is not None:
        min_RMSE[0] = min(min_RMSE[0], RMSE)
    if print_iter is not None:
        if print_iter[0] % 50 == 0:
            LOGGER.info(f"{print_iter[0]} | [{', '.join(f'{x:.2f}' for x in p0)}] | {MSE:7.3f} | {min_MSE[0]:7.3f}")
            print("%4d |" % print_iter[0], np.array(p0), "| %7.3f | %7.3f" % (RMSE, min_RMSE[0]))
        print_iter[0] += 1
    return RMSE

def M76_calibration_full(options, S0, side: OptionSide):
    """Calibrates Merton (1976) model to market quotes for given OptionSide."""
    print_iter = [0]
    min_RMSE = [100.0]
    p0 = brute(
        lambda p: M76_error_function(p, options, S0, side, print_iter, min_RMSE),
        (
            (0.075, 0.201, 0.025),  # sigma
            (0.10, 0.401, 0.1),     # lambda
            (-0.5, 0.01, 0.1),      # mu
            (0.10, 0.301, 0.1),     # delta
        ),
        finish=None,
    )
    opt = fmin(
        lambda p: M76_error_function(p, options, S0, side, print_iter, min_RMSE),
        p0, xtol=0.0001, ftol=0.0001, maxiter=550, maxfun=1050
    )
    return opt

def generate_plot(opt, options, S0, side: OptionSide):
    """Plot market and model prices for each maturity and OptionSide."""
    sigma, lamb, mu, delta = opt
    options = options.copy()
    options["Model"] = 0.0
    for row, option in options.iterrows():
        options.loc[row, "Model"] = M76_eur_option_value(
            S0, option["Strike"], option["T"], option["r"], sigma, lamb, mu, delta, side
        )
    mats = sorted(set(options["Maturity"]))
    options = options.set_index("Strike")
    for mat in mats:
        options[options["Maturity"] == mat][[side.name, "Model"]].plot(
            style=["b-", "ro"], title=f"Maturity {str(mat)[:10]} {side.name}"
        )
        plt.ylabel("Option Value")
    plt.show()

def ex_pricing():
    """Example: price a call and put option under Merton (1976) model."""
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.4
    lamb = 1
    mu = -0.2
    delta = 0.1
    for side in [OptionSide.CALL, OptionSide.PUT]:
        value = M76_eur_option_value(S0, K, T, r, sigma, lamb, mu, delta, side)
        print(f"Value of the {side.name} option under Merton (1976) is:  ${value}")

def ex_calibration(path_str="option_data_M2.h5", side: OptionSide = OptionSide.CALL):
    """Example: calibrate Merton (1976) model to market data and plot results for given OptionSide."""
    h5 = pd.HDFStore(path_str, "r")
    data = h5["data"]
    h5.close()
    S0 = 3225.93  # EURO STOXX 50 level September 30, 2014
    tol = 0.02
    options = data[(np.abs(data["Strike"] - S0) / S0) < tol]
    options = options.assign(**{dt_key: pd.to_datetime(options[dt_key]) for dt_key in ["Date", "Maturity"]})
    options = options.rename(columns={c: c.upper() for c in ["Call", "Put"]})
    for row, option in options.iterrows():
        T = (option["Maturity"] - option["Date"]).days / 365.0
        options.loc[row, "T"] = T
        options.loc[row, "r"] = 0.005  # ECB base rate
    opt = M76_calibration_full(options, S0, side)
    generate_plot(opt, options, S0, side)

if __name__ == "__main__":
    ex_pricing()
    ex_calibration(side=OptionSide.CALL)
    ex_calibration(side=OptionSide.PUT)
