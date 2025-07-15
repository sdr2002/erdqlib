
from typing import List

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brute, fmin
from erdqlib.src.common.option import OptionSide


def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
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
    c1 = kappa_v * theta_v
    c2 = -np.sqrt(
        (rho * sigma_v * u * 1j - kappa_v)**2
        - sigma_v**2 * (-u * 1j - u**2)
    )
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) / (
         kappa_v - rho * sigma_v * u * 1j - c2
    )

    # H1 and H2
    H1 = (
        1j * r * u * T
        + (c1 / sigma_v**2) * (
            (kappa_v - rho * sigma_v * u * 1j + c2)*T
            - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))
        )
    )
    H2 = (
        (kappa_v - rho * sigma_v * u * 1j + c2)
        / sigma_v**2
        * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T)))
    )

    return np.exp(H1 + H2 * v0)


def H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    r"""
    Integrand for Lewis (2001) call pricing under Heston ’93 model:

      C₀ = S₀ − (√(S₀·K)·e^{−r·T} / π)
           ∫₀^∞ Re[ e^{i·z·ln(S₀/K)} · φ^H(z − i/2, T) ]
                · dz / (z² + 1/4)

    This returns the real part of
      e^{i·u·ln(S₀/K)} · φ^H(u − i/2, T)
    divided by (u² + 1/4).
    """
    φ = H93_char_func(u - 0.5j, T, r,
                      kappa_v, theta_v, sigma_v, rho, v0)
    return (np.exp(1j * u * np.log(S0 / K)) * φ).real / (u**2 + 0.25)



def H93_eur_option_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, side: OptionSide):
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
        lambda u: H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0),
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


def H93_error_function(p0, df_options, i: List[int], min_MSE: List[float], s0: float, side: OptionSide):
    """Error function for parameter calibration via
    Lewis (2001) Fourier approach for Heston (1993).
    Parameters
    ==========
    p0: list[float]
        model parameters (kappa_v, theta_v, sigma_v, rho, v0)
    df_options: pd.DataFrame
        DataFrame with market option quotes
    i: List[int]
        List to keep track of iterations
    min_MSE: List[float]
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
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or rho < -1.0 or rho > 1.0:
        return 500.0
    if 2 * kappa_v * theta_v < sigma_v**2:
        return 500.0
    se = []
    for row, option in df_options.iterrows():
        model_value = H93_eur_option_value(
            s0,
            option["Strike"],
            option["T"],
            option["r"],
            kappa_v,
            theta_v,
            sigma_v,
            rho,
            v0,
            side
        )
        se.append((model_value - option["Call"]) ** 2)
    MSE = sum(se) / len(se)
    min_MSE[0] = min(min_MSE[0], MSE)
    if i[0] % 25 == 0:
        print("%4d |" % i[0], np.array(p0), "| %7.3f | %7.3f" % (MSE, min_MSE[0]))
    i[0] += 1
    return MSE


def H93_calibration_full(df_options: pd.DataFrame, S0:float, side: OptionSide):
    """Calibrates Heston (1993) stochastic volatility model to market quotes."""
    # First run with brute force
    # (scan sensible regions, for faster convergence)
    i = [0]
    min_MSE = [float(np.inf)]

    p0 = brute(
        lambda params, data=df_options, s0=S0, option_side=side: H93_error_function(params, data, i, min_MSE, s0, option_side),
        (
            (2.5, 10.6, 5.0),  # kappa_v
            (0.01, 0.041, 0.01),  # theta_v
            (0.05, 0.251, 0.1),  # sigma_v
            (-0.75, 0.01, 0.25),  # rho
            (0.01, 0.031, 0.01),
        ),  # v0
        finish=None,
    )

    # Second run with local, convex minimization
    # (we dig deeper where promising results)
    opt = fmin(
        lambda params, data=df_options, s0=S0, option_side=side: H93_error_function(params, data, i, min_MSE, s0, option_side),
        p0, xtol=0.000001, ftol=0.000001, maxiter=750, maxfun=900
    )
    return opt


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

    print(
        "Heston (1993) Call Option Value:   $%10.4f "
        % H93_eur_option_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, side=OptionSide.CALL)
    )


def ex_calibration(path_str: str = r"./option_dataset.h5"):
    # Market Data from www.eurexchange.com
    # as of September 30, 2014

    h5 = pd.HDFStore(
        path_str, "r"
    )  # Place this file in the same directory before running the code
    data = h5["data"]  # European call & put option data (3 maturities)
    h5.close()
    S0 = 3225.93  # EURO STOXX 50 level September 30, 2014

    # Option Selection
    tol = 0.02  # Tolerance level to select ATM options (percent around ITM/OTM options)
    options = data[(np.abs(data["Strike"] - S0) / S0) < tol]
    options = options.assign(**{dt_key: pd.to_datetime(options[dt_key]) for dt_key in ["Date", "Maturity"]})

    # Adding Time-to-Maturity and constant short-rates
    for row, option in options.iterrows():
        T = (option["Maturity"] - option["Date"]).days / 365.0
        options.loc[row, "T"] = T
        options.loc[row, "r"] = 0.02

    side: OptionSide = OptionSide.CALL
    H93_calibration_full(df_options=options, S0=S0, side=side)


if __name__ == "__main__":
    ex_pricing()
    ex_calibration()