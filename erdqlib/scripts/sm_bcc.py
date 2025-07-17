import json
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from typing import List, Optional

from erdqlib.src.common.rate import instantaneous_rate
from erdqlib.scripts.sm_bates import M76J_char_func, B96_eur_option_value_lewis
from erdqlib.scripts.sm_heston import H93_char_func
from erdqlib.src.common.option import OptionSide
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


# BCC (1997) characteristic function (H93+M76)
def BCC_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """
    BCC (1997) characteristic function
    """
    H93 = H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    M76J = M76J_char_func(u, T, lamb, mu, delta)
    return H93 * M76J


# Lewis (2001) integral value of BCC (1997)
def BCC_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """
    Lewis (2001) integral value for BCC (1997) characteristic function
    """
    char_func_value = BCC_char_func(
        u - 1j * 0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
    )
    int_func_value = (
        1 / (u**2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    )
    return int_func_value


def BCC_eur_option_value_lewis(
        S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, side: OptionSide
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
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    ==========
    """
    int_value = quad(
        lambda u: BCC_int_func(
            u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
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


def gamma(kappa_r, sigma_r) -> float:
    """
    Gamma function in CIR (1985)
    """
    return np.sqrt(kappa_r**2 + 2 * sigma_r**2)

def b1(alpha) -> float:
    """
    b1 function in CIR (1985)
    alpha is the parameter set
    """
    r0, kappa_r, theta_r, sigma_r, t, T = alpha
    g = gamma(kappa_r, sigma_r)
    x = (
        (2 * g * np.exp((kappa_r + g) * (T - t) / 2))
        / (2 * g + (kappa_r + g) * (np.exp(g * (T - t)) - 1))
    ) ** (2 * kappa_r * theta_r / sigma_r**2)

    return x


def b2(alpha):
    """
    b2 function in CIR (1985)
    alpha is the parameter set
    """
    r0, kappa_r, theta_r, sigma_r, t, T = alpha
    g = gamma(kappa_r, sigma_r)
    x = (2 * (np.exp(g * (T - t)) - 1)) / (
        2 * g + (kappa_r + g) * (np.exp(g * (T - t)) - 1)
    )

    return x


def B(alpha):
    """
    ZCB prices in the CIR (1985) model
    """
    b_1 = b1(alpha)
    b_2 = b2(alpha)
    r0, kappa_r, theta_r, sigma_r, t, T = alpha

    E_rt = theta_r + np.exp(-kappa_r * t) * (r0 - theta_r)

    zcb = b_1 * np.exp(-b_2 * E_rt)

    return zcb


@dataclass
class BCCParameters:
    # Option information
    S0: float
    K: float
    Ti: float
    Tf: float
    # CIR
    r0: float
    kappa_r: float
    theta_r: float
    sigma_r: float
    # SV
    v0: float
    kappa_v: float
    theta_v: float
    sigma_v: float
    rho: float
    # JD
    lamb: float
    mu: float
    delta: float

    def to_str(self, indent: Optional[int] = None):
        return f"BCCParameters{json.dumps(self.__dict__, indent=indent)}"

    def __str__(self, indent: Optional[int] = None) -> str:
        return self.to_str()

    def get_B_params(self) -> List[float]:
        """Get parameters for B function"""
        return [self.r0, self.kappa_r, self.theta_r, self.sigma_r, self.Ti, self.Tf]

    def get_pricing_params(self, apply_shortrate: bool) -> List[float]:
        """Get parameters for BCC pricing"""
        r: float = self.r0
        if apply_shortrate:
            r = instantaneous_rate(B(self.get_B_params()), self.Tf)

        return [
            self.S0, self.K, self.Tf, r,
            self.kappa_v, self.theta_v, self.sigma_v, self.rho,
            self.v0, self.lamb, self.mu, self.delta
        ]


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
        lamb=0.008,
        mu=-0.600,
        delta=0.001
    )

    side: OptionSide = OptionSide.CALL
    Bates_px: float = B96_eur_option_value_lewis(
        *bcc_params.get_pricing_params(apply_shortrate=False), side
    )
    LOGGER.info(f"Option value under Bates: {Bates_px}")

    BCC_call: float = BCC_eur_option_value_lewis(
        *bcc_params.get_pricing_params(apply_shortrate=True), side
    )  # type: ignore
    LOGGER.info(f"{side} Option value under BCC (1997): {BCC_call}")


if __name__ == "__main__":
    ex_pricing()