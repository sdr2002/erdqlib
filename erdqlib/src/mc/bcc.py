import json
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from erdqlib.src.common.rate import yield_to_maturity_from_spot_price
from erdqlib.src.mc.bates import BatesDynamicsParameters, BatesParameters
from erdqlib.src.mc.cir import CirDynamicsParameters, CirParameters


def gamma(kappa_r, sigma_r) -> float:
    """
    Gamma function in CIR (1985)
    """
    return np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)


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
        ) ** (2 * kappa_r * theta_r / sigma_r ** 2)

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
class BccDynamicsParameters(CirDynamicsParameters, BatesDynamicsParameters):
    def get_values(self) -> Tuple[
        float, float, float,
        float, float, float, float, float,
        float, float, float
    ]:
        return (
            self.kappa_cir, self.theta_cir, self.sigma_cir,
            self.kappa_heston, self.theta_heston, self.sigma_heston, self.rho_heston, self.v0_heston,
            self.lambd_merton, self.mu_merton, self.delta_merton
        )

    def get_bates_parameters(self) -> BatesDynamicsParameters:
        """Get Bates parameters from BCC dynamics parameters"""
        return BatesDynamicsParameters(
            x0=self.x0, r=self.r,
            kappa_heston=self.kappa_heston,
            theta_heston=self.theta_heston,
            sigma_heston=self.sigma_heston,
            rho_heston=self.rho_heston,
            v0_heston=self.v0_heston,
            lambd_merton=self.lambd_merton,
            mu_merton=self.mu_merton,
            delta_merton=self.delta_merton,
        )


@dataclass
class BCCParameters(CirParameters, BatesParameters):

    def to_str(self, indent: Optional[int] = None):
        return f"BCCParameters{json.dumps(self.__dict__, indent=indent)}"

    def get_B_params(self) -> List[float]:
        """Get parameters for B function"""
        return [self.r, self.kappa_cir, self.theta_cir, self.sigma_cir, 0., self.T]

    def get_pricing_params(self, apply_shortrate: bool) -> List[float]:
        """Get parameters for pricing: apply_shortrate for BCC (1997) model if True, else for Bates (1996) model"""
        r: float = self.r
        if apply_shortrate:
            r = yield_to_maturity_from_spot_price(t_year=self.T, price_0_t=B(self.get_B_params()))

        return [self.x0, self.T, r] + self.get_calibrable_params()

    def get_calibrable_params(self) -> List[float]:
        """Get calibration target parameters"""
        return [
            self.kappa_heston, self.theta_heston, self.sigma_heston, self.rho_heston, self.v0_heston,  # Heston
            self.lambd_merton, self.mu_merton, self.delta_merton  # Merton
        ]

    def get_bates_params(self) -> BatesParameters:
        return BatesParameters(
            x0=self.x0,
            r=yield_to_maturity_from_spot_price(t_year=self.T, price_0_t=B(self.get_B_params())),
            kappa_heston=self.kappa_heston, theta_heston=self.theta_heston, sigma_heston=self.sigma_heston, rho_heston=self.rho_heston, v0_heston=self.v0_heston,
            lambd_merton=self.lambd_merton, mu_merton=self.mu_merton, delta_merton=self.delta_merton,
            T=self.T, M=self.M, I=self.I, random_seed=self.random_seed
        )


class BCC:
    pass
