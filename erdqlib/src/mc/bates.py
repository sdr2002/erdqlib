from dataclasses import dataclass

from typing import Tuple, Type, Optional

import numpy as np

from erdqlib.src.mc.dynamics import ModelParameters
from erdqlib.src.mc.heston import HestonDynamicsParameters, HestonSearchGridType
from erdqlib.src.mc.jump import JumpOnlyDynamicsParameters, JumpOnlySearchGridType

BatesSearchGridType: Type = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]

@dataclass
class BatesDynamicsParameters(HestonDynamicsParameters, JumpOnlyDynamicsParameters):
    def get_values(self) -> Tuple[float, float, float, float, float, float, float, float]:
        return (
            self.kappa_heston, self.theta_heston, self.sigma_heston, self.rho_heston, self.v0_heston,
            self.lambd_merton, self.mu_merton, self.delta_merton
        )

    @staticmethod
    def get_heston_default_search_grid() -> HestonSearchGridType:
        return (
            (2.5, 25.6, 5.0),  # kappa_heston
            (1e-6, 0.041, 0.01),  # theta_heston
            (0.05, 0.251, 0.1),  # sigma_heston
            (-0.75, 0.01, 0.25),  # rho_heston
            (1e-6, 0.031, 0.01)  # v0_heston
        )

    @staticmethod
    def get_jumponly_default_search_grid() -> JumpOnlySearchGridType:
        """
        Return the search grid for jump-only parameters.
        This is a subset of the Bates search grid.
        """
        return  (
            (1e-6, 0.51, 0.1),  # lambda
            (-0.5, 1e-6, 0.1),  # mu
            (1e-6, 0.51, 0.1),   # delta
        )

    @staticmethod
    def do_parameters_offbound(
        kappa_v: float, theta_v: float, sigma_v: float, rho: float, v0: float,
        lambd: float, mu: float, delta: float
    ) -> bool:
        is_joff = JumpOnlyDynamicsParameters.do_parameters_offbound(
            lambd=lambd, mu=mu, delta=delta
        )
        is_hoff = HestonDynamicsParameters.do_parameters_offbound(
            kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho=rho, v0=v0
        )
        return is_joff or is_hoff

    @staticmethod
    def from_calibration_output(
            opt_arr: np.array, S0: Optional[float] = None, r: Optional[float] = None
    ) -> "BatesDynamicsParameters":
        return BatesDynamicsParameters(
            S0=S0,
            r=r,
            kappa_heston=opt_arr[0],
            theta_heston=opt_arr[1],
            sigma_heston=opt_arr[2],
            rho_heston=opt_arr[3],
            v0_heston=opt_arr[4],
            lambd_merton=opt_arr[5],
            mu_merton=opt_arr[6],
            delta_merton=opt_arr[7]
        )

    @staticmethod
    def from_dynamic_parameters(
            h_params: HestonDynamicsParameters, j_params: JumpOnlyDynamicsParameters
    ) -> "BatesDynamicsParameters":
        """
        Create BatesDynamicsParameters from Heston and JumpOnly parameters.
        """
        return BatesDynamicsParameters(
            S0=h_params.S0,
            r=h_params.r,
            kappa_heston=h_params.kappa_heston,
            theta_heston=h_params.theta_heston,
            sigma_heston=h_params.sigma_heston,
            rho_heston=h_params.rho_heston,
            v0_heston=h_params.v0_heston,
            lambd_merton=j_params.lambd_merton,
            mu_merton=j_params.mu_merton,
            delta_merton=j_params.delta_merton
        )

    def get_bounded_parameters(self) -> "BatesDynamicsParameters":
        return BatesDynamicsParameters.from_dynamic_parameters(
            h_params=HestonDynamicsParameters(
                S0=self.S0, r=self.r,
                kappa_heston=self.kappa_heston, theta_heston=self.theta_heston,
                sigma_heston=self.sigma_heston, rho_heston=self.rho_heston, v0_heston=self.v0_heston
            ).get_bounded_parameters(),
            j_params=JumpOnlyDynamicsParameters(
                S0=self.S0, r=self.r,
                lambd_merton=self.lambd_merton, mu_merton=self.mu_merton, delta_merton=self.delta_merton
            )
        )


@dataclass
class BatesParameters(ModelParameters, BatesDynamicsParameters):
    pass