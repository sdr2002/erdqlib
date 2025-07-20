from dataclasses import dataclass

from typing import Tuple

from erdqlib.src.mc.heston import HestonDynamicsParameters
from erdqlib.src.mc.jump import JumpOnlyDynamicsParameters


@dataclass
class BatesDynamicsParameters(HestonDynamicsParameters, JumpOnlyDynamicsParameters):
    def get_values(self) -> Tuple[float, float, float, float, float, float, float, float]:
        return (
            self.kappa_heston, self.theta_heston, self.sigma_heston, self.rho_heston, self.v0_heston,
            self.lambd_merton, self.mu_merton, self.delta_merton
        )