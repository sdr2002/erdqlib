from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class SamplingParameters:
    T: float # Number of years
    M: int   # Number time transitions
    I: int   # Number of paths

    random_seed: int # random seed

    def get_sampling_parameters(self) -> Self:
        return SamplingParameters(
            T=self.T, M=self.M, I=self.I, random_seed=self.random_seed
        )

    def get_dt(self) -> float:
        return float(self.T/self.M)

    def get_Myear(self) -> float:
        return 1./self.get_dt()

    def create_zeros_state_matrix(self) -> np.ndarray:
        # Create M state-transition for I paths
        return np.zeros((self.M + 1, self.I), dtype=float)

@dataclass
class DynamicsParameters:
    S0: float # Current underlying asset price
    r: float  # Risk-free rate

    def get_dynamics_parameters(self) -> Self:
        return DynamicsParameters(
            S0=self.S0, r=self.r
        )


@dataclass
class ModelParameters(SamplingParameters, DynamicsParameters):
    pass