from dataclasses import dataclass
from typing import Tuple

import numpy as np

from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType
from erdqlib.src.mc.dynamics import ModelParameters
from erdqlib.src.mc.mc import MonteCarlo
from erdqlib.src.mc.option import price_montecarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


@dataclass
class JumpParameters(ModelParameters):
    lambd: float
    mu: float
    delta: float
    sigma: float
    
    def get_interval_intensity(self) -> float:
        return self.lambd * self.get_dt()

    def get_r_offset(self) -> float:
        # Get rj, the risk-free rate offset for arbitrage-free model requirement
        rj = self.lambd * np.exp(self.mu + 0.5 * self.delta ** 2 - 1.0)
        LOGGER.info(f"  \u007brj={rj:.3g}\u007d")
        return rj


class MertonJump(MonteCarlo):

    @staticmethod
    def sample_paths(j_params: JumpParameters, z1: np.ndarray, z2: np.ndarray, y: np.ndarray) -> np.ndarray:
        rj: float = j_params.get_r_offset()
        dt: float = j_params.get_dt()
        sdt: float = np.sqrt(dt)
        S: np.ndarray = j_params.create_zeros_state_matrix()
        for t in range(0, j_params.M + 1):
            if t == 0:
                S[0] = j_params.S0
                continue
            mult = np.exp((j_params.r - rj - 0.5 * j_params.sigma ** 2) * dt + j_params.sigma * sdt * z1[t])
            mult += (np.exp(j_params.mu + j_params.delta * z2[t]) - 1) * y[t]
            S[t] = S[t - 1] * mult
            S[t] = np.maximum(S[t], 1e-6)
        return S

    @staticmethod
    def calculate_paths(model_params: JumpParameters) -> np.ndarray:
        LOGGER.info(str(model_params.__dict__))
        z1, z2, y = MertonJump.generate_random_numbers(model_params)
        return MertonJump.sample_paths(model_params, z1, z2, y)

    @staticmethod
    def generate_random_numbers(model_params: JumpParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(model_params.random_seed)
        z1 = np.random.standard_normal((model_params.M + 1, model_params.I))
        z2 = np.random.standard_normal((model_params.M + 1, model_params.I))
        poisson_interval_intensity: float = model_params.get_interval_intensity()
        LOGGER.info(f'  {{lambda*dt={poisson_interval_intensity:.2g}}}')
        y = np.random.poisson(poisson_interval_intensity, (model_params.M + 1, model_params.I))
        if len(np.where(y > 0)[0]) == 0:
            LOGGER.warning('  No jump generated')
        return z1, z2, y


if __name__ == "__main__":
    j_params: JumpParameters = JumpParameters(
        lambd = 0.75,  # Lambda of the model
        mu = -0.6,  # Mu
        delta = 0.25,  # Delta
        sigma = 0.2,

        S0 = 100,  # Current underlying asset price
        r = 0.05,  # Risk-free rate

        T = 1,  # Number of years
        M = 500,  # Total time steps
        I = 10000,  # Number of simulations
        random_seed=0
    )

    S = MertonJump.calculate_paths(j_params)
    MertonJump.plot_paths(n=300, paths=S, model_params=j_params, model_name=MertonJump.__name__, logy=True)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        d=j_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        d=j_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )}")

