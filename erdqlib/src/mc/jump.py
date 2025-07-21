from dataclasses import dataclass
from typing import Tuple, Type, Optional

import numpy as np

from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType
from erdqlib.src.mc.dynamics import ModelParameters, DynamicsParameters
from erdqlib.src.mc.dynamics import MonteCarlo
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)

JumpOnlySearchGridType: Type = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float]
]

JumpSearchGridType: Type = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float]
]


@dataclass
class JumpOnlyDynamicsParameters(DynamicsParameters):
    lambd_merton: float
    mu_merton: float
    delta_merton: float

    @staticmethod
    def from_calibration_output(opt_arr: np.array, S0: Optional[float] = None, r: Optional[float] = None) -> "JumpOnlyDynamicsParameters":
        """Create JumpOnlyDynamicsParameters from an array of parameters."""
        if len(opt_arr) != 3:
            raise ValueError(f"Expected 3 parameters, got {len(opt_arr)}")
        return JumpOnlyDynamicsParameters(
            S0=S0,
            r=r,
            lambd_merton=opt_arr[0],
            mu_merton=opt_arr[1],
            delta_merton=opt_arr[2],
        )

    @staticmethod
    def do_parameters_offbound(lambd, mu, delta) -> bool:
        return lambd < 0.0 or mu < -0.6 or mu > 0.0 or delta < 0.0

    def get_values(self) -> Tuple[float, float, float]:
        return self.lambd_merton, self.mu_merton, self.delta_merton

    @staticmethod
    def get_default_search_grid() -> JumpOnlySearchGridType:
        """Default search grid for jump parameters."""
        return (
            (0.10, 0.401, 0.1),  # lambda
            (-0.5, 0.01, 0.1),  # mu
            (0.10, 0.301, 0.1),  # delta
        )


@dataclass
class JumpDynamicsParameters(JumpOnlyDynamicsParameters):
    sigma_merton: float

    def get_values(self) -> Tuple[float, float, float, float]:
        return self.lambd_merton, self.mu_merton, self.delta_merton, self.sigma_merton

    @staticmethod
    def from_calibration_output(opt_arr: np.array, S0: Optional[float] = None, r: Optional[float] = None) -> "JumpDynamicsParameters":
        """Create JumpDynamicsParameters from an array of parameters."""
        if len(opt_arr) != 4:
            raise ValueError(f"Expected 4 parameters, got {len(opt_arr)}")
        return JumpDynamicsParameters(
            S0=S0,
            r=r,
            lambd_merton=float(opt_arr[0]),
            mu_merton=float(opt_arr[1]),
            delta_merton=float(opt_arr[2]),
            sigma_merton=float(opt_arr[3]),
        )

    @staticmethod
    def get_default_search_grid() -> JumpSearchGridType:
        """Default search grid for jump parameters."""
        return (
            (0.10, 0.401, 0.1),  # lambda
            (-0.5, 0.01, 0.1),  # mu
            (0.10, 0.301, 0.1),  # delta
            (0.075, 0.201, 0.025),  # sigma
        )

    @staticmethod
    def do_parameters_offbound(sigma: float, lambd: float, mu: float, delta: float) -> bool:
        return sigma < 0.0 or delta < 0.0 or lambd < 0.0


@dataclass
class JumpParameters(ModelParameters, JumpDynamicsParameters):
    
    def get_interval_intensity(self) -> float:
        return self.lambd_merton * self.get_dt()

    def get_r_offset(self) -> float:
        # Get rj, the risk-free rate offset for arbitrage-free model requirement
        rj = self.lambd_merton * np.exp(self.mu_merton + 0.5 * self.delta_merton ** 2 - 1.0)
        LOGGER.info(f"  \u007brj={rj:.3g}\u007d")
        return rj


class MertonJump(MonteCarlo):

    @staticmethod
    def sample_paths(j_params: JumpParameters, z1: np.ndarray, z2: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Merton jump process paths sampler
        dX_t = (r - rj - 0.5 * sigma^2) * X_t * dt + sigma * X_t * dW1_t + X_t^{-} * (N2(mu, delta^2) - 1) * dJ_t(lambda)
            where rj = lambda * (exp(mu + 0.5 * delta^2) - 1)
        X_t = X_0 * exp((r - rj - 0.5 * sigma^2) * t + sigma * sqrt(t) * Z1 + (exp(mu + delta * Z2) - 1) * J_t)
        """
        rj: float = j_params.get_r_offset()
        dt: float = j_params.get_dt()
        sdt: float = np.sqrt(dt)
        S: np.ndarray = j_params.create_zeros_state_matrix()
        for t in range(0, j_params.M + 1):
            if t == 0:
                S[0] = j_params.S0
                continue
            mult = np.exp((j_params.r - rj - 0.5 * j_params.sigma_merton ** 2) * dt + j_params.sigma_merton * sdt * z1[t])
            mult += (np.exp(j_params.mu_merton + j_params.delta_merton * z2[t]) - 1) * j[t]
            S[t] = S[t - 1] * mult
            S[t] = np.maximum(S[t], 1e-6)
        return S

    @staticmethod
    def calculate_paths(model_params: JumpParameters, *_, **__) -> np.ndarray:
        LOGGER.info(str(model_params.__dict__))
        z1, z2, j = MertonJump.generate_random_numbers(model_params)
        return MertonJump.sample_paths(model_params, z1, z2, j)

    @staticmethod
    def generate_random_numbers(model_params: JumpParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(model_params.random_seed)
        z1 = np.random.standard_normal((model_params.M + 1, model_params.I))
        z2 = np.random.standard_normal((model_params.M + 1, model_params.I))
        poisson_interval_intensity: float = model_params.get_interval_intensity()
        LOGGER.info(f'  {{lambda*dt={poisson_interval_intensity:.2g}}}')
        j = np.random.poisson(poisson_interval_intensity, (model_params.M + 1, model_params.I))
        if len(np.where(j > 0)[0]) == 0:
            LOGGER.warning('  No jump generated')
        return z1, z2, j


if __name__ == "__main__":
    j_params: JumpParameters = JumpParameters(
        lambd_merton= 0.75,  # Lambda of the model
        mu_merton= -0.6,  # Mu
        delta_merton= 0.25,  # Delta
        sigma_merton= 0.2,

        S0 = 100,  # Current underlying asset price
        r = 0.05,  # Risk-free rate

        T = 1,  # Number of years
        M = 500,  # Total time steps
        I = 10000,  # Number of simulations
        random_seed=0
    )

    S = MertonJump.calculate_paths(j_params)
    MertonJump.plot_paths(n=300, paths={'x': S}, model_params=j_params, model_name=MertonJump.__name__, logy=True)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        d=j_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        d=j_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )}")

