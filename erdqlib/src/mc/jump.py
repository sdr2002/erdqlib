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
            x0=S0,
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
            x0=S0,
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


def get_interval_intensity(lambd_merton, dt) -> float:
    return lambd_merton * dt


def get_r_offset(lambd_merton, mu_merton, delta_merton) -> float:
    # Get rj, the risk-free rate offset for arbitrage-free model requirement
    rj = lambd_merton * np.exp(mu_merton + 0.5 * delta_merton ** 2 - 1.0)
    LOGGER.info(f"  \u007brj={rj:.3g}\u007d")
    return rj


@dataclass
class JumpOnlyParameters(ModelParameters, JumpOnlyDynamicsParameters):

    def get_interval_intensity(self) -> float:
        return get_interval_intensity(
            lambd_merton=self.lambd_merton,
            dt=self.get_dt()
        )

    def get_r_offset(self) -> float:
        return get_r_offset(
            lambd_merton=self.lambd_merton,
            mu_merton=self.mu_merton,
            delta_merton=self.delta_merton
        )


@dataclass
class JumpParameters(ModelParameters, JumpDynamicsParameters):

    def get_interval_intensity(self) -> float:
        return get_interval_intensity(
            lambd_merton=self.lambd_merton,
            dt=self.get_dt()
        )

    def get_r_offset(self) -> float:
        return get_r_offset(
            lambd_merton=self.lambd_merton,
            mu_merton=self.mu_merton,
            delta_merton=self.delta_merton
        )


class MertonJump(MonteCarlo):
    X_LOWER_BOUND: float = 1e-6  # Lower bound for asset price to avoid negative path

    @staticmethod
    def sample_paths(
            j_params: JumpParameters, z1: np.ndarray, zj: np.ndarray, cj: np.ndarray
    ) -> np.ndarray:
        """Merton jump process paths sampler
        dX_t = (r - rj - 0.5 * sigma^2) * X_t * dt + sigma * X_t * dW1_t + X_t^{-} * (N2(mu, delta^2) - 1) * dJ_t(lambda)
            where rj = lambda * (exp(mu + 0.5 * delta^2) - 1)
        X_t = X_0 * { exp[(r - rj - 0.5 * sigma^2) * t + sigma * W1_t] + (exp(mu + delta * W2_t) - 1) * J_t }
        or
        X_t = X_0 * { exp[(r - rj - 0.5 * sigma^2) * t + sigma * W1_t] * (1 + (exp(mu + delta * W2_t) - 1) * J_t) }
        """
        rj: float = j_params.get_r_offset()
        dt: float = j_params.get_dt()
        sdt: float = np.sqrt(dt)

        x_arr2d: np.ndarray = j_params.create_zeros_state_matrix()
        x_arr2d[0] = j_params.x0
        for t in range(1, j_params.M + 1):
            drift: float = (j_params.r - rj - 0.5 * j_params.sigma_merton ** 2) * dt
            weiner_diffusion: np.ndarray = j_params.sigma_merton * sdt * z1[t]
            jump_diffusion: np.array = (np.exp(j_params.mu_merton + j_params.delta_merton * zj[t]) - 1) * cj[t]
            # TODO are we sure to ADD the jump diffusion instead of multiplying? seeing the result, apparently so.
            # mult: np.array = np.exp(drift + weiner_diffusion) + jump_diffusion
            mult: np.array = np.exp(drift + weiner_diffusion) * (1 + jump_diffusion)
            x_arr2d[t] = np.maximum(x_arr2d[t - 1] * mult, MertonJump.X_LOWER_BOUND)
        return x_arr2d

    @staticmethod
    def calculate_paths(model_params: JumpParameters, *_, **__) -> np.ndarray:
        LOGGER.info(str(model_params.__dict__))
        z1, zj, cj = MertonJump.generate_random_numbers(model_params)
        return MertonJump.sample_paths(
            j_params=model_params,
            z1=z1, zj=zj, cj=cj
        )

    @staticmethod
    def generate_random_numbers(model_params: JumpParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random numbers for Merton jump process.

        :param model_params: JumpParameters containing model parameters.
        :return: Tuple of arrays (z1, zj, cj) where:
            - z1: Standard normal random numbers for the first Wiener process.
            - zj: Standard normal random numbers for the second Wiener process.
            - cj: Poisson random numbers representing the number of jumps.
        """
        np.random.seed(model_params.random_seed)
        z1 = np.random.standard_normal((model_params.M + 1, model_params.I))
        zj = np.random.standard_normal((model_params.M + 1, model_params.I))
        poisson_interval_intensity: float = model_params.get_interval_intensity()
        LOGGER.info(f'  {{lambda*dt={poisson_interval_intensity:.2g}}}')
        cj = np.random.poisson(poisson_interval_intensity, (model_params.M + 1, model_params.I))
        if len(np.where(cj > 0)[0]) == 0:
            LOGGER.warning('  No jump generated')
        return z1, zj, cj


if __name__ == "__main__":
    j_params: JumpParameters = JumpParameters(
        lambd_merton= 0.75,  # Lambda of the model
        mu_merton= 0.0,  # Mu
        delta_merton= 0.25,  # Delta
        sigma_merton= 0.2,

        x0= 100,  # Current underlying asset price
        r = 0.05,  # Risk-free rate

        T = 1,  # Number of years
        M = 500,  # Total time steps
        I = 10_000,  # Number of simulations
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

