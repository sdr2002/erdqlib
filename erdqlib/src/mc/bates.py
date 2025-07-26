from dataclasses import dataclass
from typing import Tuple, Type, Optional

import numpy as np

from erdqlib.src.common.option import OptionInfo, OptionType, OptionSide
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.heston import HestonDynamicsParameters, HestonSearchGridType, Heston, HestonParameters
from erdqlib.src.mc.jump import JumpOnlyDynamicsParameters, JumpOnlySearchGridType, MertonJump, \
    JumpOnlyParameters
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)

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
    def get_default_heston_search_grid() -> HestonSearchGridType:
        return (
            (2.5, 25.6, 5.0),  # kappa_heston
            (1e-6, 0.041, 0.01),  # theta_heston
            (0.05, 0.251, 0.1),  # sigma_heston
            (-0.75, 0.01, 0.25),  # rho_heston
            (1e-6, 0.031, 0.01)  # v0_heston
        )

    @staticmethod
    def get_default_jumponly_search_grid() -> JumpOnlySearchGridType:
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
            x0=S0,
            r=r,
            kappa_heston=float(opt_arr[0]),
            theta_heston=float(opt_arr[1]),
            sigma_heston=float(opt_arr[2]),
            rho_heston=float(opt_arr[3]),
            v0_heston=float(opt_arr[4]),
            lambd_merton=float(opt_arr[5]),
            mu_merton=float(opt_arr[6]),
            delta_merton=float(opt_arr[7])
        )

    @staticmethod
    def from_dynamic_parameters(
            h_params: HestonDynamicsParameters, j_params: JumpOnlyDynamicsParameters
    ) -> "BatesDynamicsParameters":
        """
        Create BatesDynamicsParameters from Heston and JumpOnly parameters.
        """
        return BatesDynamicsParameters(
            x0=h_params.x0,
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
                x0=self.x0, r=self.r,
                kappa_heston=self.kappa_heston, theta_heston=self.theta_heston,
                sigma_heston=self.sigma_heston, rho_heston=self.rho_heston, v0_heston=self.v0_heston
            ).get_bounded_parameters(),
            j_params=JumpOnlyDynamicsParameters(
                x0=self.x0, r=self.r,
                lambd_merton=self.lambd_merton, mu_merton=self.mu_merton, delta_merton=self.delta_merton
            )
        )


@dataclass
class BatesParameters(HestonParameters, JumpOnlyParameters):

    def to_heston_parameters(self) -> HestonParameters:
        """Convert BatesParameters to HestonParameters."""
        return HestonParameters(
            x0=self.x0, r=self.r,
            kappa_heston=self.kappa_heston, theta_heston=self.theta_heston,
            sigma_heston=self.sigma_heston, rho_heston=self.rho_heston, v0_heston=self.v0_heston,
            T=self.T, M=self.M, I=self.I, random_seed=self.random_seed
        )


class Bates(Heston):

    @staticmethod
    def sample_paths(
            var_arr: np.ndarray,
            b_params: BatesParameters,
            cho_matrix: np.ndarray,
            u1: np.ndarray,
            zj: np.ndarray,
            cj: np.ndarray
    ) -> np.ndarray:
        """Heston model process paths sampler
        dX_t = (r - rj) * X_t * dt + sqrt{v_t} * X_t * dWx_t + J_t(mu, delta) * X_t^{-} * dS(lambda)
          where d<Wx, Wvar>_t = rho * dt, S ~ Poisson(lambda * dt), J ~ exp[N(ln(1+mu) - delta^2/2, delta^2)] - 1
        X_t = X_0 * { exp[(r - rj - 0.5 * v_t) * t + sqrt(v_t) * W1_t] + (exp[mu + delta * Z2_t] - 1) * J_t }
        or
        X_t = X_0 * { exp[(r - rj - 0.5 * v_t) * t + sqrt(v_t) * W1_t] * (1 + (exp[mu + delta * Z2_t] - 1) * J_t) }
        """
        rj: float = b_params.get_r_offset()
        dt: float = b_params.get_dt()
        sdt: float = np.sqrt(dt)

        x_arr2d: np.ndarray = b_params.create_zeros_state_matrix()
        x_arr2d[0] = b_params.x0
        row: int = 0
        for t in range(1, b_params.M + 1, 1):
            drift: np.ndarray = (b_params.r - rj - 0.5 * var_arr[t - 1]) * dt
            z1: np.ndarray = np.dot(cho_matrix, u1[:, t])[row]
            stochastic_diffusion: np.ndarray = np.sqrt(var_arr[t - 1]) * sdt * z1
            jump_diffusion: np.array = (np.exp(b_params.mu_merton + b_params.delta_merton * zj[t]) - 1) * cj[t]
            # TODO are we sure to ADD the jump diffusion instead of multiplying? seeing the result, apparently so.
            # mult: np.array = np.exp(drift + stochastic_diffusion) + jump_diffusion
            mult: np.array = np.exp(drift + stochastic_diffusion) * (1 + jump_diffusion)
            x_arr2d[t] = np.maximum(x_arr2d[t - 1] * mult, MertonJump.X_LOWER_BOUND)
        return x_arr2d

    @staticmethod
    def calculate_paths(
            model_params: BatesParameters, underlying_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        LOGGER.info(str(model_params.__dict__))
        rand_tensor, cho_matrix, zj, cj = Bates.generate_random_numbers(model_params=model_params)

        # Volatility process paths
        v_arr2d: np.ndarray = Bates.sample_variance_paths(
            h_params=model_params.to_heston_parameters(), cho_matrix=cho_matrix, u2=rand_tensor
        )

        # Underlying price process paths
        x_arr2d: np.ndarray = Bates.sample_paths(
            var_arr=v_arr2d, b_params=model_params,
            cho_matrix=cho_matrix, u1=rand_tensor, zj=zj, cj=cj
        )
        if underlying_only:
            return x_arr2d
        return v_arr2d, x_arr2d

    @staticmethod
    def generate_random_numbers(model_params: BatesParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(seed=model_params.random_seed)

        random_normal_arr = np.random.standard_normal((2, model_params.M + 1, model_params.I))
        LOGGER.info(f"rand shape: {random_normal_arr.shape}")

        covariance_matrix = np.array([[1.0, model_params.rho_heston], [model_params.rho_heston, 1.0]])
        covariance_cholesky_lower_arr = np.linalg.cholesky(covariance_matrix)
        LOGGER.info(f"Cov:\n{covariance_matrix}")
        LOGGER.info(f"L:\n{covariance_cholesky_lower_arr}")

        zj = np.random.standard_normal((model_params.M + 1, model_params.I))
        poisson_interval_intensity: float = model_params.get_interval_intensity()
        LOGGER.info(f'  {{lambda*dt={poisson_interval_intensity:.2g}}}')
        cj = np.random.poisson(poisson_interval_intensity, (model_params.M + 1, model_params.I))
        if len(np.where(cj > 0)[0]) == 0:
            LOGGER.warning('  No jump generated')

        return random_normal_arr, covariance_cholesky_lower_arr, zj, cj


def example_bates():
    b_params: BatesParameters = BatesParameters(
        v0_heston=0.04,
        kappa_heston=2,
        sigma_heston=0.3,
        theta_heston=0.04,
        rho_heston=-0.9,

        lambd_merton=0.75,
        mu_merton=0.0,
        delta_merton=0.25,

        x0=100,  # Current underlying asset price
        r=0.05,  # Risk-free rate

        T=1,  # Number of years
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0
    )

    V, S = Bates.calculate_paths(b_params)
    Bates.plot_paths(
        n=300, paths={'x': S, 'var': V}, model_params=b_params, model_name=Bates.__name__
    )

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        d=b_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        d=b_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )}")


if __name__ == "__main__":
    example_bates()