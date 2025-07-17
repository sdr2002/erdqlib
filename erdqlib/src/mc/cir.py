from dataclasses import dataclass

import numpy as np

from erdqlib.src.mc.dynamics import ModelParameters, DynamicsParameters
from erdqlib.src.mc.montecarlo import MonteCarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


@dataclass
class CirDynamicsParameters(DynamicsParameters):
    k: float
    theta: float
    sigma: float


@dataclass
class CirParameters(ModelParameters, CirDynamicsParameters):
    pass


class Cir(MonteCarlo):

    @staticmethod
    def sample_paths(
        v_params: CirParameters, z: np.ndarray
    ) -> np.ndarray:
        """Cox-Ingersoll-Ross process paths sampler
        dX_t = k * (theta - X_t) * dt + sigma * sqrt(X_t) * dW_t
        X_t = X_0 e^{-k*t} + theta * (1 - e^{-k*t}) + sigma * e^{-k*t} * \int_0^t e^{k*s} \sqrt(X_s) dW_s
        """
        dt: float = v_params.get_dt()
        sdt: float = np.sqrt(dt)

        x_arr2d: np.ndarray = v_params.create_zeros_state_matrix()
        for t in range(0, v_params.M + 1):
            if t == 0:
                x_arr2d[0] = v_params.S0
                continue

            dx: np.ndarray = v_params.k * (v_params.theta - x_arr2d[t - 1]) * dt + v_params.sigma * sdt * z[t]
            x_arr2d[t] = x_arr2d[t - 1] + dx

        return x_arr2d

    @staticmethod
    def calculate_paths(model_params: CirParameters, *_, **__) -> np.ndarray:
        """Merton jump process paths sampler"""
        LOGGER.info(str(model_params.__dict__))
        z_arr2d: np.ndarray = MonteCarlo.generate_random_numbers(model_params=model_params)

        x_arr2d: np.array = Cir.sample_paths(model_params, z_arr2d)
        return x_arr2d


def example_vasicek():
    v_params: CirParameters = CirParameters(
        T = 1.0,  # Maturity
        M = 500,  # Number of paths for MC
        I = 10_000,  # Number of steps
        random_seed=0,

        S0 = 0.023,
        k = 0.20,
        theta = 0.01,
        sigma = 0.012,
        r = None,  # Risk-free rate
    )

    rates = Cir.calculate_paths(v_params)
    Cir.plot_paths(n=300, paths={'x': rates}, model_params=v_params, model_name=Cir.__name__, logy=False)


if __name__ == "__main__":
    example_vasicek()