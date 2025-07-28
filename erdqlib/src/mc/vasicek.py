from dataclasses import dataclass

import numpy as np

from erdqlib.src.mc.dynamics import ModelParameters, DynamicsParameters
from erdqlib.src.mc.dynamics import MonteCarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


@dataclass
class VasicekDynamicsParameters(DynamicsParameters):
    kappa_vasicek: float
    theta_vasicek: float
    sigma_vasicek: float


@dataclass
class VasicekParameters(ModelParameters, VasicekDynamicsParameters):
    pass


class Vasicek(MonteCarlo):

    @staticmethod
    def sample_paths(
            v_params: VasicekParameters, z: np.ndarray
    ) -> np.ndarray:
        """Vasicek process paths sampler
        dX_t = k * (theta - X_t) * dt + sigma * dW_t
        X_t = X_0 e^{-k*t} + theta * (1 - e^{-k*t}) + sigma * e^{-k*t} * integral_0^t e^{k*s} dW_s
        """
        dt: float = v_params.get_dt()

        x_arr2d: np.ndarray = v_params.create_zeros_state_matrix()
        for t in range(0, v_params.M + 1):
            if t == 0:
                x_arr2d[0] = v_params.x0
                continue

            x_arr2d[t] = x_arr2d[t - 1] * np.exp(-v_params.kappa_vasicek * dt) + v_params.theta_vasicek * (1 - np.exp(-v_params.kappa_vasicek * dt)) + \
                         v_params.sigma_vasicek * np.sqrt((1 - np.exp(-2 * v_params.kappa_vasicek * dt)) / (2 * v_params.kappa_vasicek)) * z[t]

        return x_arr2d

    @staticmethod
    def calculate_paths(model_params: VasicekParameters, *_, **__) -> np.ndarray:
        """Merton jump process paths sampler"""
        LOGGER.info(str(model_params.__dict__))
        z_arr2d: np.ndarray = MonteCarlo.generate_random_numbers(model_params=model_params)

        x_arr2d: np.ndarray = Vasicek.sample_paths(model_params, z_arr2d)
        return x_arr2d
