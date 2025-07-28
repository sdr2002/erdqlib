from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from erdqlib.src.mc.dynamics import ModelParameters, DynamicsParameters
from erdqlib.src.mc.dynamics import MonteCarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)

@dataclass
class GbmDynamicsParameters(DynamicsParameters):
    sigma: float

    @staticmethod
    def do_parameters_offbound(sigma: float, *_, **__) -> bool:
        return sigma <= 0.0

    @staticmethod
    def from_calibration_output(
        opt_arr: np.ndarray,
        s0: Optional[float] = None, r: Optional[float] = None,
        *_, **__
    ) -> "GbmDynamicsParameters":
        return GbmDynamicsParameters(
            x0=s0,
            r=r,
            sigma=float(opt_arr[0])
        )

    def get_values(self) -> Tuple[float]:
        return (self.sigma,)


@dataclass
class GbmParameters(ModelParameters, GbmDynamicsParameters):
    pass


class Gbm(MonteCarlo):

    @staticmethod
    def sample_paths(g_params: GbmParameters, z: np.ndarray) -> np.ndarray:
        """ Geometric Brownian motion process paths sampler
        dX_t = r * X_t * dt + sigma * X_t * dW_t
        X_t = X_0 * exp((r - 0.5 * sigma^2) * t + sigma * sqrt(t) * Z)
        """
        dt: float = g_params.get_dt()
        sdt: float = np.sqrt(dt)
        x_arr2d: np.ndarray = g_params.create_zeros_state_matrix()
        for t in range(0, g_params.M + 1):
            if t == 0:
                x_arr2d[0] = g_params.x0
                continue
            x_arr2d[t] = x_arr2d[t - 1] * np.exp(
                (g_params.r - 0.5 * g_params.sigma ** 2) * dt
                + g_params.sigma * sdt * z[t]
            )
        return x_arr2d

    @staticmethod
    def calculate_paths(model_params: GbmParameters, *_, **__) -> np.ndarray:
        LOGGER.info(str(model_params.__dict__))
        np.random.seed(seed=model_params.random_seed)
        z_arr2d = MonteCarlo.generate_random_numbers(model_params=model_params)
        x_arr2d: np.ndarray = Gbm.sample_paths(model_params, z_arr2d)
        return x_arr2d
