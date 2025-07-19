from dataclasses import dataclass

import numpy as np

from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType
from erdqlib.src.mc.dynamics import ModelParameters
from erdqlib.src.mc.dynamics import MonteCarlo
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


@dataclass
class GbmParameters(ModelParameters):
    sigma: float


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
                x_arr2d[0] = g_params.S0
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


def example_gbm():
    g_params: GbmParameters = GbmParameters(
        sigma=0.2,

        S0=100,  # Current underlying asset price
        r=0.05,  # Risk-free rate

        T=1,  # Number of years
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0
    )

    S = Gbm.calculate_paths(g_params)
    Gbm.plot_paths(n=300, paths={'x': S}, model_params=g_params, model_name=Gbm.__name__, logy=True)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        d=g_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        d=g_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )}")


if __name__ == "__main__":
    example_gbm()
