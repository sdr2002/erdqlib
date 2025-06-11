from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from erdqlib.src.mc.dynamics import ModelParameters
from erdqlib.src.mc.option import price_montecarlo
from erdqlib.tool.logger_util import create_logger
from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType

LOGGER = create_logger(__name__)


@dataclass
class GbmParameters(ModelParameters):
    sigma: float


def gbm_underlying(
    g_params: GbmParameters, z: np.ndarray
) -> np.ndarray:
    dt: float = g_params.get_dt()
    sdt: float = np.sqrt(dt)

    S: np.ndarray = g_params.create_zeros_state_matrix()
    for t in range(0, g_params.M + 1):
        if t == 0:
            S[0] = g_params.S0
            continue

        S[t] = S[t - 1] * np.exp(
            (g_params.r - 0.5 * g_params.sigma**2) * dt
            + g_params.sigma * sdt * z[t]
        )

    return S


def generate_random_numbers(g_params: GbmParameters) -> np.ndarray:
    np.random.seed(seed=g_params.random_seed)

    random_normal_arr = np.random.standard_normal((g_params.M + 1, g_params.I))
    return random_normal_arr


def get_gbm_paths(g_params: GbmParameters) -> np.ndarray:
    """Merton jump process paths sampler"""
    LOGGER.info(str(g_params.__dict__))
    z = generate_random_numbers(g_params)

    S: np.array = gbm_underlying(g_params, z)
    return S


def plot_gbm_paths(n: int, underlying_arr: np.ndarray, g_params: GbmParameters):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 8))

    # Paths of underlying price
    ax0 = axs[0]
    ax0.plot(underlying_arr[:, :n])
    ax0.grid()
    ax0.set_title("GBM Underlying Price Paths")
    ax0.set_xlabel("Timestep")
    ax0.set_ylabel("Price")

    # Distribution of final Log return of underlying price
    logr_last: np.ndarray = np.log(underlying_arr[-1, :] / g_params.S0)
    x = np.linspace(logr_last.min(), logr_last.max(), 500)

    ax1 = axs[1]
    q5 = np.quantile(logr_last, 0.05)
    ax1.hist(
        logr_last, density=True, bins=500,
        label=f"GBM (q1={np.quantile(logr_last, 0.01):.3g}, q5={np.quantile(logr_last, 0.05):.3g},"
              f" sk={ss.skew(logr_last):.3g}, kt={ss.kurtosis(logr_last):.3g})"
    )
    ax1.axvline(x=q5, color='black', linestyle='--')
    ax1.plot(
        x, ss.norm.pdf(x, logr_last.mean(), logr_last.std()),
        color="r", label=f"Normal density (mu={logr_last.mean():.2g}, std={logr_last.std():.2g})"
    )
    ax1.set_xlabel('Log return')
    ax1.legend()
    plt.show()


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

    S = get_gbm_paths(g_params)
    plot_gbm_paths(n=300, underlying_arr=S, g_params=g_params)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        Spath=S,
        d=g_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        Spath=S,
        d=g_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )}")


if __name__ == "__main__":
    example_gbm()