from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from erdqlib.src.mc.dynamics import ModelParameters
from erdqlib.src.mc.option import OptionSide, OptionInfo, OptionType, price_montecarlo
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


# Next, let's implement the classic **stochastic equation** for the underlying asset price evolution:
def jump_underlying(
    j_params: JumpParameters, z1: np.ndarray, z2: np.ndarray, y: np.ndarray
) -> np.ndarray:
    # get rate offset
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
        S[t] = np.maximum(S[t], 1e-6)  # To ensure that the price never goes below zero!

    return S


def generate_random_numbers(j_params: JumpParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(j_params.random_seed)

    # Random numbers
    z1 = np.random.standard_normal((j_params.M + 1, j_params.I))
    z2 = np.random.standard_normal((j_params.M + 1, j_params.I))
    
    poisson_interval_intensity: float = j_params.get_interval_intensity()
    LOGGER.info(f'  \u007blambda*dt={poisson_interval_intensity:.2g}\u007d')
    y = np.random.poisson(poisson_interval_intensity, (j_params.M + 1, j_params.I))
    if len(np.where(y > 0)[0]) == 0:
        LOGGER.warning('  No jump generated')

    return z1, z2, y


def get_jump_paths(j_params: JumpParameters) -> np.ndarray:
    """Merton jump process paths sampler"""
    LOGGER.info(str(j_params.__dict__))
    z1, z2, y = generate_random_numbers(j_params)

    return jump_underlying(j_params, z1, z2, y)


def plot_jump_paths(n: int, underlying_arr: np.ndarray, j_params: JumpParameters):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 8))

    # Paths of underlying price
    ax0 = axs[0]
    ax0.plot(underlying_arr[:, :n])
    ax0.grid()
    ax0.set_title("Merton-Jump'76 Underlying Price Paths")
    ax0.set_xlabel("Timestep")
    ax0.set_ylabel("Price")

    # Distribution of final Log return of underlying price
    logr_last: np.ndarray = np.log(underlying_arr[-1, :] / j_params.S0)
    x = np.linspace(logr_last.min(), logr_last.max(), 500)

    ax1 = axs[1]
    q5 = np.quantile(logr_last, 0.05)
    ax1.hist(
        logr_last, density=True, bins=500,
        label=f"Jump (q1={np.quantile(logr_last, 0.01):.3g}, q5={np.quantile(logr_last, 0.05):.3g},"
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

    S = get_jump_paths(j_params)
    plot_jump_paths(n=300, underlying_arr=S, j_params=j_params)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        Spath=S,
        d=j_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        Spath=S,
        d=j_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )}")

