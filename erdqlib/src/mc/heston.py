from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType
from erdqlib.src.mc.dynamics import ModelParameters
from erdqlib.src.mc.option import price_montecarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


@dataclass
class HestonParameters(ModelParameters):
    v0: float
    kappa: float
    sigma: float
    theta: float
    rho: float

def heston_volatility(h_params: HestonParameters, cho_matrix: np.ndarray, rand: np.ndarray) -> np.ndarray:
    """Stochastic variance process for Heston model"""
    # v_t = v_{t-1} + \kappa (\theta - v_{t-1})dt + \sigma \sqrt{v_{t-1}} dW^{(2)})t
    v: np.ndarray = h_params.create_zeros_state_matrix()

    dt: float = h_params.get_dt()
    sdt: float = np.sqrt(dt)  # Sqrt of dt

    row: int = 1
    for t in range(0, h_params.M + 1):
        if t == 0:
            v[0] = h_params.v0
            continue
        ran = np.dot(cho_matrix, rand[:, t])[row]
        next_v = v[t - 1] + h_params.kappa * (h_params.theta - v[t - 1]) * dt + np.sqrt(v[t - 1]) * h_params.sigma * ran * sdt
        v[t] = np.maximum(0, next_v) # manual non-negative bound
    return v


# Next, let's implement the classic **stochastic equation** for the underlying asset price evolution:
def heston_underlying(
    var_arr: np.ndarray, h_params: HestonParameters, cho_matrix: np.ndarray, rand: np.ndarray
) -> np.ndarray:
    S: np.ndarray = h_params.create_zeros_state_matrix()

    dt: float = h_params.get_dt()
    sdt: float = np.sqrt(dt)

    row: int = 1
    for t in range(0, h_params.M + 1, 1):
        if t == 0:
            S[0] = h_params.S0
            continue
        ran = np.dot(cho_matrix, rand[:, t])[row]
        S[t] = S[t - 1] * np.exp((h_params.r - 0.5 * var_arr[t - 1]) * dt + np.sqrt(var_arr[t - 1]) * ran * sdt)

    return S


def generate_random_numbers(h_params: HestonParameters) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed=h_params.random_seed)

    random_normal_arr = np.random.standard_normal((2, h_params.M + 1, h_params.I))
    LOGGER.debug(f"rand shape: {random_normal_arr.shape}")

    covariance_matrix = np.array([[1.0, h_params.rho], [h_params.rho, 1.0]])
    covariance_cholesky_lower_arr = np.linalg.cholesky(covariance_matrix)
    LOGGER.debug(f"Cov:\n{covariance_matrix}")
    LOGGER.debug(f"L:\n{covariance_cholesky_lower_arr}")

    return random_normal_arr, covariance_cholesky_lower_arr


def get_heston_paths(h_params: HestonParameters) -> Tuple[np.ndarray, np.ndarray]:
    LOGGER.info(str(h_params.__dict__))

    # Gerate the source of randomness
    rand_tensor, cho_matrix = generate_random_numbers(h_params)

    # Volatility process paths
    Var = heston_volatility(h_params, cho_matrix=cho_matrix, rand=rand_tensor)

    # Underlying price process paths
    S: np.ndarray = heston_underlying(
        var_arr=Var, h_params=h_params, cho_matrix=cho_matrix, rand=rand_tensor
    )
    return Var, S


def plot_heston_paths(n: int, underlying_arr: np.ndarray, variance_arr: np.ndarray, h_params: HestonParameters):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

    # Paths of underlying price
    ax0 = axs[0,0]
    ax0.plot(range(len(underlying_arr)), underlying_arr[:, :n])
    ax0.grid()
    ax0.set_title("Heston Underlying Price paths")
    ax0.set_xlabel("Timestep")
    ax0.set_ylabel("Price")

    # Distribution of final Log return of underlying price
    ax2 = axs[1,0]

    logr_last: np.ndarray = np.log(underlying_arr[-1, :] / h_params.S0)
    q5 = np.quantile(logr_last, 0.05)
    ax2.hist(
        logr_last, density=True, bins=500,
        label=f"Heston (q1={np.quantile(logr_last, 0.01):.3g}, q5={np.quantile(logr_last, 0.05):.3g},"
              f" sk={ss.skew(logr_last):.3g}, kt={ss.kurtosis(logr_last):.3g})"
    )
    ax2.axvline(x=q5, color='black', linestyle='--')
    x_logr = np.linspace(logr_last.min(), logr_last.max(), 500)
    ax2.plot(
        x_logr, ss.norm.pdf(x_logr, logr_last.mean(), logr_last.std()),
        color="r", label=f"Normal density (mu={logr_last.mean():.2g}, std={logr_last.std():.2g})"
    )
    ax2.set_xlabel('Log return')
    ax2.legend()

    # Paths of variance of underlying price
    ax1 = axs[0,1]
    ax1.plot(range(len(variance_arr)), variance_arr[:, :n])
    ax1.grid()
    ax1.set_title("Heston Variance paths")
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Timestep")

    ax3 = axs[1,1]
    var_last = variance_arr[-1, :]
    ax3.hist(
        var_last, density=True, bins=500
    )
    ax3.axvline(x=h_params.sigma**2, color='black', linestyle='--', label='sigma^2')
    x_var: np.ndarray = np.linspace(var_last.min(), var_last.max(), 500)
    ax3.plot(
        x_var, ss.lognorm.pdf(x_var, *ss.lognorm.fit(var_last, floc=0)),
        color="r", label=f"LogNormal density"
    )
    ax3.set_xlabel('Variance')
    ax3.legend()

    plt.show()


if __name__ == "__main__":
    # Now we have all the ingredients to generate the paths for both asset price and its volatility:

    h_params: HestonParameters = HestonParameters(
        v0 = 0.04,
        kappa = 2,
        sigma = 0.3,
        theta = 0.04,
        rho = -0.9,

        S0 = 100,  # Current underlying asset price
        r = 0.05,  # Risk-free rate

        T = 1,  # Number of years
        M = 500,  # Total time steps
        I = 10000,  # Number of simulations
        random_seed=0
    )

    V, S = get_heston_paths(h_params)
    plot_heston_paths(n=300, underlying_arr=S, variance_arr=V, h_params=h_params)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        Spath=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        Spath=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )}")

