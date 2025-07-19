from dataclasses import dataclass
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType
from erdqlib.src.mc.dynamics import ModelParameters, DynamicsParameters
from erdqlib.src.mc.dynamics import MonteCarlo
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


@dataclass
class HestonDynamicsParameters(DynamicsParameters):
    v0: float
    kappa: float
    sigma: float
    theta: float
    rho: float

    @staticmethod
    def get_default_search_grid() -> Dict[str, Tuple[float, float, float]]:
        eps: float = 1e-6
        return {
            "kappa": (2.5, 30. + eps, 5.0),
            "theta": (0.001, 0.041 + eps, 0.01),
            "sigma": (0.01, 0.251 + eps, 0.5),
            "rho": (-0.9, 0.9 + eps, 0.2),
            "v0": (0.01, 0.031 + eps, 0.01)
        }

    @staticmethod
    def do_parameters_offbound(kappa_v, theta_v, sigma_v, rho, v0) -> bool:
        return (
                kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0
                or rho < -1.0 or rho > 1.0 or v0 < 0.0
                or 2 * kappa_v * theta_v < sigma_v ** 2
        )

    def get_bounded_parameters(self) -> "HestonDynamicsParameters":
        """Return the parameters that are bounded and can be used for optimization."""
        eps: float = 1e-6
        return HestonDynamicsParameters(
            S0=self.S0, r=self.r,
            kappa=np.clip(self.kappa, eps, 0.5 * self.sigma ** 2 / self.theta),  # kappa must be positive
            sigma=np.clip(self.sigma, eps, 1. - eps),  # sigma must be positive
            theta=np.clip(self.theta, 0.005 + eps, 1. - eps),  # theta must be positive
            rho=np.clip(self.rho, -1. + eps, 1 - eps),  # rho must be in (-1, 1)
            v0=np.clip(self.v0, eps, 1. - eps),  # v0 must be positive
        )

    def get_values(self) -> Tuple[float, float, float, float, float]:
        return self.kappa, self.theta, self.sigma, self.rho, self.v0


@dataclass
class HestonParameters(ModelParameters, HestonDynamicsParameters):
    pass


class Heston(MonteCarlo):

    @staticmethod
    def sample_variance_paths(h_params: HestonParameters, cho_matrix: np.ndarray, rand: np.ndarray) -> np.ndarray:
        """Stochastic variance process for Heston model
        dv_t = kappa * (theta - v_t) dt + sigma * sqrt{v_t} dW^{(2)}_t
        v_t = v_0 * exp(-kappa * t) + theta * (1 - exp(-kappa * t)) + sigma * exp(-kappa * t) * integral_0^t exp(kappa * s) * sqrt(v_s) dW2_s
        """
        v_arr2d: np.ndarray = h_params.create_zeros_state_matrix()

        dt: float = h_params.get_dt()
        sdt: float = np.sqrt(dt)  # Sqrt of dt

        row: int = 1
        for t in range(0, h_params.M + 1):
            if t == 0:
                v_arr2d[0] = h_params.v0
                continue
            ran = np.dot(cho_matrix, rand[:, t])[row]
            next_v = v_arr2d[t - 1] + h_params.kappa * (h_params.theta - v_arr2d[t - 1]) * dt + np.sqrt(
                v_arr2d[t - 1]) * h_params.sigma * ran * sdt
            v_arr2d[t] = np.maximum(0, next_v)  # manual non-negative bound
        return v_arr2d

    @staticmethod
    def sample_paths(
            var_arr: np.ndarray, h_params: HestonParameters, cho_matrix: np.ndarray, rand: np.ndarray
    ) -> np.ndarray:
        """Heston model process paths sampler
        dX_t = r * X_t * dt + sqrt{v_t} * X_t * dW1_t where d<W1, dW2> = rho * dt
        X_t = X_0 * exp((r - 0.5 * v_t) * t + sqrt{v_t} * Z)
        """
        x_arr2d: np.ndarray = h_params.create_zeros_state_matrix()

        dt: float = h_params.get_dt()
        sdt: float = np.sqrt(dt)

        row: int = 1
        for t in range(0, h_params.M + 1, 1):
            if t == 0:
                x_arr2d[0] = h_params.S0
                continue
            ran = np.dot(cho_matrix, rand[:, t])[row]
            x_arr2d[t] = x_arr2d[t - 1] * np.exp(
                (h_params.r - 0.5 * var_arr[t - 1]) * dt + np.sqrt(var_arr[t - 1]) * ran * sdt)

        return x_arr2d

    @staticmethod
    def calculate_paths(
            model_params: HestonParameters, underlying_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        LOGGER.info(str(model_params.__dict__))
        rand_tensor, cho_matrix = Heston.generate_random_numbers(model_params)

        cho_matrix: np.ndarray = np.linalg.cholesky(
            np.array([[1.0, model_params.rho], [model_params.rho, 1.0]])
        )
        LOGGER.info(f"Cholesky matrix:\n{cho_matrix}")

        # Volatility process paths
        v_arr2d: np.ndarray = Heston.sample_variance_paths(model_params, cho_matrix=cho_matrix, rand=rand_tensor)

        # Underlying price process paths
        x_arr2d: np.ndarray = Heston.sample_paths(var_arr=v_arr2d, h_params=model_params, cho_matrix=cho_matrix,
                                                  rand=rand_tensor)
        if underlying_only:
            return x_arr2d
        return v_arr2d, x_arr2d

    @staticmethod
    def generate_random_numbers(model_params: HestonParameters) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(seed=model_params.random_seed)

        random_normal_arr = np.random.standard_normal((2, model_params.M + 1, model_params.I))
        LOGGER.info(f"rand shape: {random_normal_arr.shape}")

        covariance_matrix = np.array([[1.0, model_params.rho], [model_params.rho, 1.0]])
        covariance_cholesky_lower_arr = np.linalg.cholesky(covariance_matrix)
        LOGGER.info(f"Cov:\n{covariance_matrix}")
        LOGGER.info(f"L:\n{covariance_cholesky_lower_arr}")

        return random_normal_arr, covariance_cholesky_lower_arr

    @staticmethod
    def plot_paths(
            n: int,
            paths: Dict[str, np.ndarray],
            model_params: HestonParameters,
            model_name: str,
            logy: bool = False
    ):
        # Expect variance array as extra argument
        x_paths = paths['x']
        var_paths = paths['var']
        if var_paths is None:
            raise ValueError("Variance array must be provided as an argument to plot_paths for Heston.")

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

        # Underlying price paths
        ax0 = axs[0, 0]
        ax0.plot(range(len(x_paths)), x_paths[:, :n])
        ax0.grid()
        ax0.set_title(f"{model_name} Underlying Price paths")
        ax0.set_xlabel("Timestep")
        ax0.set_ylabel("Price")

        # Distribution of final log return
        ax2 = axs[1, 0]
        logr_last = np.log(x_paths[-1, :] / model_params.S0)
        q5 = np.quantile(logr_last, 0.05)
        ax2.hist(
            logr_last, density=True, bins=500,
            label=f"{model_name} (q1={np.quantile(logr_last, 0.01):.3g}, q5={q5:.3g},"
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

        # Variance paths
        ax1 = axs[0, 1]
        ax1.plot(range(len(var_paths)), var_paths[:, :n])
        ax1.grid()
        ax1.set_title(f"{model_name} Variance paths")
        ax1.set_ylabel("Variance")
        ax1.set_xlabel("Timestep")

        # Distribution of final variance
        ax3 = axs[1, 1]
        var_last = var_paths[-1, :]
        ax3.hist(var_last, density=True, bins=500)
        ax3.axvline(x=model_params.sigma ** 2, color='black', linestyle='--', label='sigma^2')
        x_var = np.linspace(var_last.min(), var_last.max(), 500)
        ax3.plot(
            x_var, ss.lognorm.pdf(x_var, *ss.lognorm.fit(var_last, floc=0)),
            color="r", label=f"LogNormal density"
        )
        ax3.set_xlabel('Variance')
        ax3.legend()

        plt.show()


def example_heston():
    h_params: HestonParameters = HestonParameters(
        v0=0.04,
        kappa=2,
        sigma=0.3,
        theta=0.04,
        rho=-0.9,

        S0=100,  # Current underlying asset price
        r=0.05,  # Risk-free rate

        T=1,  # Number of years
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0
    )

    V, S = Heston.calculate_paths(h_params)
    Heston.plot_paths(
        n=300, paths={'x': S, 'var': V}, model_params=h_params, model_name=Heston.__name__
    )

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )}")


if __name__ == "__main__":
    example_heston()
