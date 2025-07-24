from dataclasses import dataclass

import numpy as np

from erdqlib.src.mc.dynamics import ModelParameters, DynamicsParameters
from erdqlib.src.mc.dynamics import MonteCarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


@dataclass
class CirDynamicsParameters(DynamicsParameters):
    kappa_cir: float
    theta_cir: float
    sigma_cir: float

    def get_value_arr(self) -> np.ndarray:
        return np.array([self.kappa_cir, self.theta_cir, self.sigma_cir])

    @staticmethod
    def from_calibration_output(opt_arr: np.ndarray, x0: float) -> "CirDynamicsParameters":
        return CirDynamicsParameters(
            x0=x0, r=None,
            kappa_cir=float(opt_arr[0]),
            theta_cir=float(opt_arr[1]),
            sigma_cir=float(opt_arr[2])
        )


@dataclass
class CirParameters(ModelParameters, CirDynamicsParameters):
    pass


class Cir(MonteCarlo):

    @staticmethod
    def sample_paths(
        c_params: CirParameters
    ) -> np.ndarray:
        """Cox-Ingersoll-Ross process paths sampler
        dX_t = k * (theta - X_t) * dt + sigma * sqrt(X_t) * dW_t
        X_t = X_0 e^{-k*t} + theta * (1 - e^{-k*t}) + sigma * e^{-k*t} * integral_0^t e^{k*s} sqrt(X_s) dW_s
        """
        np.random.seed(seed=c_params.random_seed)
        dt: float = c_params.get_dt()

        x_arr2d: np.ndarray = c_params.create_zeros_state_matrix()
        for t in range(0, c_params.M + 1):
            if t == 0:
                x_arr2d[0] = c_params.x0
                continue

            # inside your time‐stepping loop, at step t > 0:
            exp_k_dt = np.exp(-c_params.kappa_cir * dt)
            # scale for non‐central χ²
            c = c_params.sigma_cir ** 2 * (1 - exp_k_dt) / (4 * c_params.kappa_cir)
            # degrees of freedom
            df = 4 * c_params.kappa_cir * c_params.theta_cir / c_params.sigma_cir ** 2
            # non‐centrality parameter
            nc = (
                    x_arr2d[t - 1]
                    * 4 * c_params.kappa_cir * exp_k_dt
                    / (c_params.sigma_cir ** 2 * (1 - exp_k_dt))
            )
            # exact CIR update
            x_arr2d[t] = c * np.random.noncentral_chisquare(df, nc)

        return x_arr2d

    @staticmethod
    def calculate_paths(model_params: CirParameters, *_, **__) -> np.ndarray:
        """Merton jump process paths sampler"""
        LOGGER.info(str(model_params.__dict__))
        x_arr2d: np.ndarray = Cir.sample_paths(model_params)
        return x_arr2d


def example_cir():
    v_params: CirParameters = CirParameters(
        T = 1.0,  # Maturity
        M = 500,  # Number of paths for MC
        I = 10_000,  # Number of steps
        random_seed=0,

        x0= 0.023,
        kappa_cir= 0.20,
        theta_cir = 0.01,
        sigma_cir = 0.012,
        r = None,  # Risk-free rate
    )

    rates = Cir.calculate_paths(v_params)
    Cir.plot_paths(
        n=300,
        paths={'x': rates},
        model_params=v_params,
        model_name=Cir.__name__,
        logy=False,
        ylabel="Rate"
    )


if __name__ == "__main__":
    example_cir()