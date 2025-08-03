from dataclasses import dataclass

import numpy as np

from erdqlib.src.mc.dynamics import ModelParameters, DynamicsParameters
from erdqlib.src.mc.dynamics import MonteCarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)

CIR_MIN_RATE: float = 1e-9  # Minimum rate to avoid numerical issues in CIR process


def gamma(kappa_r, sigma_r) -> float:
    """
    Gamma function in CIR (1985)
    """
    return np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)


def b1(alpha) -> float:
    """
    b1 function in CIR (1985)
    alpha is the parameter set
    """
    r0, kappa_r, theta_r, sigma_r, t, T = alpha
    g = gamma(kappa_r, sigma_r)
    x = (
                (2 * g * np.exp((kappa_r + g) * (T - t) / 2))
                / (2 * g + (kappa_r + g) * (np.exp(g * (T - t)) - 1))
        ) ** (2 * kappa_r * theta_r / sigma_r ** 2)

    return x


def b2(alpha):
    """
    b2 function in CIR (1985)
    alpha is the parameter set
    """
    r0, kappa_r, theta_r, sigma_r, t, T = alpha
    g = gamma(kappa_r, sigma_r)
    x = (2 * (np.exp(g * (T - t)) - 1)) / (
            2 * g + (kappa_r + g) * (np.exp(g * (T - t)) - 1)
    )

    return x


def B(alpha):
    """
    ZCB prices in the CIR (1985) model
    B_0(T) = b_1(T) exp(-b_2(T) * E_0^Q(r_t|T))
    """
    # Deterministic part of the ZCB price
    b_1 = b1(alpha)
    # Multiplier to the stochastic part of the rate exponent
    b_2 = b2(alpha)
    r0, kappa_r, theta_r, sigma_r, t, T = alpha

    # Expected rate at time t under Q-measure
    E_rt = theta_r + np.exp(-kappa_r * t) * (r0 - theta_r)

    # Vasicek model of the ZCB price by short-rate process
    zcb = b_1 * np.exp(-b_2 * E_rt)

    # np.nan comes the case when the rate goes negative
    if np.isnan(zcb):
        raise ValueError("CIR ZCB calculation resulted in NaN")
    return zcb


@dataclass
class CirDynamicsParameters(DynamicsParameters):
    kappa_cir: float
    theta_cir: float
    sigma_cir: float

    def get_value_arr(self) -> np.ndarray:
        return np.array([self.kappa_cir, self.theta_cir, self.sigma_cir])

    @staticmethod
    def from_calibration_output(opt_arr: np.ndarray, x0: float, *_, **__) -> "CirDynamicsParameters":
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
        c_params: CirParameters, sampling_method: str = "euler"
    ) -> np.ndarray:
        """Cox-Ingersoll-Ross process paths sampler
        dX_t = k * (theta - X_t) * dt + sigma * sqrt(X_t) * dW_t
        X_t = X_0 e^{-k*t} + theta * (1 - e^{-k*t}) + sigma * e^{-k*t} * integral_0^t e^{k*s} sqrt(X_s) dW_s
        """
        np.random.seed(seed=c_params.random_seed)
        dt: float = c_params.get_dt()

        x_arr2d: np.ndarray = c_params.create_zeros_state_matrix()
        x_arr2d[0] = np.maximum(c_params.x0, CIR_MIN_RATE)  # Ensure non-negative initial value for CIR process
        for t in range(1, c_params.M + 1):
            if sampling_method == "exact":
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
            elif sampling_method == "euler":
                # Euler-Maruyama method for CIR process
                x_arr2d[t] = (
                    x_arr2d[t - 1]
                    + c_params.kappa_cir * (c_params.theta_cir - x_arr2d[t - 1]) * dt
                    + c_params.sigma_cir * np.sqrt(np.maximum(x_arr2d[t - 1], 0)) * np.random.normal(0, np.sqrt(dt), size=c_params.I)
                )
                x_arr2d[t] = np.maximum(x_arr2d[t], 1e-9)

        return x_arr2d

    @staticmethod
    def calculate_paths(model_params: CirParameters, *_, **__) -> np.ndarray:
        """Merton jump process paths sampler"""
        LOGGER.info(str(model_params.__dict__))
        x_arr2d: np.ndarray = Cir.sample_paths(model_params)
        return x_arr2d
