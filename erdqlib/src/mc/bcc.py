import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as ss

from erdqlib.src.common.option import OptionInfo, OptionType, OptionSide
from erdqlib.src.common.rate import implied_yield
from erdqlib.src.mc.bates import BatesDynamicsParameters, BatesParameters, Bates
from erdqlib.src.mc.cir import CirDynamicsParameters, CirParameters, B, Cir
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.jump import MertonJump
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


@dataclass
class BccDynamicsParameters(CirDynamicsParameters, BatesDynamicsParameters):
    def get_values(self) -> Tuple[
        float, float, float,
        float, float, float, float, float,
        float, float, float
    ]:
        return (
            self.kappa_cir, self.theta_cir, self.sigma_cir,
            self.kappa_heston, self.theta_heston, self.sigma_heston, self.rho_heston, self.v0_heston,
            self.lambd_merton, self.mu_merton, self.delta_merton
        )

    def get_bates_parameters(self) -> BatesDynamicsParameters:
        """Get Bates parameters from BCC dynamics parameters"""
        return BatesDynamicsParameters(
            x0=self.x0, r=self.r,
            kappa_heston=self.kappa_heston,
            theta_heston=self.theta_heston,
            sigma_heston=self.sigma_heston,
            rho_heston=self.rho_heston,
            v0_heston=self.v0_heston,
            lambd_merton=self.lambd_merton,
            mu_merton=self.mu_merton,
            delta_merton=self.delta_merton,
        )


@dataclass
class BCCParameters(BatesParameters, CirParameters):

    def to_str(self, indent: Optional[int] = None):
        return f"BCCParameters{json.dumps(self.__dict__, indent=indent)}"

    def get_B_params(self) -> List[float]:
        """Get parameters for B function"""
        return [self.r, self.kappa_cir, self.theta_cir, self.sigma_cir, 0., self.T]

    def get_pricing_params(self, apply_shortrate: bool) -> List[float]:
        """Get parameters for pricing: apply_shortrate for BCC (1997) model if True, else for Bates (1996) model"""
        r: float = self.r
        if apply_shortrate:
            r = implied_yield(t_year=self.T, price_0_t=B(self.get_B_params()))

        return [self.x0, self.T, r] + self.get_calibrable_params()

    def get_calibrable_params(self) -> List[float]:
        """Get calibration target parameters"""
        return [
            self.kappa_heston, self.theta_heston, self.sigma_heston, self.rho_heston, self.v0_heston,  # Heston
            self.lambd_merton, self.mu_merton, self.delta_merton  # Merton
        ]

    def to_bates_params(self) -> BatesParameters:
        return BatesParameters(
            x0=self.x0,
            r=implied_yield(t_year=self.T, price_0_t=B(self.get_B_params())),
            kappa_heston=self.kappa_heston, theta_heston=self.theta_heston, sigma_heston=self.sigma_heston, rho_heston=self.rho_heston, v0_heston=self.v0_heston,
            lambd_merton=self.lambd_merton, mu_merton=self.mu_merton, delta_merton=self.delta_merton,
            T=self.T, M=self.M, I=self.I, random_seed=self.random_seed
        )

    def to_cir_parameters(self) -> CirParameters:
        """Short-rates dynamics parameters: r=None becuase there's no risk-free drift for the short-rate"""
        return CirParameters(
            x0=self.r, r=None,
            kappa_cir=self.kappa_cir, theta_cir=self.theta_cir, sigma_cir=self.sigma_cir,
            T=self.T, M=self.M, I=self.I, random_seed=self.random_seed
        )


class BCC(Bates):

    @staticmethod
    def sample_paths(
            var_arr: np.ndarray,
            r_arr: np.ndarray,
            b_params: BCCParameters,
            cho_matrix: np.ndarray,
            u1: np.ndarray,
            zj: np.ndarray,
            cj: np.ndarray
    ) -> np.ndarray:
        """Heston model process paths sampler
        dX_t = (r_t - rj) * X_t * dt + sqrt{v_t} * X_t * dWx_t + J_t(mu, delta) * X_t^{-} * dS(lambda)
          where d<Wx, Wvar>_t = rho * dt, S ~ Poisson(lambda * dt), J ~ exp[N(ln(1+mu) - delta^2/2, delta^2)] - 1
        X_t = X_0 * { exp[(r - rj - 0.5 * v_t) * t + sqrt(v_t) * W1_t] + (exp[mu + delta * Z2_t] - 1) * J_t }
        or
        X_t = X_0 * { exp[(r - rj - 0.5 * v_t) * t + sqrt(v_t) * W1_t] * (1 + (exp[mu + delta * Z2_t] - 1) * J_t) }

        The v_t is the variance process and r_t is the risk-free rate process that must be sampled beforehand.
        """
        rj: float = b_params.get_r_offset()
        dt: float = b_params.get_dt()
        sdt: float = np.sqrt(dt)

        x_arr2d: np.ndarray = b_params.create_zeros_state_matrix()
        x_arr2d[0] = b_params.x0
        row: int = 0
        for t in range(1, b_params.M + 1, 1):
            drift: np.ndarray = (r_arr[t-1] - rj - 0.5 * var_arr[t - 1]) * dt
            z1: np.ndarray = np.dot(cho_matrix, u1[:, t])[row]
            stochastic_diffusion: np.ndarray = np.sqrt(var_arr[t - 1]) * sdt * z1
            jump_diffusion: np.array = (np.exp(b_params.mu_merton + b_params.delta_merton * zj[t]) - 1) * cj[t]
            # TODO are we sure to ADD the jump diffusion instead of multiplying? seeing the result, apparently so.
            # mult: np.array = np.exp(drift + stochastic_diffusion) + jump_diffusion
            mult: np.array = np.exp(drift + stochastic_diffusion) * (1 + jump_diffusion)
            x_arr2d[t] = np.maximum(x_arr2d[t - 1] * mult, MertonJump.X_LOWER_BOUND)
        return x_arr2d

    @staticmethod
    def calculate_paths(
            model_params: BCCParameters, underlying_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        LOGGER.info(model_params.to_json())
        rand_tensor, cho_matrix, zj, cj = BCC.generate_random_numbers(model_params=model_params)

        # Risk-free rate process paths
        r_arr2d: np.ndarray = Cir.sample_paths(c_params=model_params.to_cir_parameters())

        # Volatility process paths
        v_arr2d: np.ndarray = BCC.sample_variance_paths(
            h_params=model_params.to_heston_parameters(), cho_matrix=cho_matrix, u2=rand_tensor
        )

        # Underlying price process paths
        x_arr2d: np.ndarray = BCC.sample_paths(
            var_arr=v_arr2d, r_arr=r_arr2d, b_params=model_params,
            cho_matrix=cho_matrix, u1=rand_tensor, zj=zj, cj=cj
        )
        if underlying_only:
            return x_arr2d
        return r_arr2d, v_arr2d, x_arr2d

    @staticmethod
    def plot_paths(
            n: int,
            paths: Dict[str, np.ndarray],
            model_params: BCCParameters,
            model_name: str,
            logy: bool = False
    ):
        # Expect variance array as extra argument
        x_paths = paths['x']
        var_paths = paths['var']
        r_paths = paths['r']
        if var_paths is None:
            raise ValueError("Variance array must be provided as an argument to plot_paths for Heston.")

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))

        # Underlying price paths
        ax_0_0 = axs[0, 0]
        ax_0_0.plot(range(len(x_paths)), x_paths[:, :n])
        ax_0_0.grid()
        ax_0_0.set_title(f"{model_name} Underlying Price paths")
        ax_0_0.set_xlabel("Timestep")
        ax_0_0.set_ylabel("Price")

        # Distribution of final log return
        ax_1_0 = axs[1, 0]
        r_last = np.log(x_paths[-1, :] / model_params.x0)
        q5 = np.quantile(r_last, 0.05)
        ax_1_0.hist(
            r_last, density=True, bins=500,
            label=f"{model_name} q5={q5:.3g},"
                  f" sk={ss.skew(r_last):.3g}, kt={ss.kurtosis(r_last):.3g})"
        )
        ax_1_0.axvline(x=q5, color='black', linestyle='--')
        x_r = np.linspace(r_last.min(), r_last.max(), 500)
        ax_1_0.plot(
            x_r, ss.norm.pdf(x_r, r_last.mean(), r_last.std()),
            color="r", label=f"Normal density (mu={r_last.mean():.2g}, std={r_last.std():.2g})"
        )
        ax_1_0.set_xlabel('Log return')
        ax_1_0.legend()

        # Variance paths
        ax_0_1 = axs[0, 1]
        ax_0_1.plot(range(len(var_paths)), var_paths[:, :n])
        ax_0_1.grid()
        ax_0_1.set_title(f"{model_name} Variance paths")
        ax_0_1.set_ylabel("Variance")
        ax_0_1.set_xlabel("Timestep")

        # Distribution of final variance
        ax_1_1 = axs[1, 1]
        var_last = var_paths[-1, :]
        ax_1_1.hist(var_last, density=True, bins=500)
        ax_1_1.axvline(x=model_params.v0_heston, color='black', linestyle='--', label='v0')
        x_var = np.linspace(var_last.min(), var_last.max(), 500)
        try:
            ax_1_1.plot(
                x_var, ss.lognorm.pdf(x_var, *ss.lognorm.fit(var_last, floc=0)),
                color="r", label=f"LogNormal density"
            )
        except Exception as e:
            pass
        ax_1_1.set_xlabel('Variance')
        ax_1_1.legend()

        # Risk-free rate paths
        ax_0_2 = axs[0, 2]
        ax_0_2.plot(range(len(r_paths)), r_paths[:, :n])
        ax_0_2.grid()
        ax_0_2.set_title(f"{model_name} Risk-free rate paths")
        ax_0_2.set_xlabel("Timestep")
        ax_0_2.set_ylabel("Price")

        # Distribution of final log return
        ax_1_2 = axs[1, 2]
        r_last = r_paths[-1, :]
        q5 = np.quantile(r_last, 0.05)
        ax_1_2.hist(
            r_last, density=True, bins=500,
            label=f"{model_name} q5={q5:.3g},"
                  f" sk={ss.skew(r_last):.3g}, kt={ss.kurtosis(r_last):.3g})"
        )
        # ax_1_2.axvline(x=model_params.r, color='black', linestyle='--', label='r0')
        x_r = np.linspace(r_last.min(), r_last.max(), 500)
        ax_1_2.plot(
            x_r, ss.norm.pdf(x_r, r_last.mean(), r_last.std()),
            color="r", label=f"Normal density (mu={r_last.mean():.2g}, std={r_last.std():.2g})"
        )
        ax_1_2.set_xlabel('Rate')
        ax_1_2.legend()

        plt.show()


def example_bcc():
    # r for BCC works as the r0 for short-rate dynamics
    bcc_params: BCCParameters = BCCParameters(
        T=1.0,
        M=250,  # type: ignore
        I=100_000,  # type: ignore
        random_seed=0,  # type: ignore
        **{
            "x0": 3225.93,
            "r": -0.0002943493765991875,
            "lambd_merton": 8.463250888577545e-07,
            "mu_merton": -9.449456757062437e-06,
            "delta_merton": 9.821793189744765e-07,
            "kappa_heston": 2.412006106350429,
            "theta_heston": 0.022478722375727497,
            "sigma_heston": 0.32821891480428733,
            "rho_heston": -0.6712549317699577,
            "v0_heston": 0.030392880298239243,
            "kappa_cir": 0.603708218317528,
            "theta_cir": 0.03120714567592455,
            "sigma_cir": 0.19411341500935292
        }
    )

    r_arr, v_arr, x_arr = BCC.calculate_paths(model_params=bcc_params)
    BCC.plot_paths(
        n=500,
        paths={'x': x_arr, 'var': v_arr, 'r': r_arr},
        model_params=bcc_params,
        model_name=BCC.__name__
    )

    o_price: float = price_montecarlo(
        underlying_path=x_arr,
        d=bcc_params,
        o=OptionInfo(
            o_type=OptionType.ASIAN,
            K=bcc_params.x0 * 0.95,  # strike price
            side=OptionSide.PUT,
        )
    )
    LOGGER.info(f"Asian PUT option price: {o_price}")


if __name__ == "__main__":
    example_bcc()