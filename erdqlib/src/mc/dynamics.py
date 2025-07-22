from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
from typing import Self, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss


@dataclass
class SamplingParameters:
    T: float  # Number of years
    M: int  # Number time transitions
    I: int  # Number of paths

    random_seed: int  # random seed

    def get_sampling_parameters(self) -> Self:
        return SamplingParameters(
            T=self.T, M=self.M, I=self.I, random_seed=self.random_seed
        )

    def get_dt(self) -> float:
        return float(self.T / self.M)

    def get_Myear(self) -> float:
        return 1. / self.get_dt()

    def create_zeros_state_matrix(self) -> np.ndarray:
        # Create M state-transition for I paths
        return np.zeros((self.M + 1, self.I), dtype=float)


@dataclass
class DynamicsParameters:
    S0: Optional[float]  # Current underlying asset price
    r: Optional[float]  # Risk-free rate

    def get_dynamics_parameters(self) -> Self:
        return DynamicsParameters(
            S0=self.S0, r=self.r
        )

    @staticmethod
    def do_parameters_offbound(*args, **kwargs) -> bool:
        """
        Check if any of the parameters are off the bounds.
        This method should be overridden in derived classes.
        """
        raise NotImplementedError("This method should be overridden in derived classes.")

    def get_values(self) -> Tuple[...]:
        """
        Get the values of the dynamics parameters.
        This method should be overridden in derived classes.
        """
        raise NotImplementedError("This method should be overridden in derived classes.")

    @staticmethod
    def from_calibration_output(opt_arr: np.ndarray, *_, **__) -> "DynamicsParameters":
        """
        Create an instance of the dynamics parameters from calibration output.
        This method should be overridden in derived classes.
        """
        raise NotImplementedError("This method should be overridden in derived classes.")

    @staticmethod
    def get_default_search_grid() -> Tuple[...]:
        """
        Get the default search grid for the dynamics parameters.
        This method should be overridden in derived classes.
        """
        raise NotImplementedError("This method should be overridden in derived classes.")

    def __str__(self, new_line: bool = True) -> str:
        table_str: str = pd.DataFrame(asdict(self), index=[0]).to_markdown(index=False)
        return "\n" + table_str if new_line else table_str


@dataclass
class ModelParameters(SamplingParameters, DynamicsParameters):
    pass


class MonteCarlo(ABC):
    @staticmethod
    @abstractmethod
    def sample_paths(*args, **kwargs):
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def calculate_paths(*args, **kwargs):
        raise NotImplemented

    @staticmethod
    def plot_paths(
            n: int,
            paths: Dict[str, np.ndarray],
            model_params: ModelParameters,
            model_name: str,
            logy: bool = False
    ):
        x_paths: np.ndarray = paths['x']
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 8))

        # Paths of underlying price
        ax0 = axs[0]
        ax0.plot(x_paths[:, :n])
        ax0.grid()
        ax0.set_title(f"{model_name} Underlying Price Paths")
        ax0.set_xlabel("Timestep")
        ax0.set_ylabel("Price")

        # Distribution of final (log) return of underlying price
        y_arr = x_paths[-1, :] if not logy else np.log(x_paths[-1, :] / model_params.S0)
        x = np.linspace(y_arr.min(), y_arr.max(), 500)

        ax1 = axs[1]
        q5 = np.quantile(y_arr, 0.05)
        ax1.hist(
            y_arr, density=True, bins=500,
            label=f"{model_name} (q1={np.quantile(y_arr, 0.01):.3g}, q5={q5:.3g},"
                  f" sk={ss.skew(y_arr):.3g}, kt={ss.kurtosis(y_arr):.3g})"
        )
        ax1.axvline(x=q5, color='black', linestyle='--')
        ax1.plot(
            x, ss.norm.pdf(x, y_arr.mean(), y_arr.std()),
            color="r", label=f"Normal density (mu={y_arr.mean():.2g}, std={y_arr.std():.2g})"
        )
        ax1.set_xlabel(f'{"Log " if logy else ""}X_final')
        ax1.legend()
        plt.show()

    @staticmethod
    def generate_random_numbers(model_params: ModelParameters) -> np.ndarray:
        np.random.seed(seed=model_params.random_seed)
        random_normal_arr = np.random.standard_normal((model_params.M + 1, model_params.I))
        return random_normal_arr
