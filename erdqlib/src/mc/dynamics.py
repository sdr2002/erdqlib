import json
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
    M: Optional[int]  # Number time transitions
    I: Optional[int]  # Number of paths

    random_seed: Optional[int]  # random seed

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
    x0: Optional[float]  # Current underlying asset price
    r: Optional[float]  # Risk-free rate

    def get_dynamics_parameters(self) -> Self:
        return DynamicsParameters(
            x0=self.x0, r=self.r
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
    def from_calibration_output(opt_arr: np.ndarray, s0:float, r:float, *_, **__) -> "DynamicsParameters":
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
        table_str: str = pd.DataFrame(asdict(self), index=[0]).dropna(axis=1).to_markdown(index=False)
        return "\n" + table_str if new_line else table_str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4, ensure_ascii=True)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(self), index=['value'], dtype=float)

    def to_csv(self):
        self.to_dataframe().to_csv(index=False)


@dataclass
class ModelParameters(SamplingParameters, DynamicsParameters):

    def get_r_at_t(self, *_, **__) -> float:
        """
        Get the risk-free rate at expiry.
        If r is None, return 0.0.
        """
        if self.r is None:
            raise ValueError("Risk-free rate (r) is not set.")
        return self.r


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
            logy: bool = False,
            ylabel: str = "Price"
    ):
        x_paths: np.ndarray = paths['x']
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 8))

        # Paths of underlying price
        ax0 = axs[0]
        ax0.plot(x_paths[:, :n])
        ax0.grid()
        ax0.set_title(f"{model_name} Underlying Price Paths")
        ax0.set_xlabel("Timestep")
        ax0.set_ylabel(ylabel)

        # Distribution of final (log) return of underlying price
        y_arr = x_paths[-1, :] if not logy else np.log(x_paths[-1, :] / model_params.x0)
        x = np.linspace(y_arr.min(), y_arr.max(), 500)

        ax1 = axs[1]
        q5 = np.quantile(y_arr, 0.05)
        q95 = np.quantile(y_arr, 0.95)
        ax1.hist(
            y_arr, density=True, bins=500,
            label=f"{model_name} (q5={q5:.3g}, q95={q95:.3g},"
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
