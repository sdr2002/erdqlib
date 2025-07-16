from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from erdqlib.src.mc.dynamics import ModelParameters


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
            paths: np.ndarray,
            model_params: ModelParameters,
            model_name: str,
            logy: bool = False,
            *args, **kwargs
    ):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 8))

        # Paths of underlying price
        ax0 = axs[0]
        ax0.plot(paths[:, :n])
        ax0.grid()
        ax0.set_title(f"{model_name} Underlying Price Paths")
        ax0.set_xlabel("Timestep")
        ax0.set_ylabel("Price")

        # Distribution of final (log) return of underlying price
        y_arr = paths[-1, :] if not logy else np.log(paths[-1, :] / model_params.S0)
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
