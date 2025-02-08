from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from erdqlib.src.mc.dynamics import Dynamics

class Option(ABC):
    def __init__(self, K: float, r: float, dynamics: Dynamics) -> None:
        self.K: float = K
        self.r: float = r
        self.dynamics: Dynamics = dynamics

    @abstractmethod
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Payoff method not implemented!")

    def calculate_PV(self, n_paths: int = 10000, 
                    random_seed: Optional[int] = None, 
                    use_approx: bool = False) -> float:
        if use_approx:
            paths = self.dynamics.simulate_paths_approx(n_paths, random_seed)
        else:
            paths = self.dynamics.simulate_paths_exact(n_paths, random_seed)
        payoffs = self.payoff(paths)
        discounted_payoff = np.exp(-self.r * self.dynamics.T) * np.mean(payoffs)
        return discounted_payoff

class EuropeanOption(Option):
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        terminal_prices = np.mean(prices[:, -1, :], axis=1)
        return np.maximum(terminal_prices - self.K, 0)
