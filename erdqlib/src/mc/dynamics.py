from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional

class Dynamics(ABC):
    """Abstract base class representing a dynamics model for option pricing."""
    
    @abstractmethod
    def simulate_paths_exact(self, simulations: int, random_seed: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError("simulate_paths_exact method not implemented!")

    @abstractmethod
    def simulate_paths_approx(self, simulations: int, random_seed: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError("simulate_paths_approx method not implemented!")

class GeometricBrownianMotion(Dynamics):
    def __init__(self, P0: List[float], T: float, dt: float, vol: List[float], 
                 drift: List[float], correlation: List[List[float]]) -> None:
        self.P0: np.ndarray = np.array(P0)
        self.T: float = T
        self.dt: float = dt
        self.vol: np.ndarray = np.array(vol)
        self.drift: np.ndarray = np.array(drift)
        self.correlation: np.ndarray = np.array(correlation)
        self.steps: int = int(T / dt)
        self.num_assets: int = len(P0)
        self.L: np.ndarray = np.linalg.cholesky(self.correlation)
    
    def simulate_paths_exact(self, simulations: int, random_seed: Optional[int] = None) -> np.ndarray:
        if random_seed is not None:
            np.random.seed(random_seed)
        Z = np.random.normal(size=(simulations, self.steps, self.num_assets))
        dW = np.matmul(Z, self.L.T) * np.sqrt(self.dt)
        paths = np.zeros((simulations, self.steps + 1, self.num_assets))
        paths[:, 0, :] = self.P0
        for t in range(1, self.steps + 1):
            paths[:, t, :] = paths[:, t - 1, :] * np.exp(
                (self.drift - 0.5 * self.vol ** 2) * self.dt + self.vol * dW[:, t - 1, :]
            )
        return paths

    def simulate_paths_approx(self, simulations: int, random_seed: Optional[int] = None) -> np.ndarray:
        if random_seed is not None:
            np.random.seed(random_seed)
        Z = np.random.normal(size=(simulations, self.steps, self.num_assets))
        dW = np.matmul(Z, self.L.T) * np.sqrt(self.dt)
        paths = np.zeros((simulations, self.steps + 1, self.num_assets))
        paths[:, 0, :] = self.P0
        for t in range(1, self.steps + 1):
            paths[:, t, :] = paths[:, t - 1, :] * np.exp(
                self.drift * self.dt + self.vol * dW[:, t - 1, :]
            )
        return paths
