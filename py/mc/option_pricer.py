from abc import ABC, abstractmethod
import numpy as np

# Abstract class for Dynamics
class Dynamics(ABC):
    """
    Abstract base class for all price dynamics models.
    """
    @abstractmethod
    def simulate_paths(self, simulations: int):
        """
        Simulate correlated price paths for assets.
        """
        pass

# Implementation of Dynamics: Arithmetic Brownian Motion (ABM)
class ArithmeticBrownianMotion(Dynamics):
    def __init__(self, P0, T, dt, vol, drift, correlation):
        self.P0 = np.array(P0)
        self.T = T
        self.dt = dt
        self.vol = np.array(vol)
        self.drift = np.array(drift)
        self.correlation = np.array(correlation)
        self.steps = int(T / dt)
        self.num_assets = len(P0)
        self.L = np.linalg.cholesky(self.correlation)

    def simulate_paths(self, simulations: int, random_seed=None):
        np.random.seed(random_seed)
        prices = np.zeros((simulations, self.steps + 1, self.num_assets))
        prices[:, 0, :] = self.P0

        for sim in range(simulations):
            Z = np.random.normal(size=(self.steps, self.num_assets))
            correlated_Z = np.dot(Z, self.L.T)
            for t in range(1, self.steps + 1):
                dW = correlated_Z[t - 1] * np.sqrt(self.dt)
                drift_term = (self.drift - 0.5 * self.vol**2) * self.dt
                diffusion_term = self.vol * dW
                prices[sim, t] = prices[sim, t - 1] * np.exp(drift_term + diffusion_term)
        return prices

    def simulate_paths_dS(self, simulations: int, random_seed=None):
        """
        Simulates paths using the alternative GBM formula:
        ln(S_{t+dt} / S_t) = μ * dt + σ * sqrt(dt) * Z
        """
        np.random.seed(random_seed)
        prices = np.zeros((simulations, self.steps + 1, self.num_assets))
        prices[:, 0, :] = self.P0

        for sim in range(simulations):
            Z = np.random.normal(size=(self.steps, self.num_assets))
            correlated_Z = np.dot(Z, self.L.T)
            for t in range(1, self.steps + 1):
                dW = correlated_Z[t - 1] * np.sqrt(self.dt)
                drift_term = self.drift * self.dt  # No drift correction
                diffusion_term = self.vol * dW
                log_increment = drift_term + diffusion_term
                prices[sim, t] = prices[sim, t - 1] * np.exp(log_increment)
        return prices


class EuropeanOption:
    def __init__(self, K, r, dynamics):
        self.K = K
        self.r = r
        self.dynamics = dynamics

    def payoff(self, prices):
        terminal_prices = np.mean(prices[:, -1, :], axis=1)
        payoffs = np.maximum(terminal_prices - self.K, 0)
        return payoffs

    def calculate_PV(self, simulations=10000, random_seed=None, use_dS=False):
        if use_dS:
            prices = self.dynamics.simulate_paths_dS(simulations, random_seed=random_seed)
        else:
            prices = self.dynamics.simulate_paths(simulations, random_seed=random_seed)
        payoffs = self.payoff(prices)
        discounted_payoff = np.mean(payoffs) * np.exp(-self.r * self.dynamics.T)
        return discounted_payoff


# Example Usage
if __name__ == "__main__":
    # Parameters
    P0 = [100, 120]  # Initial prices for two assets
    T = 1  # 1 year to maturity
    K = 110  # Strike price
    dt = 0.01  # Time step (in years)
    vol = [0.2, 0.3]  # Volatilities (annualized)
    drift = [0.05, 0.04]  # Drifts (annualized)
    correlation = [[1.0, 0.8], [0.8, 1.0]]  # Correlation matrix
    r = 0.05  # Risk-free rate (annualized)

    # Create dynamics and option objects
    dynamics = ArithmeticBrownianMotion(P0, T, dt, vol, drift, correlation)
    option = EuropeanOption(K, r, dynamics)

    # Calculate present values using both methods
    PV_GBM = option.calculate_PV(simulations=1000, random_seed=42, use_dS=False)
    PV_GBM_alt = option.calculate_PV(simulations=1000, random_seed=42, use_dS=True)

    print(f"European Call Option Price (GBM): {PV_GBM:.2f}")
    print(f"European Call Option Price (Alternative GBM): {PV_GBM_alt:.2f}")
