from abc import ABC, abstractmethod
import numpy as np

class Dynamics(ABC):
    """
    Abstract base class representing a dynamics model for option pricing.
    """
    
    @abstractmethod
    def simulate_paths_exact(self, simulations: int, random_seed: int = None) -> np.ndarray:
        """
        Simulates correlated price paths for assets.

        Parameters:
            simulations (int): Number of simulation paths to generate.
            random_seed (int, optional): Seed for the random number generator.

        Returns:
            np.ndarray: Simulated price paths with shape (simulations, steps + 1, num_assets).
        """
        pass

class GeometricBrownianMotion(Dynamics):
    """
    Implements the Geometric Brownian Motion (GBM) dynamics model for option pricing.
    
    The GBM is defined by the stochastic differential equation:
    
        $$ dS_t = mu S_t dt + sigma S_t dW_t $$
    
    where:
        - ( S_t ) is the asset price at time ( t ),
        - ( mu ) is the drift rate,
        - ( sigma ) is the volatility,
        - ( dW_t ) is the Wiener process increment.
    """
    
    def __init__(self, P0: list, T: float, dt: float, vol: list, drift: list, correlation: list):
        """
        Initializes the GBM model with given parameters.

        Parameters:
            P0 (list): Initial prices for each asset.
            T (float): Time to maturity (in years).
            dt (float): Time step size.
            vol (list): Volatilities for each asset.
            drift (list): Drift rates for each asset.
            correlation (list): Correlation matrix between assets.
        """
        self.P0 = np.array(P0)
        self.T = T
        self.dt = dt
        self.vol = np.array(vol)
        self.drift = np.array(drift)
        self.correlation = np.array(correlation)
        self.steps = int(T / dt)
        self.num_assets = len(P0)
        self.L = np.linalg.cholesky(self.correlation)
    
    def simulate_paths_exact(self, simulations: int, random_seed: int = None) -> np.ndarray:
        """
        Simulates asset price paths using the Geometric Brownian Motion (GBM) model.

        The GBM is defined by the stochastic differential equation:

            $$ dS_t = mu S_t dt + sigma S_t dW_t $$

        where:
            - ( S_t ) is the asset price at time ( t ),
            - ( mu ) is the drift rate,
            - ( sigma ) is the volatility,
            - ( dW_t ) is the Wiener process increment.

        Parameters:
            simulations (int): Number of simulation paths.
            random_seed (int, optional): Seed for reproducibility.

        Returns:
            np.ndarray: Simulated price paths with shape (simulations, steps + 1, num_assets).
        """
        np.random.seed(random_seed)
        prices = np.zeros((simulations, self.steps + 1, self.num_assets))
        prices[:, 0, :] = self.P0

        for sim in range(simulations):
            # Generate independent standard normal random variables
            Z = np.random.normal(size=(self.steps, self.num_assets))
            # Introduce correlation between assets
            correlated_Z = np.dot(Z, self.L.T)
            for t in range(1, self.steps + 1):
                dW = correlated_Z[t - 1] * np.sqrt(self.dt)
                drift_term = (self.drift - 0.5 * self.vol**2) * self.dt
                diffusion_term = self.vol * dW
                # Update prices using the GBM formula
                prices[sim, t] = prices[sim, t - 1] * np.exp(drift_term + diffusion_term)
        return prices

    def simulate_paths_approx(self, simulations: int, random_seed: int = None) -> np.ndarray:
        """
        Simulates asset price paths using an alternative formulation of the Geometric Brownian Motion (GBM) model.

        The alternative GBM is defined by the stochastic differential equation:

            $$ dS_t = mu S_t dt + sigma S_t dW_t $$

        where:
            - ( S_t ) is the asset price at time ( t ),
            - ( mu ) is the drift rate,
            - ( sigma ) is the volatility,
            - ( dW_t ) is the Wiener process increment.

        This method does not include drift correction.

        Parameters:
            simulations (int): Number of simulation paths.
            random_seed (int, optional): Seed for reproducibility.

        Returns:
            np.ndarray: Simulated price paths with shape (simulations, steps + 1, num_assets).
        """
        np.random.seed(random_seed)
        prices = np.zeros((simulations, self.steps + 1, self.num_assets))
        prices[:, 0, :] = self.P0

        for sim in range(simulations):
            # Generate independent standard normal random variables
            Z = np.random.normal(size=(self.steps, self.num_assets))
            # Introduce correlation between assets
            correlated_Z = np.dot(Z, self.L.T)
            for t in range(1, self.steps + 1):
                dW = correlated_Z[t - 1] * np.sqrt(self.dt)
                drift_term = self.drift * self.dt  # No drift correction
                diffusion_term = self.vol * dW
                log_increment = drift_term + diffusion_term
                # Update prices using the alternative GBM formula
                prices[sim, t] = prices[sim, t - 1] * np.exp(log_increment)
        return prices

class EuropeanOption:
    """
    Class representing a European option and methods to calculate its present value.
    """
    
    def __init__(self, K: float, r: float, dynamics: Dynamics):
        """
        Initializes the EuropeanOption with strike price, risk-free rate, and dynamics model.

        Parameters:
            K (float): Strike price of the option.
            r (float): Risk-free interest rate (annualized).
            dynamics (Dynamics): An instance of a Dynamics model to simulate price paths.
        """
        self.K = K
        self.r = r
        self.dynamics = dynamics

    def payoff(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculates the payoff for a European call option based on simulated prices.

        Parameters:
            prices (np.ndarray): Simulated price paths with shape (simulations, steps + 1, num_assets).

        Returns:
            np.ndarray: Array of payoffs for each simulation.
        """
        # Calculate the average of final prices across all assets for each simulation
        terminal_prices = np.mean(prices[:, -1, :], axis=1)
        # Payoff is max(average_price - strike, 0)
        payoffs = np.maximum(terminal_prices - self.K, 0)
        return payoffs

    def calculate_PV(self, simulations: int = 10000, random_seed: int = None, use_approx: bool = False) -> float:
        """
        Calculates the present value of the European option based on simulated payoffs.

        Parameters:
            simulations (int, optional): Number of simulation paths. Default is 10,000.
            random_seed (int, optional): Seed for reproducibility.
            use_dS (bool, optional): Whether to use the alternative GBM simulation method.

        Returns:
            float: Estimated present value of the option.
        """
        # Simulate price paths using the chosen dynamics method
        if use_approx:
            prices = self.dynamics.simulate_paths_dS(simulations, random_seed=random_seed)
        else:
            prices = self.dynamics.simulate_paths_exact(simulations, random_seed=random_seed)
        
        # Calculate payoffs based on simulated prices
        payoffs = self.payoff(prices)
        
        # Discount the average payoff back to present value
        discounted_payoff = np.mean(payoffs) * np.exp(-self.r * self.dynamics.T)
        return discounted_payoff

def main():
    """
    Example usage of the EuropeanOption and GeometricBrownianMotion classes.
    """
    # Parameters for the simulation
    P0 = [100, 120]  # Initial prices for two assets
    T = 1.0  # Time to maturity in years
    K = 110.0  # Strike price of the option
    dt = 0.01  # Time step size (in years)
    vol = [0.2, 0.3]  # Annualized volatilities for each asset
    drift = [0.05, 0.04]  # Annualized drift rates for each asset
    correlation = [
        [1.0, 0.8],
        [0.8, 1.0]
    ]  # Correlation matrix between the two assets
    r = 0.05  # Risk-free interest rate (annualized)

    # Initialize dynamics and option objects
    dynamics = GeometricBrownianMotion(P0, T, dt, vol, drift, correlation)
    option = EuropeanOption(K, r, dynamics)

    # Calculate present values using both simulation methods
    PV_GBM = option.calculate_PV(simulations=1000, random_seed=42, use_approx=False)
    PV_GBM_approx = option.calculate_PV(simulations=1000, random_seed=42, use_approx=True)

    # Output the results
    print(f"European Call Option Price (GBM): {PV_GBM:.2f}")
    print(f"European Call Option Price (Alternative GBM): {PV_GBM_approx:.2f}")

if __name__ == "__main__":
    main()
