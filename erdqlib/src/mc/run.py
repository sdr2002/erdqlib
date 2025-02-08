import logging
from erdqlib.src.mc.option import EuropeanOption
from erdqlib.src.mc.dynamics import GeometricBrownianMotion
from erdqlib.src.mc.plot.plot_paths import compare_simulated_paths_with_adjusted_layout

LOGGER = logging.getLogger(__name__)


def pricing_ex0() -> None:
    """
    Example usage of the EuropeanOption and GeometricBrownianMotion classes.
    """

    # Parameters for the simulation
    random_seed: int = 42
    n_paths: int = int(1e3)

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
    PV_GBM = option.calculate_PV(n_paths=n_paths, random_seed=random_seed, use_approx=False)
    PV_GBM_approx = option.calculate_PV(n_paths=n_paths, random_seed=random_seed, use_approx=True)

    # Output the results
    LOGGER.info(f"European Call Option Price (GBM): {PV_GBM:.2f}")
    LOGGER.info(f"European Call Option Price (Alternative GBM): {PV_GBM_approx:.2f}")


def plotting_ex0():
    # Parameters for the simulation
    random_seed: int = 153
    n_paths: int = int(3e2)

    # Parameters for the simulation
    P0 = [100, 120]  # Initial prices for two assets
    T = 3.0  # Time to maturity in years
    dt = 0.01  # Time step size (in years)
    vol = [0.2, 0.3]  # Annualized volatilities for each asset
    drift = [0.05, 0.04]  # Annualized drift rates for each asset
    correlation = [
        [1.0, 0.8],
        [0.8, 1.0]
    ]  # Correlation matrix between the two assets

    # Initialize dynamics object
    dynamics = GeometricBrownianMotion(P0, T, dt, vol, drift, correlation)

    # Plot paths for both simulate_paths and simulate_paths_dS in the same plot with adjusted layout
    compare_simulated_paths_with_adjusted_layout(dynamics, n_paths, random_seed, T)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    plotting_ex0()