from erdqlib.pyq.mc.option import EuropeanOption
from erdqlib.pyq.mc.dynamics import GeometricBrownianMotion

def main() -> None:
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
    print(f"European Call Option Price (GBM): {PV_GBM:.2f}")
    print(f"European Call Option Price (Alternative GBM): {PV_GBM_approx:.2f}")

if __name__ == "__main__":
    main()