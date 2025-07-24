from dataclasses import dataclass
from typing import Optional, Tuple, SupportsFloat

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splrep

from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)

def capitalization_factor(r_year: float | np.ndarray, t_year: float | np.ndarray) -> float | np.ndarray:
    """Capitalization factor: 1 + n_year(convention) * rate_year"""
    if np.any(t_year <= 0):
        raise ValueError("Time in years must be positive")
    return 1. + t_year * r_year


def zero_rate(capitalisation_factor: float | np.ndarray, t_year: float | np.ndarray) -> float | np.ndarray:
    """Annualized continuous rate from capitalization factor and time in years"""
    if np.any(t_year <= 0):
        raise ValueError("Time in years must be positive")
    return np.log(capitalisation_factor) / t_year


def implied_yield(t_year: float, price_0_t: float, price_t_t: float = 1.0):
    """Implied yield from zero-coupon bond prices at time maturity t and valuation time 0

     Y(0,t) = -log(B_t(t)/B_0(t)) / t
     If B_t(t) is assumed to be 1 as it is the face value of the bond, then Y = -log(B(0;t))/t
     """
    return np.log(price_t_t / price_0_t) / t_year


def instantaneous_rate(t_year: float, price_0_t: float) -> float:
    """Instantaneous rate from Zero-Coupon-Bond price and time in years"""
    if t_year <= 0:
        raise ValueError("Time in years must be positive")
    return implied_yield(t_year=t_year, price_0_t=price_0_t, price_t_t=1.0)


@dataclass
class ForwardsLadder:
    maturities: np.ndarray
    rates: np.ndarray

    def __init__(self, maturities: np.ndarray, rates: np.ndarray):
        if len(maturities) != len(rates):
            raise ValueError("Maturities and rates must have the same length")
        self.maturities = maturities
        self.rates = rates


class SplineCurve:
    """Curve class to hold maturities and rates"""
    def __init__(self):
        self.maturities_data: Optional[np.ndarray] = None
        self.ytms_data: Optional[np.ndarray] = None
        self.bspline: Optional[Tuple[np.ndarray, np.ndarray, int]] = None

        self.t_i: Optional[float] = None
        self.t_f: Optional[float] = None

        self.forwards_ladder: Optional[ForwardsLadder] = None
        self._interpolated_rates: Optional[np.ndarray] = None
        self._first_derivatives: Optional[np.ndarray] = None

    def update_curve(
            self,
            maturities: np.ndarray,
            spot_rates: np.ndarray,
            n_knots: int = 3
    ):
        """Construct a curve from maturities and rates"""
        # Capitalization factors and Zero-rates
        zcb_rates: np.ndarray = zero_rate(
            capitalisation_factor=capitalization_factor(
                r_year=spot_rates,
                t_year=maturities
            ),
            t_year=maturities
        )  # Euribor is IR product which is a single cash flow, hence is a zero-coupon bond where YTM = spot-rate

        self.bspline = splrep(maturities, zcb_rates, k=n_knots)

        self.maturities_data = maturities
        self.ytms_data = zcb_rates

    def calculate_forward_rates(self, t_f: float, t_i: float = 0.0) -> ForwardsLadder:
        """Forward rates via Cubic spline"""
        if not self.bspline:
            raise ValueError("Curve is not initialized. Call update_curve() first.")

        maturities_ladder: np.ndarray = np.linspace(t_i, t_f, int((t_f - t_i)*24.))

        # Forward rate given a curve (of interpolated rates and their first derivatives)
        self._interpolated_rates = splev(maturities_ladder, self.bspline, der=0)  # Interpolated rates
        self._first_derivatives = splev(maturities_ladder, self.bspline, der=1)  # First derivative of spline
        zcb_forward_rates: np.ndarray = self._interpolated_rates + self._first_derivatives * maturities_ladder

        return ForwardsLadder(maturities=maturities_ladder, rates=zcb_forward_rates)

    def get_maturities_ladder(self) -> np.ndarray:
        if not self.forwards_ladder:
            raise ValueError("Maturities ladder is not calculated. Call get_forward_rates() first.")
        return self.forwards_ladder.maturities

    def get_forward_rates(self):
        """get_zcb_forward_rates"""
        if not self.forwards_ladder:
            raise ValueError("Forward rates are not calculated. Call get_forward_rates() first.")
        return self.forwards_ladder.rates

    def get_r0(self, source: str = "overnight") -> float:
        if not self.bspline:
            raise ValueError("Curve is not initialized. Call update_curve() first.")

        if source == "overnight":
            # Overnight rate estimated from the spline curve
            out = splev([1./365.], self.bspline, der=0)[0]  # type: ignore
        elif source == "nearest":
            # Used in the WQU instruction, but what if the nearest tenor in data was in 1W?
            out = self.ytms_data[0]
        else:
            raise ValueError(f"Invalid source: {source}. Use 'overnight' or 'nearest'.")

        if not isinstance(out, SupportsFloat):
            raise TypeError("Overnight rate is not a float. Check the spline data.")
        return float(out)

    def plot_curve(self):
        if not self.forwards_ladder:
            raise ValueError("Forward rates are not calculated. Call get_forward_rates() first.")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.maturities_data, self.ytms_data, "r.", markersize=15, label="Market quotes")
        ax.plot(self.get_maturities_ladder(), self._interpolated_rates, "--", markersize=10, label="Spot rate")
        ax.plot(self.get_maturities_ladder(), self._first_derivatives, "g--", markersize=10, label="Spot rate time derivative")
        ax.set_xlabel("Time Horizon")
        ax.set_ylabel("Zero forward rate")
        ax2 = ax.twinx()
        ax2.plot(self.get_maturities_ladder(), self.get_forward_rates(), "b--", markersize=10, label="Forward rate")
        fig.suptitle("Term Structure Euribor")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    # Example usage
    LOGGER.info(capitalization_factor(0.245 / 100, 4 / 12))  # 0.245% annual rate, 4 months