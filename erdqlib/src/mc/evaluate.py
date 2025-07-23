import warnings
from dataclasses import dataclass
from typing import List, Callable, cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import r2_score

from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType, BarrierOptionInfo
from erdqlib.src.mc.dynamics import ModelParameters
from erdqlib.src.mc.dynamics import MonteCarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def price_montecarlo(
        underlying_path: np.ndarray, d: ModelParameters, o: OptionInfo,
        t: float = 0., verbose: bool = False
) -> float:
    LOGGER.info(o.__dict__)
    assert isinstance(o.side, OptionSide)

    payoff: np.ndarray
    match o.o_type:
        case OptionType.EUROPEAN:
            if o.side == OptionSide.CALL:
                payoff = np.maximum(0, underlying_path[-1, :] - o.K)
            else:
                payoff = np.maximum(0, - underlying_path[-1, :] + o.K)
        case OptionType.DOWNANDIN:
            o = cast(BarrierOptionInfo, o)  # ensure o is BarrierOptionInfo
            # Down-and-In payoff:
            #   Payoff = European payoff * 1_{min S_path <= barrier}
            #   payoff = max(S_T - K, 0) * 1_{min S_t ≤ B}  for calls,
            #          = max(K - S_T, 0) * 1_{min S_t ≤ B}  for puts.
            # Compute the indicator of barrier breach:
            assert type(o) is BarrierOptionInfo
            knock_in = (underlying_path.min(axis=0) <= o.barrier)
            if o.side == OptionSide.CALL:
                euro_payoff = np.maximum(0, underlying_path[-1, :] - o.K)
            else:
                euro_payoff = np.maximum(0, o.K - underlying_path[-1, :])
            payoff = euro_payoff * knock_in
        case OptionType.UPANDIN:
            o = cast(BarrierOptionInfo, o)  # ensure o is BarrierOptionInfo
            assert type(o) is BarrierOptionInfo
            knock_in = (underlying_path.max(axis=0) >= o.barrier)
            if o.side == OptionSide.CALL:
                euro_payoff = np.maximum(0, underlying_path[-1, :] - o.K)
            else:
                euro_payoff = np.maximum(0, o.K - underlying_path[-1, :])
            payoff = euro_payoff * knock_in
        case OptionType.AMERICAN:
            """Least-Squares Monte Carlo for American options, regressing continuation from the *next*‐timestep values.
        
            Notation:
              Δt = T/M
              π_t^{(j)} = payoff at time t on path j
              V_{t+1}^{(j)} = current estimate of option value at t+1 on path j
        
            Continuation:
              C_t(S) = E[e^{-rΔt}·V_{t+1} | S_t = S].
        
            Regression at each t:
              • X^{(j)} = S_t^{(j)}
              • Y^{(j)} = D^{(j)} = e^{-rΔt}·V_{t+1}^{(j)}
              Fit Y ≈ ∑_{k=0}^d a_k·(X)^k to learn coefficients a_k.
        
            Steps:
              1) Initialize V_next[j] = π_T^{(j)} for all j.
              2) For t = M−1 down to 1:
                   a) D[j] = e^{-rΔt} · V_next[j]              # discount to time t
                   b) I_t = {j | π_t^{(j)} > 0}               # in‐the‐money paths
                   c) Regress {D[j]}_{j∈I_t} on basis of {S_t^{(j)}}_{j∈I_t} to get a_k.
                   d) For each j:
                        V_current[j] = max(π_t^{(j)}, D[j]) if j∈I_t
                                       = D[j]               otherwise
                   e) V_next = V_current
              3) Return price ≈ mean( e^{-rΔt} · V_next ).
        
            Only ITM paths enter the regression, since only there
            π_t and C_t compete in the exercise decision.
            """
            dt = (d.T - t) / d.M
            disc = np.exp(-d.r * dt)

            # 1) Initialize cashflows at maturity:
            if o.side == OptionSide.CALL:
                cf = np.maximum(underlying_path[-1, :] - o.K, 0)
            else:
                cf = np.maximum(o.K - underlying_path[-1, :], 0)

            # 2) Step backwards through t = M-1, ..., 1
            r2_list: List[float] = []
            if verbose:
                np.set_printoptions(formatter={'float': lambda x: "{0:0.3g}".format(x)})  # type: ignore
            for ti in range(d.M, 0, -1):
                # discount next‐step cashflow back to time ti
                cf *= disc

                St = underlying_path[ti]
                # immediate payoff if exercised at ti:
                if o.side == OptionSide.CALL:
                    intrinsic = np.maximum(St - o.K, 0)
                else:
                    intrinsic = np.maximum(o.K - St, 0)

                # only regress on in‐the‐money paths
                itm = intrinsic > 0
                if sum(itm) >= 2: # at least 2 paths to regress on
                    # fit a polynomial continuation value
                    # Basis: [1, x, x^2, ..., x^degree] then apply np.linalg.lstsq, then weights__ @ basis__
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", np.exceptions.RankWarning)
                        res: Polynomial = Polynomial.fit(
                            x=St[itm],
                            y=cf[itm],
                            deg=min(sum(itm)-1, 2) # degree of polynomial + 1 (bias dim) must be leq number of points
                        )  # .convert()

                    # evaluate the fitted polynomial for continuation
                    continuation = res(St[itm])

                    if verbose:
                        # compute and log fit performance
                        r2: float = r2_score(y_true=cf[itm], y_pred=continuation)
                        LOGGER.info(
                            f"Polynomial fit MSE at step {ti}: (R2={r2:.3g} for N={int(np.sum(itm))}), by: {res.coef}"
                        )
                        r2_list.append(r2)

                    # exercise if immediate payoff > continuation
                    exercise = intrinsic[itm] > continuation
                    cf[itm][exercise] = intrinsic[itm][exercise]

            if verbose:
                ax = pd.DataFrame({'R2': reversed(r2_list)}).plot()
                ax.set_xlabel('Timestep')
                ax.set_ylabel('R2')
                ax.set_title('Longstaff R2 by timestamp')
                plt.show()

            # 3) After backward induction, CF is already discounted to time t
            return cf.mean()
        case OptionType.ASIAN:
            if o.side == OptionSide.CALL:
                payoff = np.maximum(0, np.mean(underlying_path,axis=0) - o.K)
            else:
                payoff = np.maximum(0, - np.mean(underlying_path,axis=0) + o.K)
        case _:
            raise TypeError(f"Unknown option type: {o.type}")

    discount: float = np.exp(-d.r * (d.T - t))
    average_payoff: float = float(np.mean(payoff))
    return discount * average_payoff


@dataclass
class DeltaResult:
    s0_up: float
    o_up: float
    s0_down: float
    o_down: float
    delta: float


def calculate_delta(
        s0: float, strike: float, option_type: OptionType,
        side: OptionSide, bump_ratio: float,
        model: MonteCarlo,
        model_params_creator: Callable[[float], ModelParameters]
) -> DeltaResult:
    """Calculate the delta of the option using finite difference approximation."""
    perturb_dict = {}
    perturb_s = [s0 * (1. - bump_ratio), s0 * (1. + bump_ratio)]
    perturb_o = []
    for bumped_s0 in perturb_s:
        model_params = model_params_creator(bumped_s0)
        s_arr2d: np.ndarray = model.calculate_paths(model_params, underlying_only=True)

        option_price: float = price_montecarlo(
            underlying_path=s_arr2d,
            d=model_params,
            o=OptionInfo(
                o_type=option_type, K=strike, side=side
            ),
            t=0.
        )

        perturb_dict[bumped_s0] = option_price
        perturb_o.append(option_price)

    delta: float = (perturb_o[1] - perturb_o[0]) / (perturb_s[1] - perturb_s[0])
    perturb_dict['delta'] = delta
    return DeltaResult(
        delta=float(delta),
        s0_up=float(perturb_s[0]),
        s0_down=float(perturb_s[1]),
        o_up=float(perturb_o[0]),
        o_down=float(perturb_o[1]),
    )


# TODO add greek calculators: gamma, vega as functions just like price calculator
