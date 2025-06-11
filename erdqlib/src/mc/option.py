import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import r2_score

from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType
from erdqlib.src.mc.dynamics import ModelParameters

logging.basicConfig(
    level=logging.INFO,
    format='\033[97m[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s\033[0m',
    datefmt='%H:%M:%S'
)
LOGGER = logging.getLogger(__name__)


def price_montecarlo(
    Spath: np.ndarray, d: ModelParameters, o: OptionInfo,
    t: float = 0., verbose: bool = False
) -> float:
    LOGGER.info(o.__dict__)
    assert isinstance(o.side, OptionSide)

    payoff: np.ndarray
    match o.type:
        case OptionType.EUROPEAN:
            if o.side == OptionSide.CALL:
                payoff = np.maximum(0, Spath[-1, :] - o.K)
            else:
                payoff = np.maximum(0, - Spath[-1, :] + o.K)
        case OptionType.DOWNANDIN:
            # Down-and-In payoff:
            #   Payoff = European payoff * 1_{min S_path <= barrier}
            #   payoff = max(S_T - K, 0) * 1_{min S_t ≤ B}  for calls,
            #          = max(K - S_T, 0) * 1_{min S_t ≤ B}  for puts.
            # Compute the indicator of barrier breach:
            assert type(o) is BarrierOptionInfo
            knock_in = (Spath.min(axis=0) <= o.barrier)
            if o.side == OptionSide.CALL:
                euro_payoff = np.maximum(0, Spath[-1, :] - o.K)
            else:
                euro_payoff = np.maximum(0, o.K - Spath[-1, :])
            payoff = euro_payoff * knock_in
        case OptionType.UPANDIN:
            assert type(o) is BarrierOptionInfo
            knock_in = (Spath.max(axis=0) >= o.barrier)
            if o.side == OptionSide.CALL:
                euro_payoff = np.maximum(0, Spath[-1, :] - o.K)
            else:
                euro_payoff = np.maximum(0, o.K - Spath[-1, :])
            payoff = euro_payoff * knock_in
        case OptionType.AMERICAN:
            # Longstaff–Schwartz for American exercise:
            dt = (d.T - t) / d.M
            disc = np.exp(-d.r * dt)

            # 1) Initialize cashflows at maturity:
            if o.side == OptionSide.CALL:
                cf = np.maximum(Spath[-1, :] - o.K, 0)
            else:
                cf = np.maximum(o.K - Spath[-1, :], 0)

            # 2) Step backwards through t = M-1, ..., 1
            if verbose:
                np.set_printoptions(formatter={'float': lambda x: "{0:0.3g}".format(x)})
                r2_list = []
            for ti in range(d.M, 0, -1):
                # discount next‐step cashflow back to time ti
                cf *= disc

                St = Spath[ti]
                # immediate payoff if exercised at ti:
                if o.side == OptionSide.CALL:
                    intrinsic = np.maximum(St - o.K, 0)
                else:
                    intrinsic = np.maximum(o.K - St, 0)

                # only regress on in‐the‐money paths
                itm = intrinsic > 0
                if np.any(itm):
                    # fit a polynomial continuation value
                    res: Polynomial = Polynomial.fit(
                        x=St[itm],
                        y=cf[itm],
                        deg=5
                    ) #.convert()

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
        case _:
            raise TypeError()

    discount: float = np.exp(-d.r * (d.T - t))
    average_payoff: float = np.mean(payoff)
    return discount * average_payoff

# TODO add greek calculators: delta, gamma, vega as functions just like price calculator
