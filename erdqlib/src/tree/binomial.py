from typing import Tuple, List, Dict

import numpy as np

from erdqlib.src.common.option import OptionType, OptionSide
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def binomial_option_price_from_vol(
        S_ini, K, T, r, sigma, N, opttype: OptionType, side: OptionSide,
        u: float = None, d: float = None, print_prob: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binomial tree pricing model with risk-neutral probability of transition

    Output:
      V: N+1 (step) x N+1 (number of up transitted) ndarray containing the option price
      S: N+1 (step) x N+1 (number of up transitted) ndarray containing the stock price
      Delta: N+1 (step) x N+1 (number of up transitted) ndarray containing the delta
    """
    assert isinstance(side, OptionSide), "side must be type OptionSide"

    dt = T / N  # Define time step

    # up, down multiplier from sigma for CRR matching volatility model
    if not u or not d:
        u = np.exp(sigma * np.sqrt(dt))  # Define u
        d = np.exp(-sigma * np.sqrt(dt))  # Define d

    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probs
    if print_prob:
        LOGGER.info("P(up)=", p)
    V = np.zeros([N + 1, N + 1])  # option prices
    S = np.zeros([N + 1, N + 1])  # underlying price

    # Option greeks
    Delta = np.zeros((N + 1, N + 1))  # we'll fill only rows 0..N-1

    for i in range(0, N + 1):
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
        if side == OptionSide.CALL:
            V[N, i] = max(S[N, i] - K, 0)
        else:
            V[N, i] = max(K - S[N, i], 0)

    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            V[j, i] = np.exp(-r * dt) * (
                    p * V[j + 1, i + 1] + (1 - p) * V[j + 1, i]
            )  # Computing the European option prices
            S[j, i] = (
                    S_ini * (u ** (i)) * (d ** (j - i))
            )  # Underlying evolution for each node
            if opttype == OptionType.EUROPEAN:
                pass
            elif opttype == OptionType.AMERICAN:
                if side == OptionSide.CALL:
                    V[j, i] = max(
                        V[j, i], S[j, i] - K
                    )  # Decision between the European option price and the payoff from early-exercise
                else:
                    V[j, i] = max(
                        V[j, i], K - S[j, i]
                    )  # Decision between the European option price and the payoff from early-exercise
            else:
                raise ValueError("opttype invalid")

            # Delta = dV/dS
            Delta[j, i] = (V[j + 1, i + 1] - V[j + 1, i]) / (
                    S[j + 1, i + 1] - S[j + 1, i]
            )

    return V, S, Delta


def binomial_option_price_asian(
        S_ini: float,
        K: float,
        T: float,
        r: float,
        N: int,
        side: OptionSide,
        opttype: OptionType,
        sigma: float = None,
        u: float = None,
        d: float = None,
) -> Tuple[
    List[List[Dict[Tuple[int, ...], Tuple[float, float]]]],
    List[List[Dict[Tuple[int, ...], Tuple[float, float]]]],
    List[List[Dict[Tuple[int, ...], float]]]
]:
    """
    Returns Savg_paths, V_paths, Delta_paths each as (N+1)x(N+1) list of dicts.
    Savg_paths[j][i][path] = (average_of_S along path, prob(path))
    V_paths[j][i][path] = (option_value at node, prob(path))
    Delta_paths[j][i][path] = local delta at node (no prob)
    """
    assert opttype == OptionType.ASIAN

    dt = T / N
    if u is None or d is None:
        assert sigma, "sigma must not be None if u, d are None"
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    dcf = np.exp(-r * dt)
    LOGGER.info(f"dt={dt:.2f}, u/d={u:.2f}{d:.2f}, P(up)={p:.2f}, dcf={dcf:.4f}")

    # initialize structures
    Sarr = np.zeros([N + 1, N + 1])  # underlying price
    Ssum_paths = [[{} for _ in range(j + 1)] for j in range(N + 1)]
    V_paths = [[{} for _ in range(j + 1)] for j in range(N + 1)]
    Delta_paths = [[{} for _ in range(j + 1)] for j in range(N + 1)]

    # step 0: only empty path
    Ssum_paths[0][0][()] = (S_ini, 1.0)

    # forward: build Ssum_paths
    for j in range(N + 1):
        for i in range(j + 1):
            if j == 0 and i == 0:
                Sarr[0, 0] = S_ini
            elif j != 0:
                if j == i:
                    Sarr[j, i] = Sarr[j - 1, i - 1] * u
                else:
                    Sarr[j, i] = Sarr[j - 1, i] * d

            if j == N:
                continue

            for path, (sumS, prob) in Ssum_paths[j][i].items():
                # up move
                Su = S_ini * (u ** (sum(path) + 1)) * (d ** (j + 1 - (sum(path) + 1)))
                pu = prob * p
                path_u = path + (1,)
                Ssum_paths[j + 1][i + 1][path_u] = (float(sumS + Su), float(pu))
                # down move
                Sd = S_ini * (u ** sum(path)) * (d ** (j + 1 - sum(path)))
                pd = prob * (1 - p)
                path_d = path + (0,)
                Ssum_paths[j + 1][i][path_d] = (float(sumS + Sd), float(pd))

    # at maturity, fill V_paths[N]
    for i in range(N + 1):
        for path, (sumS, prob) in Ssum_paths[N][i].items():
            avg = sumS / (N + 1)
            if side == OptionSide.CALL:
                payoff = max(avg - K, 0)
            elif side == OptionSide.PUT:
                payoff = max(K - avg, 0)
            else:
                raise ValueError("Invalid side: ", side)
            V_paths[N][i][path] = (float(payoff), float(prob))

    # backward: fill V_paths and Delta_paths
    for j in reversed(range(N)):
        for i in range(j + 1):
            for path, (sumS, prob) in Ssum_paths[j][i].items():
                # children
                pu = p
                pd = 1 - p
                path_u = path + (1,)
                path_d = path + (0,)
                Vu, _ = V_paths[j + 1][i + 1][path_u]
                Vd, _ = V_paths[j + 1][i][path_d]
                # continuation value
                cont = dcf * (pu * Vu + pd * Vd)
                V_paths[j][i][path] = (float(cont), float(prob))

                # stock prices at children
                Su = Ssum_paths[j + 1][i + 1][path_u][0] - sumS + S_ini * (u ** (sum(path) + 1)) * d ** (
                            j + 1 - (sum(path) + 1))
                Sd = Ssum_paths[j + 1][i][path_d][0] - sumS + S_ini * (u ** sum(path)) * d ** (j + 1 - sum(path))
                # but easier: recompute S_t:
                S_t = S_ini * (u ** sum(path)) * (d ** (j - sum(path)))
                Su = S_t * u
                Sd = S_t * d

                # local delta
                Delta_paths[j][i][path] = (float(Vu) - float(Vd)) / (float(Su) - float(Sd))

    Savg_paths = [[] for _ in range(N + 1)]
    for i_row, S_row in enumerate(Ssum_paths):
        Savg_row = []
        for cell_dict in S_row:
            cell_Savg_dict = {}
            for ud_chain, Ssum_prob_tup in cell_dict.items():
                cell_Savg_dict[ud_chain] = (Ssum_prob_tup[0] / (i_row + 1), Ssum_prob_tup[1])
            Savg_row.append(cell_Savg_dict)
        Savg_paths[i_row] = Savg_row

    return Sarr, Savg_paths, V_paths, Delta_paths