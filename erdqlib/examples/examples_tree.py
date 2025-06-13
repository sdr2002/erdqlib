import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, Any, List

from erdqlib.src.common.option import OptionType, OptionSide
from erdqlib.src.tree.binomial import binomial_option_price_from_vol, binomial_option_price_asian
from erdqlib.src.tree.trinomial import TrinomialEurCall, TrinomialEurPut, TrinomialAmeCall, TrinomialAmePut
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def q5_to_q10():
    kw_ex = dict(S_ini=100, K=100, T=1 / 4, r=0.05, sigma=0.2)
    n_steps_list: List[int] = [3, 12, 60, 480]

    # Q5, Q6
    for n_steps in n_steps_list:
        V, S, D = binomial_option_price_from_vol(**kw_ex, N=n_steps, opttype=OptionType.EUROPEAN, side=OptionSide.CALL)
        LOGGER.info(f"{V[0, 0]}, {D[0, 0]}")

    for n_steps in n_steps_list:
        V, S, D = binomial_option_price_from_vol(**kw_ex, N=n_steps, opttype=OptionType.EUROPEAN, side=OptionSide.CALL)
        LOGGER.info(f"{V[0, 0]}, {D[0, 0]}")

    # Q7
    N=480
    V, S, D = binomial_option_price_from_vol(
        S_ini=100, K=100, T=1 / 4, r=0.05, sigma=0.2, N=N,
        opttype=OptionType.EUROPEAN, side=OptionSide.CALL
    )
    V_i = V[0, 0]
    V, S, D = binomial_option_price_from_vol(
        S_ini=100, K=100, T=1 / 4, r=0.05, sigma=0.25, N=N,
        opttype=OptionType.EUROPEAN, side=OptionSide.CALL
    )
    V_f = V[0, 0]

    LOGGER.info(f"Vi={V_i:.3g}, Vf={V_f:.3g}, Delta={(V_f - V_i) / ((0.25 - 0.2) * 100):.3g}")

    V, S, D = binomial_option_price_from_vol(
        S_ini=100, K=100, T=1 / 4, r=0.05, sigma=0.2, N=N,
        opttype=OptionType.EUROPEAN, side=OptionSide.PUT
    )
    V_i = V[0, 0]
    V, S, D = binomial_option_price_from_vol(
        S_ini=100, K=100, T=1 / 4, r=0.05, sigma=0.25, N=N,
        opttype=OptionType.EUROPEAN, side=OptionSide.PUT
    )
    V_f = V[0, 0]

    LOGGER.info(f"Vi={V_i:.3g}, Vf={V_f:.3g}, Delta={(V_f - V_i) / ((0.25 - 0.2) * 100):.3g}")

    # Q8, Q9
    opttype_ex = OptionType.AMERICAN
    for side_ex in [OptionSide.CALL, OptionSide.PUT]:
        LOGGER.info(side_ex)
        for n_steps in n_steps_list:
            V, S, D = binomial_option_price_from_vol(**kw_ex, N=n_steps, opttype=opttype_ex, side=side_ex)
            LOGGER.info(f"{V[0, 0]}, {D[0, 0]}")

    # Q10
    V, S, D = binomial_option_price_from_vol(
        S_ini=100, K=100, T=1 / 4, r=0.05, sigma=0.2, N=N,
        opttype=OptionType.AMERICAN, side=OptionSide.CALL
    )
    V_i = V[0, 0]
    V, S, D = binomial_option_price_from_vol(
        S_ini=100, K=100, T=1 / 4, r=0.05, sigma=0.25, N=N,
        opttype=OptionType.AMERICAN, side=OptionSide.CALL
    )
    V_f = V[0, 0]

    LOGGER.info(f"Vi={V_i:.3g}, Vf={V_f:.3g}, Delta={(V_f - V_i) / ((0.25 - 0.2) * 100):.3g}")

    V, S, D = binomial_option_price_from_vol(
        S_ini=100, K=100, T=1 / 4, r=0.05, sigma=0.2, N=N,
        opttype=OptionType.AMERICAN, side=OptionSide.PUT
    )
    V_i = V[0, 0]
    V, S, D = binomial_option_price_from_vol(
        S_ini=100, K=100, T=1 / 4, r=0.05, sigma=0.25, N=N,
        opttype=OptionType.AMERICAN, side=OptionSide.PUT
    )
    V_f = V[0, 0]

    LOGGER.info(f"Vi={V_i:.3g}, Vf={V_f:.3g}, Delta={(V_f - V_i) / ((0.25 - 0.2) * 100):.3g}")


def q15_to_22():
    V, S = TrinomialEurCall(S_ini=100.0, K=90.0, r=0.0, T=1.0, sigma=0.3).price(N=2)
    LOGGER.info(V)
    LOGGER.info(S)

    # Q15,16,17,18
    kw_ex = dict(S_ini=100, T=1 / 4, r=0.05, sigma=0.2)
    N = 300

    price_df = pd.DataFrame(dict(Type=[], OptionSide=[], K_strike=[], V_option=[], S_underlying=[]))
    # Binomial tree showcase
    for opttype in [OptionType.EUROPEAN, OptionType.AMERICAN]:
        for side in OptionSide:
            # LOGGER.info(side)
            for moneyness in [0.9, 0.95, 1.0, 1.05, 1.1]:
                K = kw_ex['S_ini'] * moneyness
                # LOGGER.info(f' K={K:.2f}')
                if opttype == OptionType.EUROPEAN and side == OptionSide.CALL:
                    V, S = TrinomialEurCall(**kw_ex, K=K).price(N=N)
                elif opttype == OptionType.EUROPEAN and side == OptionSide.PUT:
                    V, S = TrinomialEurPut(**kw_ex, K=K).price(N=N)
                elif opttype == OptionType.AMERICAN and side == OptionSide.CALL:
                    V, S = TrinomialAmeCall(**kw_ex, K=K).price(N=N)
                elif opttype == OptionType.AMERICAN and side == OptionSide.PUT:
                    V, S = TrinomialAmePut(**kw_ex, K=K).price(N=N)
                else:
                    raise ValueError(f"Invalid OptionSide and OptionType: {side}/{OptionType}")

                # LOGGER.info('  ', V[0,0])            
                price_df.loc[len(price_df)] = dict(
                    Type=opttype, OptionSide=side, K_strike=K, V_option=V[0, 0],
                    S_underlying=S[0, 0]
                )

    LOGGER.info(price_df.to_markdown(index=False))

    # Q19
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))

    opt_type = OptionType.EUROPEAN
    px_dict = {}
    for side in OptionSide:
        strike_px = price_df[(price_df.Type == opt_type) & (price_df.OptionSide == side)]
        strike_px[["K_strike", "V_option"]].plot(
            x="K_strike", y="V_option", style='.-', label=side, ax=axs[0]
        )
        px_dict[side] = strike_px[["K_strike", "V_option", "S_underlying"]].set_index("K_strike")

    axs[0].set_ylabel("option price")
    axs[0].legend(loc="lower left")
    axs[0].set_title(f"{opt_type} V_option by K_strike")

    # Q23 European option parity by rendering $c_0 - p_0= S_0 - K e^{-r T}$
    c_p = (px_dict[OptionSide.CALL] - px_dict[OptionSide.PUT])[["V_option"]].rename(columns={"V_option": "Call-Put"})
    c_p.plot(ax=axs[1], style='.-', alpha=0.3)

    px_dict[OptionSide.CALL].apply(lambda row: row["S_underlying"] - row.name * np.exp(-kw_ex['r'] * kw_ex['T']),
                                   axis=1).to_frame("S0-K*exp(-r*T)").plot(
        ax=axs[1], style='.-', alpha=0.3
    )
    axs[1].set_ylabel("Parity")
    axs[1].legend(loc="lower left")
    axs[1].set_title(f"{opt_type} parity")

    plt.tight_layout()
    plt.show()

    # Q20
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))

    opt_type = OptionType.AMERICAN
    px_dict = {}
    for side in OptionSide:
        strike_px = price_df[(price_df.Type == opt_type) & (price_df.OptionSide == side)]
        strike_px[["K_strike", "V_option"]].plot(
            x="K_strike", y="V_option", style='.-', label=side, ax=axs[0]
        )
        px_dict[side] = strike_px[["K_strike", "V_option", "S_underlying"]].set_index("K_strike")

    axs[0].set_ylabel("option price")
    axs[0].legend(loc="lower left")
    axs[0].set_title(f"{opt_type} V_option by K_strike")

    # Q24
    c_p = (px_dict[OptionSide.CALL] - px_dict[OptionSide.PUT])[["V_option"]].rename(columns={"V_option": "Call-Put"})
    c_p.plot(ax=axs[1], style='.-', alpha=0.3)

    px_dict[OptionSide.CALL].apply(lambda row: row["S_underlying"] - row.name * np.exp(-kw_ex['r'] * kw_ex['T']),
                                   axis=1).to_frame("S0-K*exp(-r*T)").plot(
        ax=axs[1], style='.-', alpha=0.3
    )
    axs[1].set_ylabel("Parity")
    axs[1].legend(loc="lower left")
    axs[1].set_title(f"{opt_type} parity")

    plt.tight_layout()
    plt.show()

    # Q21
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    side = OptionSide.CALL
    for opt_type in [OptionType.EUROPEAN, OptionType.AMERICAN]:
        price_df[(price_df.Type == opt_type) & (price_df.OptionSide == side)][["K_strike", "V_option"]].plot(
            x="K_strike", y="V_option", style='.-', label=opt_type, ax=ax, alpha=0.3
        )
    ax.set_ylabel("option price")
    ax.set_title(f"{side} V_option by K_strike")
    plt.show()

    # Q22
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    side = OptionSide.PUT
    for opt_type in [OptionType.EUROPEAN, OptionType.AMERICAN]:
        price_df[(price_df.Type == opt_type) & (price_df.OptionSide == side)][["K_strike", "V_option"]].plot(
            x="K_strike", y="V_option", style='.-', label=opt_type, ax=ax, alpha=0.3
        )
    ax.set_ylabel("option price")
    ax.set_title(f"{side} V_option by K_strike")
    plt.show()


def _option_pricing_and_delta_hedging_process(
        kw_ex: Dict[str, Any], path: List[int]
):
    """Perform option pricing and dynamic delta hedging process"""
    LOGGER.info(f"Run with {kw_ex}")
    N: int = int(kw_ex['N'])
    opttype: OptionType = kw_ex['opttype']

    match opttype:
        case OptionType.EUROPEAN | OptionType.AMERICAN:
            V, S, D = binomial_option_price_from_vol(**kw_ex)

            V0: float = V[0, 0]
            D0: float = D[0, 0]

            S_path = S[np.arange(D.shape[0]), path]
            V_path = V[np.arange(D.shape[0]), path]
            D_path = D[np.arange(D.shape[0]), path][:-1]

        case OptionType.ASIAN:
            S, Savg, V, D = binomial_option_price_asian(**kw_ex)

            V0: float = V[0][0][()][0]
            D0: float = D[0][0][()]

            # path to action chain ex> [0,1,1,1] -> [u,d,d]
            ud = np.array(np.array(path[1:]) - np.array(path[:-1]), dtype=int)

            S_path = S[np.arange(N + 1), path]

            V_path = [V[0][0][()][0]]
            for i in range(1, N + 1):
                V_i = V[i][path[i]][tuple(ud[:i])][0]
                V_path.append(V_i)
            V_path = np.array(V_path)

            D_path = [D[0][0][()]]
            for i in range(1, N):
                D_i = D[i][path[i]][tuple(ud[:i])]
                D_path.append(D_i)
            D_path = np.array(D_path)

    # Logging
    LOGGER.info(f"V0, D0:\n{V0:.4g}, {D0:.4g}")
    LOGGER.info(f"Stock tree:\n{S}")  # underlying
    LOGGER.info(f"Option tree:\n{V}")  # option
    LOGGER.info(f"Delta tree:\n{D}")  # delta

    LOGGER.info(f"Stock path:\n{S_path}")
    LOGGER.info(f"Option path:\n{V_path}")
    LOGGER.info(f"Delta path:\n{D_path}")

    # cash flow path
    C_path = -np.concatenate((
        [D_path[0] * S_path[0]],
        (D_path[1:] - D_path[:-1]) * S_path[1:-1],
        [- D_path[-1] * S_path[-1] + V_path[-1]]
    ), axis=0)
    LOGGER.info(f"Cashflow path \n{C_path}")
    dcf_path = np.exp(-kw_ex['r'] * kw_ex['T'] * (np.arange(N + 1) / N))
    LOGGER.info(f"Discount arr  \n{dcf_path}")
    pv0_sumC = np.sum(C_path * dcf_path)
    LOGGER.info(f"PV0[Sum(Cashflow)] {pv0_sumC}")

    if np.abs(V0 - np.abs(pv0_sumC)) / np.abs(pv0_sumC) > 1e-9:  # rtol as 1e-9
        LOGGER.info("Pricing model is not Arbitrage-free")
    else:
        LOGGER.info("Pricing model is Arbitrage-free as V0 + PV(t=0)[all cash flows] is approximately 0")


def q25_to_27():
    """Examples for dynamic hedging process for a path and arbitrage-free property"""

    # Q25
    _option_pricing_and_delta_hedging_process(
        kw_ex=dict(
            S_ini=180, K=182, T=1 / 2, r=0.02, sigma=0.25, N=3, side=OptionSide.PUT, opttype=OptionType.EUROPEAN
        ),
        path=[0, 1, 1, 1]
    )

    # Q26
    _option_pricing_and_delta_hedging_process(
        kw_ex=dict(
            S_ini=180, K=182, T=1 / 2, r=0.02, sigma=0.25, N=3, side=OptionSide.PUT, opttype=OptionType.AMERICAN
        ),
        path=[0, 1, 1, 1]
    )

    # Q27
    _option_pricing_and_delta_hedging_process(
        kw_ex=dict(
            S_ini=180, K=182, T=1 / 2, r=0.02, sigma=0.25, N=3, side=OptionSide.PUT, opttype=OptionType.ASIAN
        ),
        path=[0, 1, 1, 1]
    )


if __name__ == "__main__":
    q5_to_q10()
    q15_to_22()
    q25_to_27()
