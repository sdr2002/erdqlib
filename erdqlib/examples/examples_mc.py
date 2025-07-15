import copy
from pprint import pformat
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType, BarrierOptionInfo
from erdqlib.src.mc.heston import HestonParameters, get_heston_paths, plot_heston_paths
from erdqlib.src.mc.jump import JumpParameters, get_jump_paths, plot_jump_paths
from erdqlib.src.mc.option import price_montecarlo
from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)

"""
Remember to use the following general parameters:
- S0 = 80
- r = 5.5%
- sigma = 35%
- Time to maturity = 3 months
"""
COMMON_PARAMS: Dict[str, Any] = dict(
    S0=80.,
    r=0.055,
    sigma=0.35,
    T=3. / 12.
)


def q8(skip_plot: bool = True):
    """8. Using the Merton Model,
    price an ATM European call and an ATM European put with jump intensity parameter equal to 0.75.
    """
    j_params: JumpParameters = JumpParameters(
        lambd=0.75,
        mu=-0.5,
        delta=0.22,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    S = get_jump_paths(j_params)
    if not skip_plot:
        plot_jump_paths(n=300, underlying_arr=S, j_params=j_params)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        Spath=S,
        d=j_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t=0.
    ):.3g}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        Spath=S,
        d=j_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t=0.
    ):.3g}")


def q9(skip_plot: bool = True):
    """8. Using the Merton Model,
    price an ATM European call and an ATM European put with jump intensity parameter equal to 0.25.
    """
    j_params: JumpParameters = JumpParameters(
        lambd=0.25,
        mu=-0.5,
        delta=0.22,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    S = get_jump_paths(j_params)
    if not skip_plot:
        plot_jump_paths(n=300, underlying_arr=S, j_params=j_params)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        Spath=S,
        d=j_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t=0.
    ):.3g}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        Spath=S,
        d=j_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t=0.
    ):.3g}")


def _delta_per_S0(
        Sinit: float, K: float, ud_ratio: float, lambd_list: List[float], option_type: OptionType = OptionType.EUROPEAN
) -> Dict[OptionSide, Dict[float, Dict[float, float]]]:
    pricing_dict = {}
    for side in OptionSide:
        per_lambda_dict = {}
        for lambd in lambd_list:
            # Vary S0 by 1% on both up/down side - note it's not 1bps but 1% for numerical stability
            perturb_dict = {}
            perturb_s = [Sinit * (1. - ud_ratio), Sinit * (1. + ud_ratio)]
            perturb_o = []
            for S0 in perturb_s:
                dynamics_params = copy.deepcopy(COMMON_PARAMS)
                dynamics_params['S0'] = S0

                j_params: JumpParameters = JumpParameters(
                    lambd=lambd,
                    mu=-0.5,
                    delta=0.22,

                    M=500,  # Total time steps
                    I=10000,  # Number of simulations
                    random_seed=0,
                    **dynamics_params
                )

                S: np.ndarray = get_jump_paths(j_params)

                option_price: float = price_montecarlo(
                    Spath=S,
                    d=j_params,
                    o=OptionInfo(
                        type=option_type, K=K, side=side
                    ),
                    t=0.
                )

                perturb_dict[S0] = option_price
                perturb_o.append(option_price)

            perturb_dict['delta'] = (perturb_o[1] - perturb_o[0]) / (perturb_s[1] - perturb_s[0])
            per_lambda_dict[lambd] = perturb_dict

        pricing_dict[side] = per_lambda_dict

    LOGGER.info(f"\n{pformat(pricing_dict)}")
    return pricing_dict


def q10():
    """10. Calculate delta and gamma for each of the options in Questions 8 and 9.
    (Hint: You can numerically approximate this by forcing a change in the variable of interest
    –i.e., underlying stock price and delta change—and recalculating the option price).
    """
    option_type: OptionType = OptionType.EUROPEAN
    K = 80.
    lambd_list = [0.25, 0.75]

    ud_ratio: float = 0.05
    S0_list = [K * (1. - ud_ratio), K * (1. + ud_ratio)]

    pricing_dict_delta = _delta_per_S0(Sinit=K, K=K, ud_ratio=ud_ratio, lambd_list=lambd_list, option_type=option_type)
    '''
{<OptionSide.CALL: 'call'>: {0.25: {79.2: np.float64(4.789952555672441),
                                    80.8: np.float64(5.592979166793972),
                                    'delta': np.float64(0.5018916319509588)},
                             0.75: {79.2: np.float64(3.3941969793849016),
                                    80.8: np.float64(4.002606389671839),
                                    'delta': np.float64(0.38025588142933714)}},
 <OptionSide.PUT: 'put'>: {0.25: {79.2: np.float64(7.347448077028909),
                                  80.8: np.float64(6.608049790453335),
                                  'delta': np.float64(-0.4621239291097351)},
                           0.75: {79.2: np.float64(11.530334389123613),
                                  80.8: np.float64(10.709018737842372),
                                  'delta': np.float64(-0.5133222820507772)}}}
    '''
    delta_dict = {'side': [], 'lambda': [], 'delta': []}
    for side in OptionSide:
        for lambd in lambd_list:
            px_down: float = pricing_dict_delta[side][lambd][S0_list[0]]
            px_up: float = pricing_dict_delta[side][lambd][S0_list[1]]

            delta: float = (px_up - px_down) / (S0_list[1] - S0_list[0])
            delta_dict['side'].append(side)
            delta_dict['lambda'].append(lambd)
            delta_dict['delta'].append(f"{delta:.2g}")

    LOGGER.info(f"\n{pd.DataFrame(delta_dict).to_markdown(index=False)}")

    pricing_dict_down = _delta_per_S0(Sinit=S0_list[0], K=K, ud_ratio=ud_ratio, lambd_list=lambd_list,
                                      option_type=option_type)
    '''
{<OptionSide.CALL: 'call'>: {0.25: {72.2: np.float64(2.1181858004349796),
                                    79.8: np.float64(5.082475339679382),
                                    'delta': np.float64(0.3900380972690006)},
                             0.75: {72.2: np.float64(1.4264433950876434),
                                    79.8: np.float64(3.6151771857863673),
                                    'delta': np.float64(0.2879912882498323)}},
 <OptionSide.PUT: 'put'>: {0.25: {72.2: np.float64(11.423790249216312),
                                  79.8: np.float64(7.061561524399438),
                                  'delta': np.float64(-0.5739774637916943)},
                           0.75: {72.2: np.float64(15.817627949187132),
                                  79.8: np.float64(11.215167697437012),
                                  'delta': np.float64(-0.6055868752302794)}}}
    '''

    pricing_dict_up = _delta_per_S0(Sinit=S0_list[1], K=K, ud_ratio=ud_ratio, lambd_list=lambd_list,
                                    option_type=option_type)
    '''
{<OptionSide.CALL: 'call'>: {0.25: {79.8: np.float64(5.082475339679382),
                                    88.2: np.float64(10.112168503875601),
                                    'delta': np.float64(0.5987729957376448)},
                             0.75: {79.8: np.float64(3.6151771857863673),
                                    88.2: np.float64(7.5535368610850435),
                                    'delta': np.float64(0.46885234229746114)}},
 <OptionSide.PUT: 'put'>: {0.25: {79.8: np.float64(7.061561524399438),
                                  88.2: np.float64(3.9935239756858136),
                                  'delta': np.float64(-0.36524256532305027)},
                           0.75: {79.8: np.float64(11.215167697437012),
                                  88.2: np.float64(7.64747079950275),
                                  'delta': np.float64(-0.42472582118264995)}}}
    '''

    gamma_dict = {'side': [], 'lambda': [], 'gamma': []}
    for side in OptionSide:
        for lambd in lambd_list:
            delta_down: float = pricing_dict_down[side][lambd]['delta']
            delta_up: float = pricing_dict_up[side][lambd]['delta']

            gamma: float = (delta_up - delta_down) / (S0_list[1] - S0_list[0])
            gamma_dict['side'].append(side)
            gamma_dict['lambda'].append(lambd)
            gamma_dict['gamma'].append(f"{gamma:.2g}")

    LOGGER.info(f"\n{pd.DataFrame(gamma_dict).to_markdown(index=False)}")
    '''
       side  lambda     gamma
    0  call    0.25  0.027454
    1  call    0.75  0.023583
    2   put    0.25  0.025878
    3   put    0.75  0.023183
    '''


def _q15(S0: float, side: OptionSide, skip_plot: bool = True):
    """8. Using the Merton Model,
    price an ATM European call and an ATM European put with jump intensity parameter equal to 0.75.
    """
    params_to_overide = copy.deepcopy(COMMON_PARAMS)
    params_to_overide['S0'] = S0
    j_params: JumpParameters = JumpParameters(
        lambd=0.75,
        mu=-0.5,
        delta=0.22,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **params_to_overide
    )

    S = get_jump_paths(j_params)
    if not skip_plot:
        plot_jump_paths(n=300, underlying_arr=S, j_params=j_params)

    K = barrier = 65.

    price_dict = {'type': [], 'side': [], 'S0': [], 'K': [], 'Barrier': [], 'option_price': []}
    for o_type in [OptionType.EUROPEAN, OptionType.DOWNANDIN]:
        if o_type == OptionType.EUROPEAN:
            o_info = OptionInfo(
                type=OptionType.EUROPEAN, K=K, side=side,
            )
        elif o_type == OptionType.DOWNANDIN:
            o_info = BarrierOptionInfo(
                type=OptionType.DOWNANDIN, K=K, side=side,
                barrier=barrier
            )
        else:
            raise ValueError()

        option_price: float = price_montecarlo(
            Spath=S,
            d=j_params,
            o=o_info,
            t=0.
        )
        LOGGER.info(f"{o_type} {side}: {option_price:.3g}")

        price_dict['type'].append(o_type)
        price_dict['side'].append(o_info.side)
        price_dict['S0'].append(j_params.S0)
        price_dict['K'].append(K)
        price_dict['Barrier'].append(barrier)
        price_dict['option_price'].append(f"{option_price:.4g}")

    LOGGER.info(f"\n{pd.DataFrame(price_dict).to_markdown(index=False)}")


def q15(skip_plot=True):
    _q15(S0=65., side=OptionSide.PUT, skip_plot=skip_plot)

    _q15(S0=95., side=OptionSide.CALL, skip_plot=skip_plot)
    _q15(S0=80., side=OptionSide.CALL, skip_plot=skip_plot)
    _q15(S0=65., side=OptionSide.CALL, skip_plot=skip_plot)


def q5(skip_plot: bool = True):
    """5. Using the Heston Model and Monte-Carlo simulation, price an ATM European call
    and an ATM European put, using a correlation value of -0.30.
    """
    rho = -0.3

    h_params: HestonParameters = HestonParameters(
        v0=0.032,
        kappa=1.85,
        theta=0.045,
        rho=rho,
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    V, S = get_heston_paths(h_params)
    if not skip_plot:
        plot_heston_paths(n=300, underlying_arr=S, variance_arr=V, h_params=h_params)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        Spath=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        Spath=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t=0.
    )}")


def q6(skip_plot: bool = True):
    """5. Using the Heston Model and Monte-Carlo simulation, price an ATM European call
    and an ATM European put, using a correlation value of -0.30.
    """
    rho = -0.7

    h_params: HestonParameters = HestonParameters(
        v0=0.032,
        kappa=1.85,
        theta=0.045,
        rho=rho,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    V, S = get_heston_paths(h_params)
    if not skip_plot:
        plot_heston_paths(n=300, underlying_arr=S, variance_arr=V, h_params=h_params)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        Spath=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        Spath=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t=0.
    )}")


def _delta_per_S0_q7(
    Sinit: float, K: float, ud_ratio: float, rho_list: List[float], option_type: OptionType = OptionType.EUROPEAN
) -> Dict[OptionSide, Dict[float, Dict[float, float]]]:
    pricing_dict = {}
    for side in OptionSide:
        per_lambda_dict = {}
        for rho in rho_list:
            # Vary S0 by 1% on both up/down side - note it's not 1bps but 1% for numerical stability
            perturb_dict = {}
            perturb_s = [Sinit * (1. - ud_ratio), Sinit * (1. + ud_ratio)]
            perturb_o = []
            for S0 in perturb_s:
                dynamics_params = copy.deepcopy(COMMON_PARAMS)
                dynamics_params['S0'] = S0

                h_params: HestonParameters = HestonParameters(
                    v0=0.032,
                    kappa=1.85,
                    theta=0.045,
                    rho=rho,

                    M=500,  # Total time steps
                    I=10000,  # Number of simulations
                    random_seed=0,
                    **dynamics_params
                )

                V, S = get_heston_paths(h_params)

                option_price: float = price_montecarlo(
                    Spath=S,
                    d=h_params,
                    o=OptionInfo(
                        type=option_type, K=K, side=side
                    ),
                    t=0.
                )

                perturb_dict[S0] = option_price
                perturb_o.append(option_price)

            perturb_dict['delta'] = (perturb_o[1] - perturb_o[0]) / (perturb_s[1] - perturb_s[0])
            per_lambda_dict[rho] = perturb_dict

        pricing_dict[side] = per_lambda_dict

    LOGGER.info(f"\n{pformat(pricing_dict)}")
    return pricing_dict


def q7():
    """Calculate delta and gamma for each of the options in Questions 5 and 6. (Hint:
    You can numerically approximate this by forcing a change in the variable of interest
    –i.e., underlying stock price and delta change—and recalculating the option price).
    """
    option_type: OptionType = OptionType.EUROPEAN
    K = 80.
    rho_list = [-0.3, -0.7]

    ud_ratio: float = 0.01
    S0 = K * 0.9
    S0_list = [S0 * (1. - ud_ratio), S0 * (1. + ud_ratio)]

    pricing_dict_delta = _delta_per_S0_q7(Sinit=S0, K=K, ud_ratio=ud_ratio, rho_list=rho_list, option_type=option_type)
    delta_dict = {'side': [], 'rho': [], 'delta': []}
    for side in OptionSide:
        for rho in rho_list:
            px_down: float = pricing_dict_delta[side][rho][S0_list[0]]
            px_up: float = pricing_dict_delta[side][rho][S0_list[1]]

            delta: float = (px_up - px_down) / (S0_list[1] - S0_list[0])
            delta_dict['side'].append(side)
            delta_dict['rho'].append(rho)
            delta_dict['delta'].append(f"{delta:.2g}")

    LOGGER.info(f"\n{pd.DataFrame(delta_dict).to_markdown(index=False)}")

    pricing_dict_down = _delta_per_S0_q7(Sinit=S0_list[0], K=K, ud_ratio=ud_ratio, rho_list=rho_list,
                                         option_type=option_type)

    pricing_dict_up = _delta_per_S0_q7(Sinit=S0_list[1], K=K, ud_ratio=ud_ratio, rho_list=rho_list,
                                       option_type=option_type)

    gamma_dict = {'side': [], 'rho': [], 'gamma': []}
    for side in OptionSide:
        for rho in rho_list:
            delta_down: float = pricing_dict_down[side][rho]['delta']
            delta_up: float = pricing_dict_up[side][rho]['delta']

            gamma: float = (delta_up - delta_down) / (S0_list[1] - S0_list[0])
            gamma_dict['side'].append(side)
            gamma_dict['rho'].append(rho)
            gamma_dict['gamma'].append(f"{gamma:.2g}")

    LOGGER.info(f"\n{pd.DataFrame(gamma_dict).to_markdown(index=False)}")


def q13_1(skip_plot: bool = True):
    """
    Repeat Questions 5 and 7 for the case of an American call option (no need to
price the put). Comment on the differences you observe from original Questions
5 and 7.
    """
    rho = -0.3

    h_params: HestonParameters = HestonParameters(
        v0=0.032,
        kappa=1.85,
        theta=0.045,
        rho=rho,
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    V, S = get_heston_paths(h_params)
    if not skip_plot:
        plot_heston_paths(n=300, underlying_arr=S, variance_arr=V, h_params=h_params)

    LOGGER.info(f"AMR CALL: {price_montecarlo(
        Spath=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.AMERICAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t=0.,
        verbose=True
    )}")

    LOGGER.info(f"AMR PUT: {price_montecarlo(
        Spath=S,
        d=h_params,
        o=OptionInfo(
            type=OptionType.AMERICAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t=0.
    )}")


def q13_2():
    """Americal option call for heston - Delta and Gamma"""
    option_type: OptionType = OptionType.EUROPEAN
    K = 80.
    rho_list = [-0.3, -0.7]

    ud_ratio: float = 0.01
    S0 = K * 0.9
    S0_list = [S0 * (1. - ud_ratio), S0 * (1. + ud_ratio)]

    pricing_dict_delta = _delta_per_S0_q7(Sinit=S0, K=K, ud_ratio=ud_ratio, rho_list=rho_list, option_type=OptionType.AMERICAN)

    delta_dict = {'side': [], 'rho': [], 'delta': []}
    for side in OptionSide:
        for rho in rho_list:
            px_down: float = pricing_dict_delta[side][rho][S0_list[0]]
            px_up: float = pricing_dict_delta[side][rho][S0_list[1]]

            delta: float = (px_up - px_down) / (S0_list[1] - S0_list[0])
            delta_dict['side'].append(side)
            delta_dict['rho'].append(rho)
            delta_dict['delta'].append(f"{delta:.2g}")

    LOGGER.info(f"\n{pd.DataFrame(delta_dict).to_markdown(index=False)}")

    pricing_dict_down = _delta_per_S0_q7(Sinit=S0_list[0], K=K, ud_ratio=ud_ratio, rho_list=rho_list,
                                         option_type=option_type)

    pricing_dict_up = _delta_per_S0_q7(Sinit=S0_list[1], K=K, ud_ratio=ud_ratio, rho_list=rho_list,
                                       option_type=option_type)

    gamma_dict = {'side': [], 'rho': [], 'gamma': []}
    for side in OptionSide:
        for rho in rho_list:
            delta_down: float = pricing_dict_down[side][rho]['delta']
            delta_up: float = pricing_dict_up[side][rho]['delta']

            gamma: float = (delta_up - delta_down) / (S0_list[1] - S0_list[0])
            gamma_dict['side'].append(side)
            gamma_dict['rho'].append(rho)
            gamma_dict['gamma'].append(f"{gamma:.2g}")

    LOGGER.info(f"\n{pd.DataFrame(gamma_dict).to_markdown(index=False)}")


def _q14(S0: float, skip_plot: bool = True):
    """Using Heston model data from Question 6, price a European up-and-in call option
    (CUI) with a barrier level of $95 and a strike price of $95 as well. This CUI option
    becomes alive only if the stock price reaches (at some point before maturity) the
    """
    params_to_overide = copy.deepcopy(COMMON_PARAMS)
    params_to_overide['S0'] = S0

    rho = -0.7

    h_params: HestonParameters = HestonParameters(
        v0=0.032,
        kappa=1.85,
        theta=0.045,
        rho=rho,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **params_to_overide
    )

    V, S = get_heston_paths(h_params)
    if not skip_plot:
        plot_heston_paths(n=300, underlying_arr=S, variance_arr=V, h_params=h_params)

    K = barrier = 95.

    price_dict = {'type': [], 'side': [], 'S0': [], 'K': [], 'Barrier': [], 'option_price': []}
    for side in OptionSide:
        for o_type in [OptionType.EUROPEAN, OptionType.UPANDIN]:
            if o_type == OptionType.EUROPEAN:
                o_info = OptionInfo(
                    type=OptionType.EUROPEAN, K=K, side=side,
                )
            elif o_type == OptionType.UPANDIN:
                o_info = BarrierOptionInfo(
                    type=OptionType.UPANDIN, K=K, side=side,
                    barrier=barrier
                )
            else:
                raise ValueError()

            option_price: float = price_montecarlo(
                Spath=S,
                d=h_params,
                o=o_info,
                t=0.
            )
            LOGGER.info(f"{o_type} CALL: {option_price:.3g}")

            price_dict['type'].append(o_type)
            price_dict['side'].append(o_info.side)
            price_dict['S0'].append(h_params.S0)
            price_dict['K'].append(K)
            price_dict['Barrier'].append(barrier)
            price_dict['option_price'].append(f"{option_price:.3g}")

    LOGGER.info(f"\n{pd.DataFrame(price_dict).to_markdown(index=False)}")


def q14(skip_plot=True):
    _q14(S0=80., skip_plot=skip_plot)
    _q14(S0=65., skip_plot=skip_plot)


if __name__ == "__main__":
    # TODO Q13_2 resolve the polynomial fit warning:
    #    numpy/polynomial/polynomial.py:1476: RankWarning: The fit may be poorly conditioned
    #    return pu._fit(polyvander, x, y, deg, rcond, full, w)
    q13_2()
    # q15()