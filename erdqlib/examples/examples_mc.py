import copy
from pprint import pformat
from typing import Any, Dict, List

import pandas as pd

from erdqlib.src.common.option import OptionInfo, OptionSide, OptionType, BarrierOptionInfo
from erdqlib.src.mc.bates import Bates, BatesParameters
from erdqlib.src.mc.bcc import BCC, BCCParameters
from erdqlib.src.mc.cir import CirParameters, Cir
from erdqlib.src.mc.evaluate import price_montecarlo, calculate_delta, DeltaResult
from erdqlib.src.mc.gbm import Gbm, GbmParameters
from erdqlib.src.mc.heston import HestonParameters, Heston
from erdqlib.src.mc.jump import JumpParameters, MertonJump
from erdqlib.src.mc.vasicek import VasicekParameters, Vasicek
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
        lambd_merton=0.75,
        mu_merton=-0.5,
        delta_merton=0.22,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    S = MertonJump.calculate_paths(j_params)
    if not skip_plot:
        MertonJump.plot_paths(n=300, paths={'x': S}, model_params=j_params, model_name=MertonJump.__name__, logy=True)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        model_params=j_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t_i=0.
    ):.3g}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        model_params=j_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t_i=0.
    ):.3g}")


def q9(skip_plot: bool = True):
    """8. Using the Merton Model,
    price an ATM European call and an ATM European put with jump intensity parameter equal to 0.25.
    """
    j_params: JumpParameters = JumpParameters(
        lambd_merton=0.25,
        mu_merton=-0.5,
        delta_merton=0.22,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    S = MertonJump.calculate_paths(j_params)
    if not skip_plot:
        MertonJump.plot_paths(n=300, paths={'x': S}, model_params=j_params, model_name=MertonJump.__name__, logy=True)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        model_params=j_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t_i=0.
    ):.3g}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        model_params=j_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t_i=0.
    ):.3g}")


def replace_params_map(original_params: Dict[str, Any], s0: float) -> Dict[str, Any]:
    dynamics_params: Dict[str, Any] = copy.deepcopy(original_params)
    dynamics_params['S0'] = s0
    return dynamics_params


def _delta_per_S0(
        Sinit: float, K: float, ud_ratio: float, lambd_list: List[float], option_type: OptionType = OptionType.EUROPEAN
) -> Dict[OptionSide, Dict[float, DeltaResult]]:
    pricing_dict = {}
    for side in OptionSide:
        per_lambda_dict = {}
        for lambd in lambd_list:
            # Vary S0 by 1% on both up/down side - note it's not 1bps but 1% for numerical stability
            delta_per_lambda: DeltaResult = calculate_delta(
                s0=Sinit, strike=K, option_type=option_type, side=side,
                bump_ratio=ud_ratio,
                model=MertonJump(),
                model_params_creator= lambda s0: JumpParameters(
                    lambd_merton=lambd,
                    mu_merton=-0.5,
                    delta_merton=0.22,

                    M=500,  # Total time steps
                    I=10000,  # Number of simulations
                    random_seed=0,
                    **replace_params_map(COMMON_PARAMS, s0)
                )
            )
            per_lambda_dict[lambd] = delta_per_lambda

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

    pricing_dict_delta: Dict[OptionSide, Dict[float, DeltaResult]] = _delta_per_S0(
        Sinit=K, K=K, ud_ratio=ud_ratio, lambd_list=lambd_list, option_type=option_type
    )
    delta_dict = {'side': [], 'lambda': [], 'delta': []}
    for side in OptionSide:
        for lambd in lambd_list:
            delta_result: DeltaResult = pricing_dict_delta[side][lambd]
            px_down: float = delta_result.o_down
            px_up: float = delta_result.o_up

            delta: float = (px_up - px_down) / (S0_list[1] - S0_list[0])
            delta_dict['side'].append(side)
            delta_dict['lambda'].append(lambd)
            delta_dict['delta'].append(f"{delta:.2g}")

    LOGGER.info(f"\n{pd.DataFrame(delta_dict).to_markdown(index=False)}")

    pricing_dict_down: Dict[OptionSide, Dict[float, DeltaResult]] = _delta_per_S0(
        Sinit=S0_list[0], K=K, ud_ratio=ud_ratio, lambd_list=lambd_list, option_type=option_type
    )

    pricing_dict_up: Dict[OptionSide, Dict[float, DeltaResult]] = _delta_per_S0(
        Sinit=S0_list[1], K=K, ud_ratio=ud_ratio, lambd_list=lambd_list, option_type=option_type
    )

    gamma_dict = {'side': [], 'lambda': [], 'gamma': []}
    for side in OptionSide:
        for lambd in lambd_list:
            delta_down: float = pricing_dict_down[side][lambd].delta
            delta_up: float = pricing_dict_up[side][lambd].delta

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
        lambd_merton=0.75,
        mu_merton=-0.5,
        delta_merton=0.22,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **params_to_overide
    )

    S = MertonJump.calculate_paths(j_params)
    if not skip_plot:
        MertonJump.plot_paths(n=300, paths={'x': S}, model_params=j_params, model_name=MertonJump.__name__, logy=True)

    K = barrier = 65.

    price_dict = {'type': [], 'side': [], 'S0': [], 'K': [], 'Barrier': [], 'option_price': []}
    for o_type in [OptionType.EUROPEAN, OptionType.DOWNANDIN]:
        if o_type == OptionType.EUROPEAN:
            o_info = OptionInfo(
                o_type=OptionType.EUROPEAN, K=K, side=side,
            )
        elif o_type == OptionType.DOWNANDIN:
            o_info = BarrierOptionInfo(
                o_type=OptionType.DOWNANDIN, K=K, side=side,
                barrier=barrier
            )
        else:
            raise ValueError()

        option_price: float = price_montecarlo(
            underlying_path=S,
            model_params=j_params,
            o_info=o_info,
            t_i=0.
        )
        LOGGER.info(f"{o_type} {side}: {option_price:.3g}")

        price_dict['type'].append(o_type)
        price_dict['side'].append(o_info.side)
        price_dict['S0'].append(j_params.x0)
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
        v0_heston=0.032,
        kappa_heston=1.85,
        theta_heston=0.045,
        rho_heston=rho,
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    V, S = Heston.calculate_paths(h_params)
    if not skip_plot:
        Heston.plot_paths(n=300, paths={'x':S, 'var':V}, model_params=h_params, model_name=Heston.__name__)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        model_params=h_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t_i=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        model_params=h_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t_i=0.
    )}")


def q6(skip_plot: bool = True):
    """5. Using the Heston Model and Monte-Carlo simulation, price an ATM European call
    and an ATM European put, using a correlation value of -0.30.
    """
    rho = -0.7

    h_params: HestonParameters = HestonParameters(
        v0_heston=0.032,
        kappa_heston=1.85,
        theta_heston=0.045,
        rho_heston=rho,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    V, S = Heston.calculate_paths(h_params)
    if not skip_plot:
        Heston.plot_paths(n=300, paths={'x': S, 'var': V}, model_params=h_params, model_name=Heston.__name__)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        model_params=h_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t_i=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        model_params=h_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t_i=0.
    )}")


def _delta_per_S0_q7(
    s_init: float, strike: float, ud_ratio: float, rho_list: List[float], option_type: OptionType = OptionType.EUROPEAN
) -> Dict[OptionSide, Dict[float, DeltaResult]]:
    pricing_dict = {}
    for side in OptionSide:
        per_rho_dict = {}
        for rho in rho_list:
            # Vary S0 by 1% on both up/down side - note it's not 1bps but 1% for numerical stability
            delta_per_rho: DeltaResult = calculate_delta(
                s0=s_init, strike=strike, option_type=option_type, side=side,
                bump_ratio=ud_ratio,
                model=Heston(),
                model_params_creator=lambda s0: HestonParameters(
                    v0_heston=0.032,
                    kappa_heston=1.85,
                    theta_heston=0.045,
                    rho_heston=rho,

                    M=500,  # Total time steps
                    I=10000,  # Number of simulations
                    random_seed=0,
                    **replace_params_map(COMMON_PARAMS, s0)
                )
            )
            per_rho_dict[rho] = delta_per_rho
        pricing_dict[side] = per_rho_dict

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

    pricing_dict_delta: Dict[OptionSide, Dict[float, DeltaResult]] = _delta_per_S0_q7(
        s_init=S0, strike=K, ud_ratio=ud_ratio, rho_list=rho_list, option_type=option_type
    )
    delta_dict = {'side': [], 'rho': [], 'delta': []}
    for side in OptionSide:
        for rho in rho_list:
            px_down: float = pricing_dict_delta[side][rho].s0_down
            px_up: float = pricing_dict_delta[side][rho].s0_up

            delta: float = (px_up - px_down) / (S0_list[1] - S0_list[0])
            delta_dict['side'].append(side)
            delta_dict['rho'].append(rho)
            delta_dict['delta'].append(f"{delta:.2g}")

    LOGGER.info(f"\n{pd.DataFrame(delta_dict).to_markdown(index=False)}")

    pricing_dict_down: Dict[OptionSide, Dict[float, DeltaResult]] = _delta_per_S0_q7(
        s_init=S0_list[0], strike=K, ud_ratio=ud_ratio, rho_list=rho_list,
                                         option_type=option_type)

    pricing_dict_up: Dict[OptionSide, Dict[float, DeltaResult]] = _delta_per_S0_q7(
        s_init=S0_list[1], strike=K, ud_ratio=ud_ratio, rho_list=rho_list,
                                       option_type=option_type)

    gamma_dict = {'side': [], 'rho': [], 'gamma': []}
    for side in OptionSide:
        for rho in rho_list:
            delta_down: float = pricing_dict_down[side][rho].delta
            delta_up: float = pricing_dict_up[side][rho].delta

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
        v0_heston=0.032,
        kappa_heston=1.85,
        theta_heston=0.045,
        rho_heston=rho,
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **COMMON_PARAMS
    )

    V, S = Heston.calculate_paths(h_params)
    if not skip_plot:
        Heston.plot_paths(n=300, paths={'x': S, 'var': V}, model_params=h_params, model_name=Heston.__name__)

    LOGGER.info(f"AMR CALL: {price_montecarlo(
        underlying_path=S,
        model_params=h_params,
        o_info=OptionInfo(
            o_type=OptionType.AMERICAN, K=COMMON_PARAMS['S0'], side=OptionSide.CALL
        ),
        t_i=0.,
        verbose=True
    )}")

    LOGGER.info(f"AMR PUT: {price_montecarlo(
        underlying_path=S,
        model_params=h_params,
        o_info=OptionInfo(
            o_type=OptionType.AMERICAN, K=COMMON_PARAMS['S0'], side=OptionSide.PUT
        ),
        t_i=0.
    )}")


def q13_2():
    """Americal option call for heston - Delta and Gamma"""
    option_type: OptionType = OptionType.EUROPEAN
    K = 80.
    rho_list = [-0.3, -0.7]

    ud_ratio: float = 0.01
    S0 = K * 0.9
    S0_list = [S0 * (1. - ud_ratio), S0 * (1. + ud_ratio)]

    pricing_dict_delta: Dict[OptionSide, Dict[float, DeltaResult]] = _delta_per_S0_q7(
        s_init=S0, strike=K, ud_ratio=ud_ratio, rho_list=rho_list, option_type=OptionType.AMERICAN
    )

    delta_dict = {'side': [], 'rho': [], 'delta': []}
    for side in OptionSide:
        for rho in rho_list:
            px_down: float = pricing_dict_delta[side][rho].s0_down
            px_up: float = pricing_dict_delta[side][rho].s0_up

            delta: float = (px_up - px_down) / (S0_list[1] - S0_list[0])
            delta_dict['side'].append(side)
            delta_dict['rho'].append(rho)
            delta_dict['delta'].append(f"{delta:.2g}")

    LOGGER.info(f"\n{pd.DataFrame(delta_dict).to_markdown(index=False)}")

    pricing_dict_down: Dict[OptionSide, Dict[float, DeltaResult]] = _delta_per_S0_q7(
        s_init=S0_list[0], strike=K, ud_ratio=ud_ratio, rho_list=rho_list,
                                         option_type=option_type)

    pricing_dict_up: Dict[OptionSide, Dict[float, DeltaResult]] = _delta_per_S0_q7(
        s_init=S0_list[1], strike=K, ud_ratio=ud_ratio, rho_list=rho_list,
                                       option_type=option_type)

    gamma_dict = {'side': [], 'rho': [], 'gamma': []}
    for side in OptionSide:
        for rho in rho_list:
            delta_down: float = pricing_dict_down[side][rho].delta
            delta_up: float = pricing_dict_up[side][rho].delta

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
        v0_heston=0.032,
        kappa_heston=1.85,
        theta_heston=0.045,
        rho_heston=rho,

        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0,
        **params_to_overide
    )

    V, S = Heston.calculate_paths(h_params)
    if not skip_plot:
        Heston.plot_paths(n=300, paths={'x': S, 'var': V}, model_params=h_params, model_name=Heston.__name__)

    K = barrier = 95.

    price_dict = {'type': [], 'side': [], 'S0': [], 'K': [], 'Barrier': [], 'option_price': []}
    for side in OptionSide:
        for o_type in [OptionType.EUROPEAN, OptionType.UPANDIN]:
            if o_type == OptionType.EUROPEAN:
                o_info = OptionInfo(
                    o_type=OptionType.EUROPEAN, K=K, side=side,
                )
            elif o_type == OptionType.UPANDIN:
                o_info = BarrierOptionInfo(
                    o_type=OptionType.UPANDIN, K=K, side=side,
                    barrier=barrier
                )
            else:
                raise ValueError()

            option_price: float = price_montecarlo(
                underlying_path=S,
                model_params=h_params,
                o_info=o_info,
                t_i=0.
            )
            LOGGER.info(f"{o_type} CALL: {option_price:.3g}")

            price_dict['type'].append(o_type)
            price_dict['side'].append(o_info.side)
            price_dict['S0'].append(h_params.x0)
            price_dict['K'].append(K)
            price_dict['Barrier'].append(barrier)
            price_dict['option_price'].append(f"{option_price:.3g}")

    LOGGER.info(f"\n{pd.DataFrame(price_dict).to_markdown(index=False)}")


def q14(skip_plot=True):
    _q14(S0=80., skip_plot=skip_plot)
    _q14(S0=65., skip_plot=skip_plot)


def example_gbm():
    g_params: GbmParameters = GbmParameters(
        sigma=0.2,

        x0=100,  # Current underlying asset price
        r=0.05,  # Risk-free rate

        T=1,  # Number of years
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0
    )

    S = Gbm.calculate_paths(g_params)
    Gbm.plot_paths(n=300, paths={'x': S}, model_params=g_params, model_name=Gbm.__name__, logy=True)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        model_params=g_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t_i=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        model_params=g_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t_i=0.
    )}")


def example_heston():
    h_params: HestonParameters = HestonParameters(
        v0_heston=0.04,
        kappa_heston=2,
        sigma_heston=0.3,
        theta_heston=0.04,
        rho_heston=-0.9,

        x0=100,  # Current underlying asset price
        r=0.05,  # Risk-free rate

        T=1,  # Number of years
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0
    )

    V, S = Heston.calculate_paths(h_params)
    Heston.plot_paths(
        n=300, paths={'x': S, 'var': V}, model_params=h_params, model_name=Heston.__name__
    )

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        model_params=h_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t_i=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        model_params=h_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t_i=0.
    )}")


def example_jump():
    j_params: JumpParameters = JumpParameters(
        lambd_merton=0.75,  # Lambda of the model
        mu_merton=0.0,  # Mu
        delta_merton=0.25,  # Delta
        sigma_merton=0.2,

        x0=100,  # Current underlying asset price
        r=0.05,  # Risk-free rate

        T=1,  # Number of years
        M=500,  # Total time steps
        I=10_000,  # Number of simulations
        random_seed=0
    )

    S = MertonJump.calculate_paths(j_params)
    MertonJump.plot_paths(n=300, paths={'x': S}, model_params=j_params, model_name=MertonJump.__name__, logy=True)

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        model_params=j_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t_i=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        model_params=j_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t_i=0.
    )}")


def example_bates():
    b_params: BatesParameters = BatesParameters(
        v0_heston=0.04,
        kappa_heston=2,
        sigma_heston=0.3,
        theta_heston=0.04,
        rho_heston=-0.9,

        lambd_merton=0.75,
        mu_merton=0.0,
        delta_merton=0.25,

        x0=100,  # Current underlying asset price
        r=0.05,  # Risk-free rate

        T=1,  # Number of years
        M=500,  # Total time steps
        I=10000,  # Number of simulations
        random_seed=0
    )

    V, S = Bates.calculate_paths(b_params)
    Bates.plot_paths(
        n=300, paths={'x': S, 'var': V}, model_params=b_params, model_name=Bates.__name__
    )

    LOGGER.info(f"EUR CALL: {price_montecarlo(
        underlying_path=S,
        model_params=b_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t_i=0.
    )}")

    LOGGER.info(f"EUR PUT: {price_montecarlo(
        underlying_path=S,
        model_params=b_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t_i=0.
    )}")


def example_bcc():
    def example_bcc():
        # r for BCC works as the r0 for short-rate dynamics
        bcc_params: BCCParameters = BCCParameters(
            T=1.0,
            M=250,  # type: ignore
            I=100_000,  # type: ignore
            random_seed=0,  # type: ignore
            **{
                "x0": 3225.93,
                "r": -0.0002943493765991875,
                "lambd_merton": 8.463250888577545e-07,
                "mu_merton": -9.449456757062437e-06,
                "delta_merton": 9.821793189744765e-07,
                "kappa_heston": 2.412006106350429,
                "theta_heston": 0.022478722375727497,
                "sigma_heston": 0.32821891480428733,
                "rho_heston": -0.6712549317699577,
                "v0_heston": 0.030392880298239243,
                "kappa_cir": 0.603708218317528,
                "theta_cir": 0.03120714567592455,
                "sigma_cir": 0.19411341500935292
            }
        )

        r_arr, v_arr, x_arr = BCC.calculate_paths(model_params=bcc_params)
        BCC.plot_paths(
            n=500,
            paths={'x': x_arr, 'var': v_arr, 'r': r_arr},
            model_params=bcc_params,
            model_name=BCC.__name__
        )

        o_price: float = price_montecarlo(
            underlying_path=x_arr,
            model_params=bcc_params,
            o_info=OptionInfo(
                o_type=OptionType.ASIAN,
                K=bcc_params.x0 * 0.95,  # strike price
                side=OptionSide.PUT,
            )
        )
        LOGGER.info(f"Asian PUT option price: {o_price}")


def example_vasicek():
    v_params: VasicekParameters = VasicekParameters(
        T=1.0,  # Maturity
        M=500,  # Number of paths for MC
        I=10_000,  # Number of steps
        random_seed=0,

        x0=0.023,
        kappa_vasicek=0.20,
        theta_vasicek=0.01,
        sigma_vasicek=0.0012,
        r=None,  # Risk-free rate
    )

    rates = Vasicek.calculate_paths(v_params)
    Vasicek.plot_paths(n=300, paths={'x': rates}, model_params=v_params, model_name=Vasicek.__name__, logy=False)


def example_cir():
    v_params: CirParameters = CirParameters(
        T = 1.0,  # Maturity
        M = 500,  # Number of paths for MC
        I = 10_000,  # Number of steps
        random_seed=0,

        x0= 0.023,
        kappa_cir= 0.20,
        theta_cir = 0.01,
        sigma_cir = 0.012,
        r = None,  # Risk-free rate
    )

    rates = Cir.calculate_paths(v_params)
    Cir.plot_paths(
        n=300,
        paths={'x': rates},
        model_params=v_params,
        model_name=Cir.__name__,
        logy=False,
        ylabel="Rate"
    )


if __name__ == "__main__":
    # # Call all the questions functions below
    # q5(skip_plot=True)
    # q6(skip_plot=True)
    # q7()
    # q8(skip_plot=True)
    # q9(skip_plot=True)
    # q10()
    # q13_1(skip_plot=True)
    # q13_2()
    # q15(skip_plot=True)
    # q14(skip_plot=True)
    example_cir()