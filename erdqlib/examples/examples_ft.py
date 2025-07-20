from dataclasses import asdict

import pandas as pd

from erdqlib.src.common.option import OptionSide, OptionInfo, OptionType
from erdqlib.src.ft.heston import HestonFtiCalibrator, plot_Heston
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.heston import HestonDynamicsParameters, HestonParameters, Heston
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)


def ex_step1_a():
    S0 = 232.9
    r = 1.5 / 100

    df_options: pd.DataFrame = load_option_data(
        path_str=get_path_from_package("erdqlib@examples/data/sm_gwp1_option_data_pivoted.csv"),
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r,  # constant short rate
        strike_tol=None,
        n_bdays_per_year=365.0,
        days_to_maturity_target=15
    )
    LOGGER.info(f"df_options:\n{df_options.to_markdown(index=False)}")

    # a) Heston Calibration on European PUT
    side: OptionSide = OptionSide.PUT
    params_heston: HestonDynamicsParameters = HestonFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        search_grid=HestonDynamicsParameters.get_default_search_grid()
    )
    LOGGER.info(f"Heston calib for {side}: {params_heston}")
    '''
    Heston calib for put: HestonDynamicsParameters(S0=232.9, r=0.015, kappa_heston=1.1195502670984554, theta_heston=0.046401794157976295, sigma_heston=0.32233256442193337, rho_heston=-0.9898495790181241, v0_heston=0.1243614465557908)
    '''
    plot_Heston(params_heston, df_options, S0, side)


def ex_step1_c():
    S0 = 232.9
    r = 1.5 / 100

    # Calibrated parameters for Heston model from ex_step1_a
    h_params_dynamics: HestonDynamicsParameters = HestonDynamicsParameters(
        S0=S0,
        r=r,
        kappa_heston=1.1195502670984554,
        theta_heston=0.046401794157976295,
        sigma_heston=0.32233256442193337,
        rho_heston=-0.9898495790181241,
        v0_heston=0.1243614465557908
    )

    h_params: HestonParameters = HestonParameters(
        T=20/365.0,
        M=20,
        I=10_000,
        random_seed=0,
        **asdict(h_params_dynamics)
    )

    var_paths, s_paths = Heston.calculate_paths(h_params)
    Heston.plot_paths(
        n=1_000, paths={'x': s_paths, 'var': var_paths}, model_params=h_params, model_name=Heston.__name__
    )

    option_type: OptionType = OptionType.ASIAN
    option_side: OptionSide = OptionSide.CALL
    for moneyness in [0.95, 0.98, 1.0, 1.02, 1.05]:
        K: float = S0 * moneyness
        LOGGER.info(f"{option_type} {option_side} K={K}: {price_montecarlo(
            underlying_path=s_paths,
            d=h_params,
            o=OptionInfo(
                type=option_type, K=K, side=option_side
            ),
            t=0.
        )}")

    '''
    [00:51:07] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 221.255, 'side': <OptionSide.CALL: 'call'>}
    [00:51:07] {examples_ft.py:75} INFO - asian call K=221.255: 12.17486406680433
    [00:51:07] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 228.242, 'side': <OptionSide.CALL: 'call'>}
    [00:51:07] {examples_ft.py:75} INFO - asian call K=228.242: 6.8842189623751855
    [00:51:07] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 232.9, 'side': <OptionSide.CALL: 'call'>}
    [00:51:07] {examples_ft.py:75} INFO - asian call K=232.9: 4.28329262909716
    [00:51:07] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 237.55800000000002, 'side': <OptionSide.CALL: 'call'>}
    [00:51:07] {examples_ft.py:75} INFO - asian call K=237.55800000000002: 2.45646282146261
    [00:51:07] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 244.54500000000002, 'side': <OptionSide.CALL: 'call'>}
    [00:51:07] {examples_ft.py:75} INFO - asian call K=244.54500000000002: 0.919981847152032
    '''


if __name__ == "__main__":
    # ex_step1_a()
    ex_step1_c()