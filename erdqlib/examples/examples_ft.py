from dataclasses import asdict

import pandas as pd

from erdqlib.src.common.option import OptionSide, OptionInfo, OptionType
from erdqlib.src.ft.bates import BatesFtiCalibrator, plot_Bates_full
from erdqlib.src.ft.heston import HestonFtiCalibrator, plot_Heston
from erdqlib.src.mc.bates import BatesDynamicsParameters
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.heston import HestonDynamicsParameters, HestonParameters, Heston
from erdqlib.src.mc.jump import JumpOnlyDynamicsParameters
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
    plot_Heston(opt_params=params_heston, df_options=df_options, S0=S0, side=side)


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
    [22:03:46] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 221.255, 'side': <OptionSide.CALL: 'call'>}
    [22:03:46] {examples_ft.py:75} INFO - asian call K=221.255: 12.17486406680433
    [22:03:46] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 228.242, 'side': <OptionSide.CALL: 'call'>}
    [22:03:46] {examples_ft.py:75} INFO - asian call K=228.242: 6.8842189623751855
    [22:03:46] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 232.9, 'side': <OptionSide.CALL: 'call'>}
    [22:03:46] {examples_ft.py:75} INFO - asian call K=232.9: 4.28329262909716
    [22:03:46] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 237.55800000000002, 'side': <OptionSide.CALL: 'call'>}
    [22:03:46] {examples_ft.py:75} INFO - asian call K=237.55800000000002: 2.45646282146261
    [22:03:46] {evaluate.py:23} INFO - {'type': <OptionType.ASIAN: 'asian'>, 'K': 244.54500000000002, 'side': <OptionSide.CALL: 'call'>}
    [22:03:46] {examples_ft.py:75} INFO - asian call K=244.54500000000002: 0.919981847152032
    '''


def ex_step2_a():
    S0 = 232.9
    r = 1.5 / 100

    df_options: pd.DataFrame = load_option_data(
        path_str=get_path_from_package("erdqlib@examples/data/sm_gwp1_option_data_pivoted.csv"),
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r,  # constant short rate
        strike_tol=None,
        n_bdays_per_year=365.0,
        days_to_maturity_target=60
    )
    LOGGER.info(f"df_options:\n{df_options.to_markdown(index=False)}")

    # a) Heston Calibration on European PUT
    side: OptionSide = OptionSide.PUT
    params_bates: BatesDynamicsParameters = BatesFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        heston_search_grid=HestonDynamicsParameters.get_default_search_grid(),
        jumponly_search_grid=JumpOnlyDynamicsParameters.get_default_search_grid()
    )
    LOGGER.info(f"Bates calib for {side}: {params_bates}")
    '''
    Bates calib for put: BatesDynamicsParameters(S0=232.9, r=None, lambd_merton=np.float64(0.038991695769667906), mu_merton=np.float64(-0.5909283419483802), delta_merton=np.float64(0.0020056143265649896), kappa_heston=0.1620668058996896, theta_heston=0.19549150663111103, sigma_heston=0.2517247864641945, rho_heston=-0.9892603030965936, v0_heston=0.12987597220099956)
    '''
    kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta = params_bates.get_values()
    plot_Bates_full(
        df_options=df_options,
        model_values=BatesFtiCalibrator.calculate_option_price_batch(
            df_options=df_options, S0=S0,
            kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho=rho, v0=v0,
            lambd=lambd, mu=mu, delta=delta,
            side=side
        ),
        side=side
    )


if __name__ == "__main__":
    # ex_step1_a()
    # ex_step1_c()

    ex_step2_a()

    pass