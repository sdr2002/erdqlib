import numpy as np
import pandas as pd

from erdqlib.src.common.option import OptionType, OptionSide, OptionDataColumn
from erdqlib.src.common.rate import SplineCurve, ForwardsLadder
from erdqlib.src.ft.bcc import BccFtiCalibrator, plot_BCC_full
from erdqlib.src.ft.calibrator import FtMethod, FtiCalibrator
from erdqlib.src.ft.cir import CirCalibrator, plot_calibrated_cir
from erdqlib.src.mc.bates import BatesDynamicsParameters, BatesParameters, Bates
from erdqlib.src.mc.bcc import BCCParameters, BCC, BccDynamicsParameters
from erdqlib.src.mc.cir import CirDynamicsParameters, CirParameters, Cir
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)

from erdqlib.src.common.option import OptionInfo
from erdqlib.src.mc.evaluate import price_montecarlo

S0: float = 232.9
n_days_per_annum: int =250

path_options_data: str = get_path_from_package("erdqlib@examples/data/sm_gwp1_option_data_pivoted.csv")
path_rates_data: str = get_path_from_package("erdqlib@examples/data/sm_gwp1_euribor.csv")


def ex_step3_a(skip_plot: bool = False):
    # Euribor Market data
    euribor_df: pd.DataFrame = pd.read_csv(path_rates_data)
    LOGGER.info(f"Rates:\n{euribor_df.to_markdown(index=False)}")

    maturities: np.ndarray = euribor_df[OptionDataColumn.TENOR].values  # Maturities in years with 30/360 convention
    euribors: np.ndarray = euribor_df[OptionDataColumn.RATE].values  # Euribor rates in rate unit

    # Interpolation and Forward rates via Cubic spline
    scurve: SplineCurve = SplineCurve()
    scurve.update_curve(maturities=maturities, spot_rates=euribors)
    forward_rates: ForwardsLadder = scurve.calculate_forward_rates(t_f=1.0)

    params_cir: CirDynamicsParameters = CirCalibrator.calibrate(
        r0=scurve.get_r0(),
        curve_forward_rates=forward_rates.rates,
        maturities_ladder=forward_rates.maturities
    )  # [0.06190266 0.23359687 0.14892987]
    LOGGER.info(f"CIR params:{params_cir}")

    if not skip_plot:
        scurve.plot_curve()
        plot_calibrated_cir(
            model_params=params_cir.get_value_arr(),
            maturities_ladder=forward_rates.maturities,
            market_forward_rates=forward_rates.rates,
            r0=scurve.get_r0()
        )


def ex_step3_b():
    v_params: CirParameters = CirParameters(
        T = 1.0,  # Maturity
        M = int(n_days_per_annum),  # Number of paths for MC
        I = 10_000,  # Number of steps
        random_seed=0,
        **{
            "x0": 0.00648,
            "r": None,
            "kappa_cir": 0.6989623744196691,
            "theta_cir": 0.10868234595604083,
            "sigma_cir": 0.0010018697187414139
        }
    )

    rates = Cir.calculate_paths(v_params)
    Cir.plot_paths(
        n=int(n_days_per_annum),
        paths={'x': rates},
        model_params=v_params,
        model_name=Cir.__name__,
        logy=False,
        ylabel="Rate"
    )


def ex_step3_b_bcc_calibration(skip_plot: bool = False):
    ft_method: FtMethod = FtMethod.LEWIS
    side: OptionSide = OptionSide.PUT
    target_dtm: int = 120

    ### Load data on September 30, 2014 with r from the CIR model instead of constant overnight rate 0.005
    # Euribor Market data
    df_euribor: pd.DataFrame = pd.read_csv(path_rates_data)
    # EURO STOXX 50 level
    df_options: pd.DataFrame = load_option_data(
        path_str=path_options_data, S0=S0,
        r_provider=lambda *_, **__: np.nan,
        days_to_maturity_target=target_dtm
    )
    LOGGER.info(f"Loaded df_options:\n{df_options.to_markdown()}")

    params_bcc: BccDynamicsParameters = BccFtiCalibrator.calibrate(
        df_rates=df_euribor, df_options=df_options, S0=S0, side=side,
        heston_search_grid=BatesDynamicsParameters.get_default_heston_search_grid(),
        jumponly_search_grid=BatesDynamicsParameters.get_default_jumponly_search_grid(),
        ft_method=ft_method, skip_plot=skip_plot
    )
    LOGGER.info(f"BCC calib: {params_bcc}")
    LOGGER.info(f"\n{params_bcc.to_json()}")

    if not skip_plot:
        plot_BCC_full(
            bcc_params=params_bcc,
            df_options=df_options,
            S0=S0,
            side=side
        )


def ex_step3_b_bcc_pricing():
    # r for BCC works as the r0 for short-rate dynamics
    bcc_params: BCCParameters = BCCParameters(
        T=1.0,
        M=250,  # type: ignore
        I=100_000,  # type: ignore
        random_seed=0,  # type: ignore
        **{
            "x0": 232.9,
            "r": 0.0066152822490763005,
            "lambd_merton": 1.2278781717417533,
            "mu_merton": -0.1239414769874404,
            "delta_merton": 0.36576662517593594,
            "kappa_heston": 0.0005976203046843384,
            "theta_heston": 0.15302601447995312,
            "sigma_heston": 0.013524160114265105,
            "rho_heston": -0.012751011356139805,
            "v0_heston": 0.025250476477415518,
            "kappa_cir": 0.6975357466282398,
            "theta_cir": 0.10692607624899234,
            "sigma_cir": 0.0010021514062057968
        }
    )

    r_arr, v_arr, x_arr = BCC.calculate_paths(model_params=bcc_params)
    BCC.plot_paths(
        n=500,
        paths={'x': x_arr, 'var': v_arr, 'r': r_arr},
        model_params=bcc_params,
        model_name=BCC.__name__
    )

    o_side = OptionSide.PUT
    for o_type in [OptionType.EUROPEAN, OptionType.ASIAN]:
        o_price: float = price_montecarlo(
            underlying_path=x_arr,
            model_params=bcc_params,
            o_info=OptionInfo(
                o_type=o_type,
                K=bcc_params.x0 * 0.95,  # strike price
                side=o_side,
            )
        )
        LOGGER.info(f"{o_type} {o_side} option price: {o_price}")


def ex_step3_b_bates_pricing():
    # r for BCC works as the r0 for short-rate dynamics
    bates_params: BatesParameters = BatesParameters(
        T=1.0,
        M=250,  # type: ignore
        I=100_000,  # type: ignore
        random_seed=0,  # type: ignore
        **{
            "x0": 232.9,
            "r": 0.0066152822490763005,
            "lambd_merton": 1.6519104881865774,
            "mu_merton": -0.11215354241730985,
            "delta_merton": 0.3965512332994626,
            "kappa_heston": 2.846435947295412e-08,
            "theta_heston": 0.17610219057306495,
            "sigma_heston": 0.00010012628083022352,
            "rho_heston": 0.7055603835367572,
            "v0_heston": 0.024265907755035142
        }
    )

    v_arr, x_arr = Bates.calculate_paths(model_params=bates_params)
    Bates.plot_paths(
        n=500,
        paths={'x': x_arr, 'var': v_arr},
        model_params=bates_params,
        model_name=Bates.__name__
    )

    o_side = OptionSide.PUT
    for o_type in [OptionType.EUROPEAN, OptionType.ASIAN]:
        o_price: float = price_montecarlo(
            underlying_path=x_arr,
            model_params=bates_params,
            o_info=OptionInfo(
                o_type=o_type,
                K=bates_params.x0 * 0.95,  # strike price
                side=o_side,
            )
        )
        LOGGER.info(f"{o_type} {o_side} option price: {o_price}")


def ex_compare_fti_versus_mc_bcc_european_option():
    bcc_params = BCCParameters(
        x0=100.,
        r=-0.032 / 100,
        T=1.,

        # We price with FTI for this example, hence the MC configs are not needed
        M=250,  # type: ignore
        I=100_000,  # type: ignore
        random_seed=3,  # type: ignore

        kappa_cir=0.068,
        theta_cir=0.207,
        sigma_cir=0.112,

        kappa_heston=0.068,
        theta_heston=0.207,
        sigma_heston=0.112,
        rho_heston=-0.821,
        v0_heston=0.035,

        lambd_merton=0.008,
        mu_merton=-0.600,
        delta_merton=0.001,
    )

    K = 90.
    for side in OptionSide:
        x0, T, r, kappa_heston, theta_heston, sigma_heston, rho_heston, v0_heston, lambd_merton, mu_merton, delta_merton = bcc_params.get_pricing_params(
            apply_shortrate=True
        )
        bcc_price_fti: float = FtiCalibrator.calculate_option_price_lewis(
            x0=x0, T=T, r=r, K=K,
            characteristic_integral=lambda u: BccFtiCalibrator.calculate_integral_characteristic(
                u=u, S0=x0, K=K, T=T, r=r,
                kappa_heston=kappa_heston, theta_heston=theta_heston, sigma_heston=sigma_heston, rho_heston=rho_heston, v0_heston=v0_heston,
                lambd_merton=lambd_merton, mu_merton=mu_merton, delta_merton=delta_merton,
            ),
            side=side
        )
        LOGGER.info(f"{side} EUR Option value under BCC FTI-Lewis: {bcc_price_fti}")

        bcc_price_fti: float = BccFtiCalibrator.calculate_option_price_carrmadan(
            x0=x0, T=T, r=r, K=90.,
            kappa_heston=kappa_heston, theta_heston=theta_heston, sigma_heston=sigma_heston, rho_heston=rho_heston, v0_heston=v0_heston,
            lambd_merton=lambd_merton, mu_merton=mu_merton, delta_merton=delta_merton,
            side=side
        )
        LOGGER.info(f"{side} EUR Option value under BCC FTI-CarrMadan: {bcc_price_fti}")

        r_arr, v_arr, x_arr = BCC.calculate_paths(model_params=bcc_params)
        bcc_price_mc: float = price_montecarlo(
            underlying_path=x_arr,
            model_params=bcc_params,
            o_info=OptionInfo(
                o_type=OptionType.EUROPEAN,
                K=90.,  # strike price
                side=side,
            )
        )
        LOGGER.info(f"{side} EUR Option value under BCC MC: {bcc_price_mc}")


if __name__ == "__main__":
    # ex_step3_a()
    # ex_step3_b()
    # ex_step3_b_bcc_calibration()
    # ex_step3_b_bcc_pricing()
    # ex_step3_b_bates_pricing()
    ex_compare_fti_versus_mc_bcc_european_option()