from typing import Tuple, Type

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fmin

from erdqlib.src.common.option import OptionSide, OptionDataColumn
from erdqlib.src.common.rate import SplineCurve, ForwardsLadder, implied_yield
from erdqlib.src.ft.bates import BatesFtiCalibrator
from erdqlib.src.ft.calibrator import FtMethod, plot_calibration_result, FtiCalibrator
from erdqlib.src.ft.cir import CirCalibrator, plot_calibrated_cir
from erdqlib.src.ft.heston import HestonFtiCalibrator
from erdqlib.src.mc.bates import BatesDynamicsParameters
from erdqlib.src.mc.bcc import BCCParameters, BccDynamicsParameters, B
from erdqlib.src.mc.cir import CirDynamicsParameters
from erdqlib.src.mc.heston import HestonDynamicsParameters, HestonSearchGridType
from erdqlib.src.mc.jump import JumpOnlyDynamicsParameters, JumpOnlySearchGridType
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)

BccSearchGridType: Type = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]


class BccFtiCalibrator(BatesFtiCalibrator):
    @staticmethod
    def calibrate_full(
            df_options: pd.DataFrame,
            S0: float,
            r0: float,
            initial_params: np.ndarray,
            cir_params: CirDynamicsParameters,
            side: OptionSide,
            ft_method: FtMethod = FtMethod.LEWIS
    ) -> BccDynamicsParameters:
        """Full calibration of BCC (1997)"""
        print_iter = [0]
        min_MSE = [5000.0]

        LOGGER.info("fmin optimization begins")
        opt: np.ndarray = fmin(
            lambda p: BccFtiCalibrator.calculate_error_full(
                p0=p, options=df_options, s0=S0,
                print_iter=print_iter, min_MSE=min_MSE, side=side, ft_method=ft_method
            ),
            initial_params,
            xtol=0.000001, ftol=0.000001, maxiter=450, maxfun=750
        )
        LOGGER.info(
            f"Full optimisation result: {print_iter[0]} | [{', '.join(f'{x:.2f}' for x in opt)}] | {min_MSE[0]:7.3f}")
        return BccDynamicsParameters(
            x0=S0,

            r=r0,
            kappa_cir=cir_params.kappa_cir,
            theta_cir=cir_params.theta_cir,
            sigma_cir=cir_params.sigma_cir,

            kappa_heston=float(opt[0]),
            theta_heston=float(opt[1]),
            sigma_heston=float(opt[2]),
            rho_heston=float(opt[3]),
            v0_heston=float(opt[4]),

            lambd_merton=float(opt[5]),
            mu_merton=float(opt[6]),
            delta_merton=float(opt[7]),
        )

    @staticmethod
    def calibrate(
            df_rates: pd.DataFrame, df_options: pd.DataFrame,
            S0: float, side: OptionSide,
            heston_search_grid: HestonSearchGridType = HestonDynamicsParameters.get_default_search_grid(),
            jumponly_search_grid: JumpOnlySearchGridType = JumpOnlyDynamicsParameters.get_default_search_grid(),
            ft_method: FtMethod = FtMethod.LEWIS,
            skip_plot: bool = True
    ) -> BccDynamicsParameters:
        # Interpolation and Forward rates via Cubic spline
        scurve: SplineCurve = SplineCurve()
        scurve.update_curve(
            maturities=df_rates[OptionDataColumn.TENOR].values,
            spot_rates=df_rates[OptionDataColumn.RATE].values
        )
        forward_rates: ForwardsLadder = scurve.calculate_forward_rates(
            t_f=float(df_rates[OptionDataColumn.TENOR].iloc[-1])
        )
        r0: float = scurve.get_r0(source="overnight")

        # Calibration of CIR parameters
        params_cir: CirDynamicsParameters = CirCalibrator.calibrate(
            r0=r0,
            curve_forward_rates=forward_rates.rates,
            maturities_ladder=forward_rates.maturities
        )
        LOGGER.info(f"CIR calib: {params_cir}")
        df_options[OptionDataColumn.RATE]=df_options.apply(
            axis=1,
            func=lambda row: implied_yield(
                t_year=row[OptionDataColumn.TENOR],
                price_0_t=B([
                    r0,
                    params_cir.kappa_cir, params_cir.theta_cir, params_cir.sigma_cir,
                    0, row[OptionDataColumn.TENOR]]
                ),
                price_t_t=1.0
            )
        )
        if not skip_plot:
            scurve.plot_curve()
            plot_calibrated_cir(
                model_params=params_cir.get_value_arr(),
                maturities_ladder=forward_rates.maturities,
                market_forward_rates=forward_rates.rates,
                r0=r0
            )

        ### Short calibration
        # Heston component calib
        LOGGER.info("\n--- Short (Heston) calibration ---")
        h_params: HestonDynamicsParameters = HestonFtiCalibrator.calibrate(
            df_options=df_options, S0=S0, r=None, side=side,
            search_grid=heston_search_grid
        )
        LOGGER.info(f"Heston calib: {h_params}")

        # Merton jump component calibration
        LOGGER.info("\n--- Short (Jump-only) calibration ---")
        params_jump: JumpOnlyDynamicsParameters = BccFtiCalibrator.calibrate_jump(
            df_options=df_options,
            kappa_v=h_params.kappa_heston, theta_v=h_params.theta_heston, sigma_v=h_params.sigma_heston,
            rho=h_params.rho_heston, v0=h_params.v0_heston,
            jump_search_grid=jumponly_search_grid,
            S0=S0, side=side
        )
        initial_bates_params: BatesDynamicsParameters = BatesDynamicsParameters.from_dynamic_parameters(
            h_params=h_params, j_params=params_jump
        )
        if not skip_plot:
            kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta = initial_bates_params.get_values()
            plot_calibration_result(
                df_options=df_options,
                model_values=BatesFtiCalibrator.calculate_option_price_batch(
                    df_options=df_options,
                    S0=S0,
                    kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho=rho, v0=v0,
                    lambd=lambd, mu=mu, delta=delta,
                    side=side,
                    ft_method=ft_method
                ),
                side=side
            )

        ### Full calibration
        params_bcc: BccDynamicsParameters = BccFtiCalibrator.calibrate_full(
            df_options=df_options, S0=S0, r0=r0,
            initial_params=np.array(
                h_params.get_values() + params_jump.get_values()
            ),
            cir_params=params_cir,
            side=side,
            ft_method=ft_method
        )

        return params_bcc


def plot_BCC_full(bcc_params: BccDynamicsParameters, df_options: pd.DataFrame, S0:float, side: OptionSide):
    _, _, _, kappa_v, theta_v, sigma_v, rho, v0, lambd, mu, delta = bcc_params.get_values()
    df_options[OptionDataColumn.MODEL] = BccFtiCalibrator.calculate_option_price_batch(
        df_options=df_options, S0=S0,
        kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v, rho=rho, v0=v0,
        lambd=lambd, mu=mu, delta=delta,
        side=side, ft_method=FtMethod.LEWIS
    )
    for maturity, df_options_per_maturity in df_options.groupby(OptionDataColumn.DAYSTOMATURITY):
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.grid()
        plt.title(f"(Full-calib) {side.name} at Maturity {maturity}")
        plt.ylabel("option values")
        plt.plot(df_options_per_maturity[OptionDataColumn.STRIKE], df_options_per_maturity[side.name], "b",
                 label="market")
        plt.plot(df_options_per_maturity[OptionDataColumn.STRIKE], df_options_per_maturity[OptionDataColumn.MODEL],
                 "ro", label="model")
        plt.legend(loc=0)
        axis1 = [
            min(df_options_per_maturity[OptionDataColumn.STRIKE]) - 10,
            max(df_options_per_maturity[OptionDataColumn.STRIKE]) + 10,
            min(df_options_per_maturity[side.name]) - 10,
            max(df_options_per_maturity[side.name]) + 10,
        ]
        plt.axis(axis1)  # type: ignore

        plt.subplot(212)
        plt.grid()
        wi = 5.0
        diffs = df_options_per_maturity[OptionDataColumn.MODEL].values - df_options_per_maturity[side.name].values
        plt.bar(df_options_per_maturity[OptionDataColumn.STRIKE].values - wi / 2, diffs, width=wi)
        plt.ylabel("difference")
        axis2 = [
            min(df_options_per_maturity[OptionDataColumn.STRIKE]) - 10,
            max(df_options_per_maturity[OptionDataColumn.STRIKE]) + 10,
            min(diffs) * 1.1,
            max(diffs) * 1.1,
        ]
        plt.axis(axis2)  # type: ignore
        plt.tight_layout()
        plt.show()


def ex_pricing():
    bcc_params = BCCParameters(
        x0=100.,
        r=-0.032 / 100,
        T=1.,

        # We price with FTI for this example, hence the MC configs are not needed
        M=None,  # type: ignore
        I=None,  # type: ignore
        random_seed=None,  # type: ignore

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

    side: OptionSide = OptionSide.CALL
    x0, T, r, kappa_heston, theta_heston, sigma_heston, rho_heston, v0_heston, lambd_merton, mu_merton, delta_merton = bcc_params.get_pricing_params(
        apply_shortrate=False
    )
    K = 90.
    bates_price: float = FtiCalibrator.calculate_option_price_lewis(
        x0=x0, T=T, r=r, K=K,
        characteristic_integral=lambda u: BccFtiCalibrator.calculate_integral_characteristic(
            u=u, S0=x0, K=K, T=T, r=r,
            kappa_heston=kappa_heston, theta_heston=theta_heston, sigma_heston=sigma_heston, rho_heston=rho_heston, v0_heston=v0_heston,
            lambd_merton=lambd_merton, mu_merton=mu_merton, delta_merton=delta_merton,
        ),
        side=side
    )
    LOGGER.info(f"{side} Option value under Bates: {bates_price}")

    x0, T, r, kappa_heston, theta_heston, sigma_heston, rho_heston, v0_heston, lambd_merton, mu_merton, delta_merton = bcc_params.get_pricing_params(
        apply_shortrate=True
    )
    bcc_price: float = FtiCalibrator.calculate_option_price_lewis(
        x0=x0, T=T, r=r, K=90.,
        characteristic_integral=lambda u: BccFtiCalibrator.calculate_integral_characteristic(
            u=u, S0=x0, K=K, T=T, r=r,
            kappa_heston=kappa_heston, theta_heston=theta_heston, sigma_heston=sigma_heston, rho_heston=rho_heston, v0_heston=v0_heston,
            lambd_merton=lambd_merton, mu_merton=mu_merton, delta_merton=delta_merton,
        ),
        side=side
    )
    LOGGER.info(f"{side} Option value under BCC: {bcc_price}")


def ex_calibration(
        options_data_path: str,
        rates_data_path: str,
        side: OptionSide,
        skip_plot: bool
):
    S0 = 3225.93
    ft_method: FtMethod = FtMethod.LEWIS

    ### Load data on September 30, 2014 with r from the CIR model instead of constant overnight rate 0.005
    # Euribor Market data
    df_euribor: pd.DataFrame = pd.read_csv(rates_data_path)
    # EURO STOXX 50 level
    df_options: pd.DataFrame = load_option_data(
        path_str=options_data_path, S0=S0,
        r_provider=lambda *_, **__: np.nan,
        days_to_maturity_target=171
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


if __name__ == "__main__":
    ex_pricing()
    ex_calibration(
        options_data_path=get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv"),
        rates_data_path=get_path_from_package("erdqlib@src/ft/data/euribor_20140930.csv"),
        side=OptionSide.CALL,
        skip_plot=False
    )