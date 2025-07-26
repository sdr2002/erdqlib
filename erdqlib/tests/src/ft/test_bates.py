from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from erdqlib.src.common.option import OptionSide, OptionType, OptionInfo
from erdqlib.src.ft.bates import BatesFtiCalibrator
from erdqlib.src.ft.calibrator import FtiCalibrator, FtMethod
from erdqlib.src.mc.bates import BatesParameters, BatesDynamicsParameters
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def bates_params():
    return BatesParameters(
        # Underlying
        x0=10.65,
        r=0.,
        # Monte Carlo
        T=5./365.,
        M=None,
        I=None,
        random_seed=0,
        # Stochastic Volatility (Bates)
        kappa_heston=0.86,
        theta_heston=0.16,
        sigma_heston=0.15,
        rho_heston=-0.95,
        v0_heston=0.016,
        # Jump
        lambd_merton=0.1,
        mu_merton=-0.05,
        delta_merton=0.9,
    )

def test_bates_option_price_fti(bates_params):
    """Test Bates model option pricing using Lewis' method."""
    # Calculate the option prices using Bates model
    option_info: OptionInfo = OptionInfo(
        o_type=OptionType.EUROPEAN,
        K=22.0,
        side=OptionSide.PUT,
    )
    for ft_method in FtMethod:
        option_price: float = BatesFtiCalibrator.calculate_option_price(
            S0=bates_params.x0,
            K=option_info.K,
            T=bates_params.T,
            r=bates_params.r,
            kappa_v=bates_params.kappa_heston,
            theta_v=bates_params.theta_heston,
            sigma_v=bates_params.sigma_heston,
            rho_v=bates_params.rho_heston,
            v0=bates_params.v0_heston,
            lambd=bates_params.lambd_merton,
            mu=bates_params.mu_merton,
            delta=bates_params.delta_merton,
            side=option_info.side,
            ft_method=ft_method,
        )

        np.testing.assert_approx_equal(option_price, 11.35, significant=4)


def test_bates_calibration():
    """Test Bates model calibration."""
    # Check if BatesCalibrator is a subclass of FtiCalibrator
    assert issubclass(BatesFtiCalibrator, FtiCalibrator)

    data_path: str = get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv")
    side: OptionSide = OptionSide.CALL

    S0: float = 3225.93  # EURO STOXX 50 level 30.09.2014
    r: float = 0.02
    ft_method: FtMethod = FtMethod.LEWIS  # or FtMethod.CARRMADAN

    df_options: pd.DataFrame = load_option_data(
        path_str=data_path,
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r,  # constant short rate
        days_to_maturity_target=17 # 17 days to maturity
    )

    params_bates: BatesDynamicsParameters = BatesFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        heston_search_grid=BatesDynamicsParameters.get_default_heston_search_grid(),
        jumponly_search_grid=BatesDynamicsParameters.get_default_jumponly_search_grid(),
        ft_method=ft_method,
        h_params_path=get_path_from_package("erdqlib@src/ft/data/opt_sv_M2.csv")
    )

    compare_arrays_and_update(
        actual=params_bates.to_dataframe().values,
        expected_path=Path(get_path_from_package("erdqlib@tests/src/ft/data/test_bates_calibration.csv"))
    )
