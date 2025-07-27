from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from erdqlib.src.common.option import OptionSide, OptionInfo, OptionType
from erdqlib.src.ft.calibrator import FtiCalibrator, FtMethod
from erdqlib.src.ft.heston import HestonFtiCalibrator
from erdqlib.src.mc.heston import HestonDynamicsParameters, HestonParameters
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def heston_params():
    return HestonParameters(
        # Underlying
        x0=100.0,  # Current underlying asset price
        r=0.02,  # Risk-free rate
        # Monte Carlo
        T=0.5,  # Time to maturity in years
        M=None,  # Number of time steps
        I=None,  # Number of paths
        random_seed=0,  # Random seed for reproducibility
        # Heston model parameters
        kappa_heston=1.5,
        theta_heston=0.02,
        sigma_heston=0.15,
        rho_heston=0.1,
        v0_heston=0.01,
    )


def test_bates_option_price_fti(heston_params):
    """Test Bates model option pricing using Lewis' method."""
    # Calculate the option prices using Bates model
    option_info: OptionInfo = OptionInfo(
        o_type=OptionType.EUROPEAN,
        K=100.0,
        side=OptionSide.PUT,
    )
    for ft_method in FtMethod:
        option_price: float = HestonFtiCalibrator.calculate_option_price(
            S0=heston_params.x0,
            K=option_info.K,
            T=heston_params.T,
            r=heston_params.r,
            kappa_v=heston_params.kappa_heston,
            theta_v=heston_params.theta_heston,
            sigma_v=heston_params.sigma_heston,
            rho_v=heston_params.rho_heston,
            v0=heston_params.v0_heston,
            side=option_info.side,
            ft_method=ft_method,
        )

        np.testing.assert_approx_equal(option_price, 2.659, significant=4)


def test_heston_calibration(heston_params):
    """Test Heston model calibration."""
    # Check if HestonCalibrator is a subclass of FtiCalibrator
    assert issubclass(HestonFtiCalibrator, FtiCalibrator)

    # Perform calibration
    # Market Data from www.eurexchange.com
    # as of September 30, 2014
    data_path: str = get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv")
    side: OptionSide = OptionSide.CALL

    S0: float = 3225.93  # EURO STOXX 50 level September 30, 2014
    r: float = 0.02

    df_options: pd.DataFrame = load_option_data(
        path_str=data_path,
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r,  # constant short rate
        days_to_maturity_target=17  # 17 days to maturity
    )

    params_heston: HestonDynamicsParameters = HestonFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        search_grid=HestonDynamicsParameters.get_default_search_grid()
    )

    compare_arrays_and_update(
        actual=params_heston.to_dataframe().values,
        expected_path=Path(get_path_from_package("erdqlib@tests/src/ft/data/test_heston_calibration.csv"))
    )
