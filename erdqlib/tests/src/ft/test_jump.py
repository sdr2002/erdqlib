from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from erdqlib.src.common.option import OptionSide, OptionInfo, OptionType
from erdqlib.src.ft.calibrator import FtiCalibrator, FtMethod
from erdqlib.src.ft.jump import JumpFtiCalibrator
from erdqlib.src.mc.jump import JumpDynamicsParameters, JumpParameters
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def jump_params():
    return JumpParameters(
        # Underlying
        x0=100.0,  # Current underlying asset price
        r=0.05,  # Risk-free rate
        # Monte Carlo
        T=1.0,
        M=None,
        I=None,
        random_seed=0,
        # Merton model
        lambd_merton=1.,
        mu_merton=-0.2,
        delta_merton=0.1,
        sigma_merton=0.4,
    )


def test_bates_option_price_fti(jump_params):
    """Test Bates model option pricing using Lewis' method."""
    # Calculate the option prices using Bates model
    option_info: OptionInfo = OptionInfo(
        o_type=OptionType.EUROPEAN,
        K=100.0,
        side=OptionSide.PUT,
    )
    for ft_method in FtMethod:
        option_price: float = JumpFtiCalibrator.calculate_option_price(
            S0=jump_params.x0,
            K=option_info.K,
            T=jump_params.T,
            r=jump_params.r,
            lambd=jump_params.lambd_merton,
            mu=jump_params.mu_merton,
            delta=jump_params.delta_merton,
            sigma=jump_params.sigma_merton,
            side=option_info.side,
            ft_method=ft_method,
        )

        np.testing.assert_approx_equal(option_price, 15.07, significant=4)


def test_jump_calibration():
    """Test Jump model calibration."""

    # Check if JumpFtiCalibrator is a subclass of FtiCalibrator
    assert issubclass(JumpFtiCalibrator, FtiCalibrator)

    # Perform calibration
    data_path: str = get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv")
    side: OptionSide = OptionSide.CALL

    S0: float = 3225.93 # EURO STOXX 50 level September 30, 2014
    r: float = 0.005

    df_options: pd.DataFrame = load_option_data(
        path_str=data_path,
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r,  # constant short rate
        days_to_maturity_target=17  # 17 days to maturity
    )

    params_jump: JumpDynamicsParameters = JumpFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        search_grid=JumpDynamicsParameters.get_default_search_grid()
    )

    compare_arrays_and_update(
        actual=params_jump.to_dataframe().values,
        expected_path=Path(get_path_from_package("erdqlib@tests/src/ft/data/test_jump_calibration.csv"))
    )
