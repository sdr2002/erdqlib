import numpy as np
import pandas as pd

from erdqlib.src.ft.calibrator import FtiCalibrator
from erdqlib.src.ft.jump import JumpFtiCalibrator
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.src.common.option import OptionSide
from erdqlib.src.mc.jump import JumpDynamicsParameters
from erdqlib.tool.path import get_path_from_package


def test_jump_calibration():
    """Test Jump model calibration."""

    # Check if JumpFtiCalibrator is a subclass of FtiCalibrator
    assert issubclass(JumpFtiCalibrator, FtiCalibrator)

    # Perform calibration
    data_path: str = get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv")
    side: OptionSide = OptionSide.CALL

    S0: float = 3225.93  # EURO STOXX 50 level September 30, 2014
    r: float = 0.005

    df_options: pd.DataFrame = load_option_data(
        path_str=data_path,
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r  # constant short rate
    )

    opt_params: JumpDynamicsParameters = JumpFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        search_grid=JumpDynamicsParameters.get_default_search_grid()
    )

    expected_jump_params: JumpDynamicsParameters = JumpDynamicsParameters(
        S0=S0,
        r=r,
        sigma_merton=0.15619381,
        lambd_merton=0.009201,
        mu_merton=-0.20380034,
        delta_merton=0.07715499,
    )

    # Assert the values of the calibrated parameters equal the expected values
    for param in vars(expected_jump_params).keys():
        np.testing.assert_approx_equal(
            getattr(opt_params, param),
            getattr(expected_jump_params, param),
            significant=3,
            err_msg=f"Parameter {param} does not match expected value."
        )