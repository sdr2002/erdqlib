import pandas as pd
import pytest

from erdqlib.src.common.option import OptionSide
from erdqlib.src.ft.calibrator import FtiCalibrator
from erdqlib.src.ft.jump import JumpFtiCalibrator
from erdqlib.src.mc.jump import JumpDynamicsParameters
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def jump_params():
    return JumpDynamicsParameters(
        lambd_merton=0.009200997845176707,
        mu_merton=-0.2038003415438602,
        delta_merton=0.07715498913481206,
        sigma_merton=0.15619381366824533,

        x0=3225.93,  # Current underlying asset price
        r=0.005,  # Risk-free rate
    )


def test_jump_calibration(jump_params):
    """Test Jump model calibration."""

    # Check if JumpFtiCalibrator is a subclass of FtiCalibrator
    assert issubclass(JumpFtiCalibrator, FtiCalibrator)

    # Perform calibration
    data_path: str = get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv")
    side: OptionSide = OptionSide.CALL

    S0: float = jump_params.x0  # EURO STOXX 50 level September 30, 2014
    r: float = jump_params.r

    df_options: pd.DataFrame = load_option_data(
        path_str=data_path,
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r  # constant short rate
    )

    opt_params: JumpDynamicsParameters = JumpFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        search_grid=JumpDynamicsParameters.get_default_search_grid()
    )

    assert opt_params == jump_params