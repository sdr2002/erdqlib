import pandas as pd
import pytest

from erdqlib.src.common.option import OptionSide
from erdqlib.src.ft.calibrator import FtiCalibrator
from erdqlib.src.ft.heston import HestonFtiCalibrator
from erdqlib.src.mc.heston import HestonDynamicsParameters
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def heston_params():
    return HestonDynamicsParameters(
        kappa_heston=5.330467739880874,
        theta_heston=0.018719611049762303,
        sigma_heston=0.44673097677209606,
        rho_heston=-0.6141271947376961,
        v0_heston=0.027766599221167913,

        x0=3225.93,  # Current underlying asset price
        r=0.02,  # Risk-free rate
    )


def test_heston_calibration(heston_params):
    """Test Heston model calibration."""
    # Check if HestonCalibrator is a subclass of FtiCalibrator
    assert issubclass(HestonFtiCalibrator, FtiCalibrator)

    # Perform calibration
    # Market Data from www.eurexchange.com
    # as of September 30, 2014
    data_path: str = get_path_from_package("erdqlib@src/ft/data/stoxx50_20140930.csv")
    side: OptionSide = OptionSide.CALL

    S0: float = heston_params.x0  # EURO STOXX 50 level September 30, 2014
    r: float = heston_params.r

    df_options: pd.DataFrame = load_option_data(
        path_str=data_path,
        S0=S0,  # EURO STOXX 50 level September 30, 2014
        r_provider=lambda *_: r  # constant short rate
    )

    opt_params: HestonDynamicsParameters = HestonFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        search_grid=HestonDynamicsParameters.get_default_search_grid()
    )

    assert opt_params == heston_params
