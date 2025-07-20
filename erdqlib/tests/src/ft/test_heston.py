import numpy as np
import pandas as pd

from erdqlib.src.common.option import OptionSide
from erdqlib.src.ft.calibrator import FtiCalibrator
from erdqlib.src.util.data_loader import load_option_data
from erdqlib.src.ft.heston import HestonFtiCalibrator
from erdqlib.src.mc.heston import HestonDynamicsParameters
from erdqlib.tool.path import get_path_from_package


def test_heston_calibration():
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
        r_provider=lambda *_: r  # constant short rate
    )

    opt_params: HestonDynamicsParameters = HestonFtiCalibrator.calibrate(
        df_options=df_options, S0=S0, r=r, side=side,
        search_grid=HestonDynamicsParameters.get_default_search_grid()
    )

    expected_heston_params: HestonDynamicsParameters = HestonDynamicsParameters(
        S0=S0,
        r=r,
        kappa_heston=5.047,
        theta_heston=0.018,
        sigma_heston=0.434,
        rho_heston=-0.447,
        v0_heston=0.027,
    )

    # Assert the values of the calibrated parameters equal the expected values
    for param in vars(expected_heston_params).keys():
        np.testing.assert_approx_equal(
            getattr(opt_params, param),
            getattr(expected_heston_params, param),
            significant=2,
            err_msg=f"Parameter {param} does not match expected value."
        )
