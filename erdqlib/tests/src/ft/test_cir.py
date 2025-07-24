import numpy as np
import pandas as pd
import pytest

from erdqlib.src.common.option import OptionDataColumn
from erdqlib.src.common.rate import SplineCurve, ForwardsLadder
from erdqlib.src.ft.cir import CirCalibrator
from erdqlib.src.mc.cir import CirDynamicsParameters
from erdqlib.tool.logger_util import create_logger
from erdqlib.tool.path import get_path_from_package

LOGGER = create_logger(__name__)


@pytest.fixture
def cir_params():
    return CirDynamicsParameters(
        x0=-0.000320835506166735,
        r=None,  # this is dynamics for the short rate, there is no risk-free drift factor kind of thing
        kappa_cir=0.3763235887119173,
        theta_cir=0.04488952147663766,
        sigma_cir=0.18380960611583302
    )


def test_cir_shortrate_calibration(cir_params):
    # Euribor Market data
    euribor_df: pd.DataFrame = pd.read_csv(
        get_path_from_package("erdqlib@src/ft/data/euribor_20140930.csv")
    )
    LOGGER.info(f"Rates data:\n{euribor_df.to_markdown(index=False)}")

    maturities: np.ndarray = euribor_df[OptionDataColumn.MATURITY].values  # Maturities in years with 30/360 convention
    euribors: np.ndarray = euribor_df[OptionDataColumn.RATE].values  # Euribor rates in rate unit

    # Interpolation and Forward rates via Cubic spline
    scurve: SplineCurve = SplineCurve()
    scurve.update_curve(maturities=maturities, spot_rates=euribors)
    forward_rates: ForwardsLadder = scurve.calculate_forward_rates(t_f=1.0)

    # Calibration of CIR parameters
    params_cir: CirDynamicsParameters = CirCalibrator.calibrate(
        r0=scurve.get_r0(source="overnight"),
        curve_forward_rates=forward_rates.rates,
        maturities_ladder=forward_rates.maturities
    )

    assert params_cir == cir_params
