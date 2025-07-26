from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from erdqlib.src.common.option import OptionDataColumn
from erdqlib.src.common.rate import SplineCurve, ForwardsLadder
from erdqlib.src.ft.cir import CirCalibrator
from erdqlib.src.mc.cir import CirDynamicsParameters
from erdqlib.tests.test_data import compare_arrays_and_update
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


def test_cir_forward_rate(cir_params):
    """Test the forward rate calculation for the CIR model."""
    maturities_ladder: np.ndarray = np.array([0., 0.25, 0.5, 1.0])  # Example maturities in years
    r0: float = cir_params.x0

    forward_rates: np.ndarray = CirCalibrator.calculate_forward_rate(
        alpha=np.array([cir_params.kappa_cir, cir_params.theta_cir, cir_params.sigma_cir]),
        maturities_ladder=maturities_ladder,
        r0=r0
    )

    LOGGER.info(f"Forward rates:\n{forward_rates}")
    np.testing.assert_array_almost_equal(
        forward_rates,
        np.array([-0.00029348392530115946, 0.0016899649181064454, 0.003496827391669189, 0.006651532318784888])
    )


def test_cir_shortrate_calibration():
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

    compare_arrays_and_update(
        actual=params_cir.to_dataframe().values,
        expected_path=Path(get_path_from_package("erdqlib@tests/src/ft/data/test_cir_calibration.csv"))
    )
