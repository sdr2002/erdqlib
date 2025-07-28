from pathlib import Path

import numpy as np
import pytest

from erdqlib.src.common.option import OptionInfo, OptionType, OptionSide
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.gbm import Gbm, GbmParameters
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def gbm_params():
    return GbmParameters(
        sigma=0.2,
        x0=100,
        r=0.0001,
        T=1,
        M=4,
        I=10,
        random_seed=0
    )


def test_gbm_paths_values(gbm_params):
    paths: np.ndarray = Gbm.calculate_paths(gbm_params)
    compare_arrays_and_update(
        paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_gbm_paths_values.csv"))
    )


def test_gbm_eur_option_price(gbm_params):
    s_paths: np.ndarray = Gbm.calculate_paths(gbm_params)
    call_price: float = price_montecarlo(
        underlying_path=s_paths,
        model_params=gbm_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t_i=0.
    )

    put_price: float = price_montecarlo(
        underlying_path=s_paths,
        model_params=gbm_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t_i=0.
    )

    expected_call_price: float = 8.159436544125452
    np.testing.assert_approx_equal(call_price, expected_call_price, significant=4)

    expected_put_price: float = 9.795892828816381
    np.testing.assert_approx_equal(put_price, expected_put_price, significant=4)
