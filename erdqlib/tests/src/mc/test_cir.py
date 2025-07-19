from pathlib import Path

import numpy as np
import pytest

from erdqlib.src.common.option import OptionInfo, OptionType, OptionSide
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.cir import Cir, CirParameters
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def cir_params():
    return CirParameters(
        T=1,
        M=4,
        I=10,
        random_seed=0,

        S0=0.03,
        k=0.20,
        theta=0.01,
        sigma=0.0012,
        r=0.001,  # shift parameter, which is Risk-free rate in risk-neutral measure
    )


def test_cir_paths_values(cir_params):
    paths: np.ndarray = Cir.calculate_paths(cir_params)
    compare_arrays_and_update(
        paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_cir_paths_values.csv"))
    )


def test_cir_eur_option_price(cir_params):
    s_paths: np.ndarray = Cir.calculate_paths(cir_params)
    call_price: float = price_montecarlo(
        underlying_path=s_paths,
        d=cir_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=0.02, side=OptionSide.CALL
        ),
        t=0.
    )

    put_price: float = price_montecarlo(
        underlying_path=s_paths,
        d=cir_params,
        o=OptionInfo(
            type=OptionType.EUROPEAN, K=0.04, side=OptionSide.PUT
        ),
        t=0.
    )

    expected_call_price: float = 0.006326179843270637
    np.testing.assert_approx_equal(call_price, expected_call_price, significant=4)

    expected_put_price: float = 0.013653830153396865
    np.testing.assert_approx_equal(put_price, expected_put_price, significant=4)