from pathlib import Path

import numpy as np
import pytest

from erdqlib.src.common.option import OptionInfo, OptionType, OptionSide
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.vasicek import Vasicek, VasicekParameters
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def vasicek_params():
    return VasicekParameters(
        T=1,
        M=4,
        I=10,
        random_seed=0,

        x0=0.03,
        kappa_vasicek=0.20,
        theta_vasicek=0.01,
        sigma_vasicek=0.0012,
        r=0.001,  # shift parameter, which is Risk-free rate in risk-neutral measure
    )


def test_vasicek_paths_values(vasicek_params):
    paths: np.ndarray = Vasicek.calculate_paths(vasicek_params)
    compare_arrays_and_update(
        paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_vasicek_paths_values.csv"))
    )


def test_vasicek_eur_option_price(vasicek_params):
    s_paths: np.ndarray = Vasicek.calculate_paths(vasicek_params)
    call_price: float = price_montecarlo(
        underlying_path=s_paths,
        model_params=vasicek_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=0.02, side=OptionSide.CALL
        ),
        t_i=0.
    )

    put_price: float = price_montecarlo(
        underlying_path=s_paths,
        model_params=vasicek_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN, K=0.04, side=OptionSide.PUT
        ),
        t_i=0.
    )

    expected_call_price: float = 0.006306683189013548  # Replace with actual expected value
    np.testing.assert_approx_equal(call_price, expected_call_price, significant=4)

    expected_put_price: float = 0.013673326807653955 # Replace with actual expected value
    np.testing.assert_approx_equal(put_price, expected_put_price, significant=4)
