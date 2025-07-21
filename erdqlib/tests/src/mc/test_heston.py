from pathlib import Path

import numpy as np
import pytest

from erdqlib.src.common.option import OptionInfo, OptionType, OptionSide
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.heston import Heston, HestonParameters
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def heston_params():
    return HestonParameters(
        v0_heston=0.04,
        kappa_heston=2,
        sigma_heston=0.3,
        theta_heston=0.04,
        rho_heston=-0.9,
        S0=100,
        r=0.05,
        T=1,
        M=4,
        I=10,
        random_seed=0
    )

def test_heston_paths_values(heston_params):
    var_paths, s_paths = Heston.calculate_paths(heston_params) # type: np.ndarray, np.ndarray
    compare_arrays_and_update(
        s_paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_heston_paths_values_s.csv"))
    )
    compare_arrays_and_update(
        var_paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_heston_paths_values_var.csv"))
    )

def test_heston_eur_option_price(heston_params):
    var_paths, s_paths = Heston.calculate_paths(heston_params)  # type: np.ndarray, np.ndarray
    call_price: float = price_montecarlo(
        underlying_path=s_paths,
        d=heston_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )

    put_price: float = price_montecarlo(
        underlying_path=s_paths,
        d=heston_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )

    expected_call_price: float = 9.250468788265543
    np.testing.assert_approx_equal(call_price, expected_call_price, significant=4)

    expected_put_price: float = 6.905996174060089
    np.testing.assert_approx_equal(put_price, expected_put_price, significant=4)