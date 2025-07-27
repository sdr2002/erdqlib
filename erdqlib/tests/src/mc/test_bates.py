from pathlib import Path

import numpy as np
import pytest

from erdqlib.src.common.option import OptionInfo, OptionType, OptionSide
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.bates import Bates, BatesParameters
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def bates_params():
    return BatesParameters(
        # Underlying
        x0=10.65,
        r=0.,
        # Monte Carlo
        T=5./365.,
        M=5,
        I=10,
        random_seed=0,
        # Stochastic Volatility (Heston)
        kappa_heston=0.86,
        theta_heston=0.16,
        sigma_heston=0.15,
        rho_heston=-0.95,
        v0_heston=0.016,
        # Jump
        lambd_merton=0.1,
        mu_merton=-0.05,
        delta_merton=0.9,
    )

def test_bates_paths_values(bates_params):
    var_paths, s_paths = Bates.calculate_paths(bates_params) # type: np.ndarray, np.ndarray
    compare_arrays_and_update(
        s_paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_bates_paths_values_s.csv"))
    )
    compare_arrays_and_update(
        var_paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_bates_paths_values_var.csv"))
    )

def test_bates_eur_option_price(bates_params):
    bates_params.I = 10_000

    var_paths, s_paths = Bates.calculate_paths(bates_params)  # type: np.ndarray, np.ndarray
    call_price: float = price_montecarlo(
        underlying_path=s_paths,
        d=bates_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=22., side=OptionSide.CALL
        ),
        t=0.
    )

    put_price: float = price_montecarlo(
        underlying_path=s_paths,
        d=bates_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=22., side=OptionSide.PUT
        ),
        t=0.
    )

    expected_call_price: float = 5.034410794188844e-05
    np.testing.assert_approx_equal(call_price, expected_call_price, significant=4)

    expected_put_price: float = 11.361086815435737
    np.testing.assert_approx_equal(put_price, expected_put_price, significant=4)