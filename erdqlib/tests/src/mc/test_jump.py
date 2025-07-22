from pathlib import Path

import numpy as np
import pytest

from erdqlib.src.common.option import OptionInfo, OptionType, OptionSide
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.src.mc.jump import MertonJump, JumpParameters
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def jump_params():
    return JumpParameters(
        lambd_merton=0.75,  # Lambda of the model
        mu_merton=-0.6,  # Mu
        delta_merton=0.25,  # Delta
        sigma_merton=0.2,

        S0=100,  # Current underlying asset price
        r=0.05,  # Risk-free rate

        T=1,  # Number of years
        M=4,  # Total time steps
        I=10,  # Number of simulations
        random_seed=0
    )

def test_jump_paths_values(jump_params):
    paths: np.ndarray = MertonJump.calculate_paths(jump_params)
    compare_arrays_and_update(
        paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_jump_paths_values.csv"))
    )

def test_jump_eur_option_price(jump_params):
    s_paths: np.ndarray = MertonJump.calculate_paths(jump_params)
    call_price: float = price_montecarlo(
        underlying_path=s_paths,
        d=jump_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=95., side=OptionSide.CALL
        ),
        t=0.
    )

    put_price: float = price_montecarlo(
        underlying_path=s_paths,
        d=jump_params,
        o=OptionInfo(
            o_type=OptionType.EUROPEAN, K=105., side=OptionSide.PUT
        ),
        t=0.
    )

    expected_call_price: float = 1.8476007195750768
    np.testing.assert_approx_equal(call_price, expected_call_price, significant=4)

    expected_put_price: float = 44.77402984425289
    np.testing.assert_approx_equal(put_price, expected_put_price, significant=4)