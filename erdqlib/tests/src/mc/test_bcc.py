from pathlib import Path

import numpy as np
import pytest

from erdqlib.src.common.option import OptionInfo, OptionType, OptionSide
from erdqlib.src.mc.bcc import BCC, BCCParameters
from erdqlib.src.mc.evaluate import price_montecarlo
from erdqlib.tests.test_data import compare_arrays_and_update
from erdqlib.tool.path import get_path_from_package


@pytest.fixture
def bcc_params():
    return BCCParameters(
        # Underlying
        x0=10.65,
        # Monte Carlo
        T=5./365.,
        M=5,  # type=ignore
        I=10,  # type=ignore
        random_seed=0,  # type=ignore
        # Rate (CIR)
        r=0.,  # r for BCC works as the r0 for short-rate dynamics
        kappa_cir=0.688,
        theta_cir=0.109,
        sigma_cir=0.001,
        # Stochastic Volatility (Heston)
        kappa_heston=0.86,
        theta_heston=0.16,
        sigma_heston=0.15,
        rho_heston=-0.95,
        v0_heston=0.016,
        # Jump (Merton's Jump diffusion)
        lambd_merton=0.1,
        mu_merton=-0.05,
        delta_merton=0.9,
    )


def test_bcc_paths(bcc_params):
    r_paths, var_paths, s_paths = BCC.calculate_paths(model_params=bcc_params)

    compare_arrays_and_update(
        r_paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_bcc_paths_values_r.csv"))
    )
    compare_arrays_and_update(
        var_paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_bcc_paths_values_var.csv"))
    )
    compare_arrays_and_update(
        s_paths,
        Path(get_path_from_package("erdqlib@tests/src/mc/data/test_bcc_paths_values_s.csv"))
    )

def test_bcc_asian_option_price(bcc_params):
    bcc_params.I = 10_000

    r_paths, var_paths, s_paths = BCC.calculate_paths(model_params=bcc_params)

    put_eur_price: float = price_montecarlo(
        underlying_path=s_paths,
        model_params=bcc_params,
        o_info=OptionInfo(
            o_type=OptionType.EUROPEAN,
            K=22.,  # strike price
            side=OptionSide.PUT,
        )
    )

    expected_eur_put_price: float = 11.361027110174069
    np.testing.assert_approx_equal(put_eur_price, expected_eur_put_price, significant=4)

    put_asian_price: float = price_montecarlo(
        underlying_path=s_paths,
        model_params=bcc_params,
        o_info=OptionInfo(
            o_type= OptionType.ASIAN,
            K=22.,  # strike price
            side=OptionSide.PUT,
        )
    )

    expected_asian_put_price: float = 11.355110303648043
    np.testing.assert_approx_equal(put_asian_price, expected_asian_put_price, significant=4)
