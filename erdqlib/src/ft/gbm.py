import numpy as np
from scipy.integrate import quad

from erdqlib.src.common.option import OptionSide, OptionType
from erdqlib.src.ft.calibrator import FtiCalibrator, FtMethod


class GbmFtiCalibrator(FtiCalibrator):
    @staticmethod
    def calculate_characteristic(
            u: complex | np.ndarray, x0:float, T: float, r: float, sigma: float
    ) -> complex | np.ndarray:
        """
        Computes general Black-Scholes model characteristic function
        to be used in Fourier pricing methods like Lewis (2001) and Carr-Madan (1999):
        exp( i * u * log(S_0) + i * u*(r - 0.5 * sigma**2)* T - 0.5 * u^2 * sigma^2 * T )

        :param x0: log(S_0)
        :param u: u
        """
        return np.exp(((x0 / T + r - 0.5 * sigma ** 2) * 1j * u - 0.5 * sigma ** 2 * u ** 2) * T)

    @staticmethod
    def calculate_integral_characteristic(
            u: float, S0: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """
        Calculate the integral characteristic for the Geometric Brownian Motion (GBM) model.
        For GBM, this is not applicable as GBM does not use characteristic functions in this context.
        """
        cf_val: complex = GbmFtiCalibrator.calculate_characteristic(
            u=u - 0.5j, x0=np.log(S0/S0), T=T, r=r, sigma=sigma
        )
        phase = np.exp(1j * u * np.log(S0 / K))
        return (phase * cf_val).real / (u ** 2 + 0.25)

    @staticmethod
    def calculate_option_price_lewis(
            S0: float, K: float, T: float, r: float, sigma: float, side: OptionSide
    ) -> float:
        """
        Calculate the option price using the Lewis formula for the Geometric Brownian Motion (GBM) model.
        This is a simplified version and does not use characteristic functions.
        """
        int_value = quad(
            lambda u: GbmFtiCalibrator.calculate_integral_characteristic(
                u=u, S0=S0, K=K, T=T, r=r, sigma=sigma
            ),
            0,
            np.inf,
            limit=250,
        )[0]
        call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value)
        if side is OptionSide.CALL:
            return call_value
        elif side is OptionSide.PUT:
            return call_value - S0 + K * np.exp(-r * T)
        raise ValueError(f"Invalid side: {side}")

    @staticmethod
    def calculate_option_price_carrmadan(
            S0: float, K: float, T: float, r: float, sigma: float, side: OptionSide
    ) -> float:
        k = np.log(K / S0)
        x0 = np.log(S0 / S0)
        g = 1  # Factor to increase accuracy
        N = g * 4096
        eps = (g * 150) ** -1
        eta = 2 * np.pi / (N * eps)
        b = 0.5 * N * eps - k
        u = np.arange(1, N + 1, 1)
        vo = eta * (u - 1)

        # Modifications to ensure integrability
        if S0 >= 0.95 * K:  # ITM Case
            alpha = 1.5
            v = vo - (alpha + 1) * 1j
            modcharFunc = np.exp(-r * T) * (
                    GbmFtiCalibrator.calculate_characteristic(u=v, x0=x0, T=T, r=r, sigma=sigma)
                    / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)
            )

        else:
            alpha = 1.1
            v = (vo - 1j * alpha) - 1j
            modcharFunc1 = np.exp(-r * T) * (
                    1 / (1 + 1j * (vo - 1j * alpha))
                    - np.exp(r * T) / (1j * (vo - 1j * alpha))
                    - GbmFtiCalibrator.calculate_characteristic(u=v, x0=x0, T=T, r=r, sigma=sigma)
                    / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha))
            )

            v = (vo + 1j * alpha) - 1j
            modcharFunc2 = np.exp(-r * T) * (
                    1 / (1 + 1j * (vo + 1j * alpha))
                    - np.exp(r * T) / (1j * (vo + 1j * alpha))
                    - GbmFtiCalibrator.calculate_characteristic(u=v, x0=x0, T=T, r=r, sigma=sigma)
                    / ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha))
            )

        # Numerical FFT Routine
        delt = np.zeros(N)
        delt[0] = 1
        j = np.arange(1, N + 1, 1)
        SimpsonW = (3 + (-1) ** j - delt) / 3
        if S0 >= 0.95 * K:
            FFTFunc = np.exp(1j * b * vo) * modcharFunc * eta * SimpsonW
            payoff = (np.fft.fft(FFTFunc)).real
            CallValueM = np.exp(-alpha * k) / np.pi * payoff
        else:
            FFTFunc = (
                    np.exp(1j * b * vo) * (modcharFunc1 - modcharFunc2) * 0.5 * eta * SimpsonW
            )
            payoff = (np.fft.fft(FFTFunc)).real
            CallValueM = payoff / (np.sinh(alpha * k) * np.pi)

        pos: int = int((k + b) / eps)
        call_value: float = CallValueM[pos] * S0
        if side is OptionSide.CALL:
            return call_value
        elif side is OptionSide.PUT:
            return call_value - S0 + K * np.exp(-r * T)
        raise ValueError(f"Invalid side: {side}")

    @staticmethod
    def calculate_option_price(
            S0: float, K: float, T: float, r: float, sigma: float,
            side: OptionSide, o_type: OptionType = OptionType.EUROPEAN,
            ft_method: FtMethod = FtMethod.LEWIS
    ):
        if o_type is not OptionType.EUROPEAN:
            raise NotImplementedError()

        """Valuation of European call option in H93 model via Lewis (2001)

        Parameter definition:
        ==========
        S0: float
            initial stock/index level
        K: float
            strike price
        T: float
            time-to-maturity (for t=0)
        r: float
            constant risk-free short rate
        sigma: float
            volatility of variance
        Returns
        =======
        call_value: float
            present value of European call option
        """
        if ft_method is FtMethod.LEWIS:
            return GbmFtiCalibrator.calculate_option_price_lewis(
                S0=S0, K=K, T=T, r=r, sigma=sigma, side=side
            )
        elif ft_method is FtMethod.CARRMADAN:
            return GbmFtiCalibrator.calculate_option_price_carrmadan(
                S0=S0, K=K, T=T, r=r, sigma=sigma, side=side
            )
        raise ValueError(f"Invalid FtMethod method: {ft_method}")