import numpy as np


class TrinomialModel(object):  # Here we start defining our 'class' --> Trinomial Model!
    # First, a method to initialize our `TrinomialModel` algorithm!
    def __init__(self, S_ini, r, sigma, mat):
        self.__s0 = S_ini
        self.__r = r
        self.__sigma = sigma
        self.__T = mat

    # Second, we build a method (function) to compute the risk-neutral probabilities!
    def __compute_probs(self):
        self.__pu = (
                            (
                                    np.exp(self.__r * self.__h / 2)
                                    - np.exp(-self.__sigma * np.sqrt(self.__h / 2))
                            )
                            / (
                                    np.exp(self.__sigma * np.sqrt(self.__h / 2))
                                    - np.exp(-self.__sigma * np.sqrt(self.__h / 2))
                            )
                    ) ** 2
        self.__pd = (
                            (
                                    -np.exp(self.__r * self.__h / 2)
                                    + np.exp(self.__sigma * np.sqrt(self.__h / 2))
                            )
                            / (
                                    np.exp(self.__sigma * np.sqrt(self.__h / 2))
                                    - np.exp(-self.__sigma * np.sqrt(self.__h / 2))
                            )
                    ) ** 2
        self.__pm = 1 - self.__pu - self.__pd

        assert 0 <= self.__pu <= 1.0, "p_u should lie in [0, 1] given %s" % self.__pu
        assert 0 <= self.__pd <= 1.0, "p_d should lie in [0, 1] given %s" % self.__pd
        assert 0 <= self.__pm <= 1.0, "p_m should lie in [0, 1] given %s" % self.__pm

    def get_probabilities(self):
        return self.__pu, self.__pm, self.__pd

    # Third, this method checks whether the given parameters are alright and that we have a 'recombining tree'!
    @staticmethod
    def get_transition_multiplier(sigma, h) -> tuple([float, float]):
        up = np.exp(sigma * np.sqrt(2 * h))
        down = 1 / up

        assert up > 0.0, "up should be non negative"
        assert down < up, "up <= 1. / up = down"

        return up, down

    def __check_up_value(self, up=None):
        if up is None:
            self.__up, self.__down = TrinomialModel.get_transition_multiplier(sigma=self.__sigma, h=self.__h)
        else:
            assert up > 0.0, "up should be non negative"
            self.__up = up
            self.__down = 1 / up
            assert self.__down < up, "up <= 1. / up = down"

    # Four, we use this method to compute underlying stock price path
    def __gen_stock_vec(self, nb):
        vec_u = self.__up * np.ones(nb)
        np.cumprod(vec_u, out=vec_u)

        vec_d = self.__down * np.ones(nb)
        np.cumprod(vec_d, out=vec_d)

        res = np.concatenate((vec_d[::-1], [1.0], vec_u))
        res *= self.__s0

        return res

    # Fifth, we declare a Payoff method to be completed afterwards depending on the instrument we are pricing!
    def payoff(self, stock_vec):
        raise NotImplementedError()

    # Sixth, compute current prices!
    def compute_current_price(self, crt_vec_stock, nxt_vec_option):
        expectation = np.zeros(crt_vec_stock.size)
        for i in range(expectation.size):
            tmp = nxt_vec_option[i] * self.__pd
            tmp += nxt_vec_option[i + 1] * self.__pm
            tmp += nxt_vec_option[i + 2] * self.__pu

            expectation[i] = tmp

        return self.__discount * expectation

    # Seventh, Option pricing!
    def price(self, N: int, up: float = None):
        assert N > 0, "N shoud be > 0"

        N = int(N)

        S = np.zeros([N + 1, 2 * N + 1])  # underlying price
        V = np.zeros([N + 1, 2 * N + 1])  # option prices

        self.__h = self.__T / N
        self.__check_up_value(up)
        self.__compute_probs()

        self.__discount = np.exp(-self.__r * self.__h)

        final_vec_stock = self.__gen_stock_vec(N)
        S[-1, :len(final_vec_stock)] = final_vec_stock
        final_payoff = self.payoff(final_vec_stock)
        nxt_vec_option = final_payoff
        V[-1, :len(nxt_vec_option)] = nxt_vec_option

        for i in range(1, N + 1):
            vec_stock = self.__gen_stock_vec(N - i)
            S[N - i, :len(vec_stock)] = vec_stock

            nxt_vec_option = self.compute_current_price(vec_stock, nxt_vec_option)
            V[N - i, :len(nxt_vec_option)] = nxt_vec_option

        return V, S


# S_ini, K, T, r, sigma, N, opttype: OptionType, side: Side

class TrinomialEurCall(TrinomialModel):
    def __init__(self, S_ini, K, T, r, sigma):
        super(TrinomialEurCall, self).__init__(S_ini, r, sigma, T)
        self.__K = K

    def payoff(self, s):
        return np.maximum(s - self.__K, 0.0)


class TrinomialEurPut(TrinomialModel):
    def __init__(self, S_ini, K, T, r, sigma):
        super(TrinomialEurPut, self).__init__(S_ini, r, sigma, T)
        self.__K = K

    def payoff(self, s):
        return np.maximum(- s + self.__K, 0.0)


class TrinomialAmeCall(TrinomialModel):
    def __init__(self, S_ini, K, T, r, sigma):
        super(TrinomialAmeCall, self).__init__(S_ini, r, sigma, T)
        self.__K = K

    def payoff(self, s):
        return np.maximum(s - self.__K, 0.0)

    def compute_current_price(self, vec_stock, nxt_vec_option):
        nxt_vec_option = super().compute_current_price(vec_stock, nxt_vec_option)
        return np.maximum(nxt_vec_option, vec_stock - self.__K)


class TrinomialAmePut(TrinomialModel):
    def __init__(self, S_ini, K, T, r, sigma):
        super(TrinomialAmePut, self).__init__(S_ini, r, sigma, T)
        self.__K = K

    def payoff(self, s):
        return np.maximum(- s + self.__K, 0.0)

    def compute_current_price(self, vec_stock, nxt_vec_option):
        nxt_vec_option = super().compute_current_price(vec_stock, nxt_vec_option)
        return np.maximum(nxt_vec_option, -vec_stock + self.__K)