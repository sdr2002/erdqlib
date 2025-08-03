import numpy as np
from numpy.random import seed, rand

from erdqlib.plib.hmm import HmmFilter
from m4.markovchain_ar1 import rouwenhurst_method, tauchen_method


def hmm_smoother_params(y_, xi_1_2, xi_2_2):
    mu_ = np.zeros(2)
    sig_ = np.zeros(2)
    for k in range(2):
        xi_12_k = np.array([xi_1_2[k], xi_2_2[k]])
        mu_[k] = np.sum(xi_12_k * y_)/np.sum(xi_12_k)
        sig_[k] = np.sqrt(np.sum(xi_12_k * (y_ - mu_[k])**2)/np.sum(xi_12_k))

    return "mu:", mu_, "sig", sig_


def _credit_rating_counts_to_p__(
    P0 = np.array(
        [
            [87.06, 9.06, 0.53, 0.05, 0.11, 0.03, 0.05, 0.0, 3.11],
            [0.48, 87.23, 7.77, 0.47, 0.05, 0.06, 0.02, 0.02, 3.89],
            [0.03, 1.6, 88.58, 5.0, 0.26, 0.11, 0.02, 0.05, 4.35],
            [0, 0.09, 3.25, 86.49, 3.56, 0.43, 0.1, 0.16, 5.92],
            [0.01, 0.03, 0.11, 4.55, 77.82, 6.8, 0.55, 0.63, 9.51],
            [0.0, 0.02, 0.07, 0.15, 4.54, 74.6, 4.96, 3.34, 12.33],
            [0.0, 0.0, 0.1, 0.17, 0.55, 12.47, 43.11, 28.3, 15.31],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0],
        ]
    )
):
    """
    Convert the transition matrix P0 to a normalized transition matrix P.
    The last column of P0 is ignored as it represents the NR (not rated) state.
    """
    P = P0[:, 0:P0.shape[1] - 1]  # noQA E203
    P = (P.T / np.sum(P, axis=1)).T
    return P


def bond_forward_price(
    # Bond static info
    RATINGS = dict(
        [
            ("AAA", 0),
            ("AA", 1),
            ("A", 2),
            ("BBB", 3),
            ("BB", 4),
            ("B", 5),
            ("CCC", 6),
            ("D", 7),
        ]
    ),  # noqa: B006
    CURR_RATING = "AA",
    FACE_VALUE = 100.0,
    COUPON = 7.0,
    D_RECOVERY = 32.5,
    # VAR_PR = 99.0  # unused

    p__ = None,
    F = np.array(
        [
            [2.70, 3.13, 3.55, 3.84],
            [2.74, 3.17, 3.59, 3.88],
            [2.79, 3.24, 3.70, 3.99],
            [3.08, 3.50, 3.94, 4.22],
            [4.16, 4.52, 5.09, 5.45],
            [4.54, 5.27, 6.02, 6.39],
            [11.29, 11.27, 10.52, 10.14],
        ]
    )
):
    # print("Bond maturity: ", MAT)
    print("Transition Matrix (normalized):\n", p__)
    print("Forward rates matrix\n", F)

    assert len(RATINGS) == p__.shape[0]
    N_RATINGS = len(RATINGS)
    # print("Number of Ratings =", N_RATINGS)
    N_YEARS = F.shape[1]
    # print("Number of years in the forward rates matrix =", N_YEARS)

    bond_values = np.zeros(N_RATINGS)
    bond_values[N_RATINGS - 1] = D_RECOVERY

    # Forward rates applied to multi-year discount
    '''
    It re-uses a single forward rate to stand in for a multi-year discount, 
    or tried to sum independently discounted cash-flows as if they were spot-discounted.
    '''
    for r in range(0, N_RATINGS - 1):
        bond_values[r] = COUPON
        for t in range(N_YEARS):
            payoff = COUPON + FACE_VALUE * (t == N_YEARS - 1)
            bond_values[r] += payoff / (1 + F[r, t] / 100.0) ** (t + 1)

    # # I guess whether this must be the correct version:
    # for r in range(0, N_RATINGS - 1):
    #     bond_values[r] = 0.0
    #     for t in range(N_YEARS-1, -1, -1):
    #         is_maturity = (N_YEARS - 1 == t)
    #         bond_values[r] += (COUPON + FACE_VALUE * is_maturity) / (1 + (F[r, t] / FACE_VALUE))
    #     bond_values[r] += COUPON

    print("Bond values by rating:\n ", bond_values)
    print("P.transition:\n ", p__[RATINGS[CURR_RATING], :])
    pw_values = np.multiply(bond_values, p__[RATINGS[CURR_RATING], :])
    bond_val = np.sum(pw_values)
    print("Bond value (one-year ahead): ", bond_val)

DEFAULT_RATING = "F"

def credit_default_time(
    p__ = np.array([[0.9, 0.09, 0.01], [0.2, 0.75, 0.05], [0, 0, 1]]),
    CURR_RATING = "I",
    RATINGS = dict(
        [
            ("I", 0),
            ("S", 1),
            (DEFAULT_RATING, 2),
        ]
    ),
    N_HISTORIES = 1000,
    LEN_HIST = 100,
    random_seed=12345,
):
    assert CURR_RATING in RATINGS, f"Current rating ({CURR_RATING}) must be in RATINGS dictionary."
    assert DEFAULT_RATING in RATINGS, f"Default grade ({DEFAULT_RATING}) must be in RATINGS dictionary."

    # seed random number generator
    seed(random_seed)
    # print(p__)
    # print(p__.shape)

    np.set_printoptions(precision=3, suppress=True)

    # SAMPLING FROM THE MARKOV CHAIN
    # Simulate how long it takes a firm to default starting with some current rating

    histories = np.zeros((N_HISTORIES, LEN_HIST), np.int8)
    histories[:, 0] = RATINGS[CURR_RATING]

    # print(histories)
    randarray = rand(N_HISTORIES, LEN_HIST)
    # print(randarray)

    default_time = np.zeros(N_HISTORIES)
    default_sum = 0

    for i in range(0, N_HISTORIES):
        for j in range(1, LEN_HIST):
            for r in RATINGS:
                if randarray[i, j] < np.cumsum(p__[histories[i, j - 1], :])[RATINGS[r]]:
                    histories[i, j] = RATINGS[r]
                    break
            if histories[i, j] == RATINGS[DEFAULT_RATING]:
                break
        # Compute the average time to default
        if np.max(histories[i, :]) == RATINGS[DEFAULT_RATING]:
            where_default = np.where((histories[i, :] == RATINGS[DEFAULT_RATING]))
            default_time[i] = where_default[0][0]
            default_sum += 1
        else:
            default_time[i] = 0.0

    print("Default time:", np.sum(default_time) / default_sum)


def hmm_backward_onestep(
    xi_prob_t_t=np.array([[0.2, 0.6, 0.2]]),
    xi_prob_t1_t=np.array([[0.3, 0.4, 0.3]]),
    xi_prob_t1_T=np.array([[0.25, 0.5, 0.25]]),
    p_transition=np.array([[0.5,0.25,0.25], [0.25,0.5,0.25], [0.25,0.25,0.5]]),
):
    """Get xi_tT from Kim smoother for one step."""
    d_states=xi_prob_t1_t.shape[1]

    xi_prob_t_T = np.zeros((1,d_states), dtype=float)
    for i in range(0,1):
        for j in range(d_states):
            numer = p_transition[j]
            denom = xi_prob_t1_T/xi_prob_t1_t
            xi_prob_t_T[i,j] = xi_prob_t_t[i, j] * np.dot(denom[i], numer)

    return xi_prob_t_T


def compute_expected_value(p__, n_steps_transition, s_t, values):
    p2__ = np.linalg.matrix_power(p__, n_steps_transition)
    expected_t_0 = p2__[s_t - 1].T @ values
    return p2__, expected_t_0


def coin_toss(
        target_purse = 5,
        init_purse = 1,
        stop_purse=0,
        n_samples = 10_000,  # number of histories or simulations
        t_steps = 100,  # Length of each simulation
        random_seed=12345,  # Seed for reproducibility
):
    # seed random number generator
    assert target_purse > init_purse > stop_purse, "Target purse must be greater than initial purse and stop purse."
    seed(random_seed)

    N_STATES = target_purse + 1

    # S = np.zeros((N_STATES, 1))
    P = np.zeros((N_STATES, N_STATES))
    P[stop_purse, stop_purse] = 1.0
    P[-1, -1] = 1.0

    idx = np.arange(stop_purse+1, N_STATES - 1)
    P[idx, idx - 1] = 0.5
    P[idx, idx + 1] = 0.5

    # print("Transition matrix:\n", P)
    histories = np.zeros((n_samples, t_steps))
    histories[:, 0] = init_purse * np.ones(n_samples)
    randarray = rand(n_samples, t_steps)

    for i in range(0, n_samples):
        increments = np.where(randarray[i, 1:] >= 0.5, 1, -1)
        histories[i] = np.clip(
            np.cumsum(np.insert(increments, 0, histories[i, 0])),
            0, target_purse
        )
        stop_idx = np.where((histories[i] == target_purse) | (histories[i] == stop_purse))[0]
        if stop_idx.size > 0:
            stopped_ind = stop_idx[0]
            stopped_val = histories[i, stopped_ind]
            histories[i, stopped_ind:] = stopped_val

    end_gambles = {'target': [], 'stop': []}

    for i in range(0, n_samples):
        if np.max(histories[i, :]) >= target_purse:
            where_gamble_ends_T = np.where((histories[i, :] == target_purse))
            end_gambles["target"].append(where_gamble_ends_T[0][0])
        elif np.min(histories[i, :]) <= stop_purse:
            where_gamble_ends_0 = np.where((histories[i, :] < 1))
            end_gambles["stop"].append(where_gamble_ends_0[0][0])


    print(
        "Probability of getting the target:",
        np.sum(np.max(histories, axis=1) == target_purse) / n_samples,
        "\nProbability of losing all the money:",
        np.sum(np.min(histories, axis=1) < 1) / n_samples,
    )
    print(
        "Expected time until reaching a target result:",
        np.sum(end_gambles["target"]) / len(end_gambles["target"]),
        "\nExpected time until reaching a stop-loss result:",
        np.sum(end_gambles["stop"]) / len(end_gambles["stop"]),
        "\nExpected time until reaching either target/stop-loss result:",
        (np.sum(end_gambles["target"]) + np.sum(end_gambles["stop"])) / (len(end_gambles["target"]) + len(end_gambles["stop"])),
        "\nTotal number of simulations reached to end:",
        len(end_gambles["target"]) + len(end_gambles["stop"]),
    )


def phi_stationary(p__ = np.array([[0.75, 0.25], [1.0, 0.0]])):
    """
    Compute the stationary distribution of a Markov chain given its transition matrix.
    """
    assert p__.shape[0] == p__.shape[1], "Transition matrix must be square."

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(p__.T)

    with np.errstate(divide='ignore'):#, invalid='ignore'):
        stationary_phis = eigenvectors / np.sum(eigenvectors, axis=0)
    return stationary_phis # collection of stationary distributions as column vectors


def sample_markov_chain(
        # Transition matrix P (states 0,1,2 correspond to 1,2,3)
        p__ = np.array([
            [0.5, 0.4, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.4, 0.5]
        ]),
        n_samples = 10000,
        random_seed=12345,
        initial_state=0  # Start from state 0 (state 1 in human indexing)
):
    # Set seed for reproducibility
    np.random.seed(random_seed)
    counts = np.zeros_like(p__, dtype=int)
    d_states = p__.shape[1]

    # Simulate n_steps transitions
    for _ in range(n_samples):
        next_state = np.random.choice(a=np.arange(d_states), p=p__[initial_state])
        counts[initial_state, next_state] += 1
        initial_state = next_state

    # Display the counts matrix
    return counts


def hmm_forward_onestep(
        p__=np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]),
        xi_t_t=np.array([[0.2, 0.6, 0.2]]),
):
    """
    Compute the forward probabilities and xi_t1_t for a given HMM.
    """
    # n = xi_t_t.shape[1]
    # xi_t1_t = np.zeros((1, n))
    # for j in range(n):
    #     # sum over i: xi_t_t[i] * P[i,j]
    #     xi_t1_t[0, j] = xi_t_t[0, :] @ p__[:, j]
    #
    # return xi_t1_t
    return xi_t_t @ p__  # simple row-vector multiplication


def run_practice_1():
    print("Q1)")
    print(
        hmm_smoother_params(
            y_=np.array([0.4, -0.625]),
            xi_1_2=np.array([0.55, 0.45]),
            xi_2_2=np.array([0.3, 0.7])
        )
    )

    print("Q3)")
    bond_forward_price(
        p__=_credit_rating_counts_to_p__()
    )

    print("Q4)")
    print(
        HmmFilter.forward_alg(
            pi0=np.array([0.2,0.8]),
            d_states=2,
            t_sample=1,
            p_transition=np.array([[0.75, 0.25],[0.25, 0.75]]),
            mu=np.array([-1,1]),
            sigma=1.,
            Y=np.array([-0.5])
        )
    )

    print("Q5)")
    print(
        hmm_smoother_params(
            y_= np.array([0.4, -0.5]),
            xi_1_2=np.array([0.55, 0.45]),
            xi_2_2=np.array([0.3, 0.7])
        )
    )

    print("Q6)")
    credit_default_time(
        p__=np.array([[0.9, 0.09, 0.01], [0.2, 0.75, 0.05], [0, 0, 1]]),
        CURR_RATING="AAA",
        RATINGS=dict(
            [
                ("AAA", 0),
                ("AA", 1),
                (DEFAULT_RATING, 2),
            ]
        ),

        N_HISTORIES=1000,
        LEN_HIST=100,
        random_seed=12345,
    )

    print("Q7)")
    rouwenhurst_method(rho=0.5, sigma=1.0, n_grid=3)

    print("Q9)")
    print(
        hmm_backward_onestep(
            xi_prob_t_t=np.array([[0.2, 0.6, 0.2]]),
            xi_prob_t1_t=np.array([[0.3, 0.4, 0.3]]),
            xi_prob_t1_T=np.array([[0.25, 0.5, 0.25]]),
            p_transition=np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]),
        )
    )

    print("Q10)")
    print(compute_expected_value(
        p__=np.array([[0.75, 0.25], [1, 0]]),
        n_steps_transition=2,
        s_t=2,
        values=np.array([1, 2])
    ))

    print("Q11)")
    # Should be solved analytically, but here I leave a simulation
    coin_toss(
        target_purse=100,
        init_purse=20,
        n_samples=100_000,  # number of histories or simulations
        t_steps=10_000,  # Length of each simulation
        random_seed=12345,  # Seed for reproducibility
    )

    print("Q13)")
    print(
        hmm_smoother_params(
            y_=np.array([0.4, -0.5]),
            xi_1_2=np.array([0.55, 0.45]),
            xi_2_2=np.array([0.4, 0.6])
        )
    )

    print("Q14)")
    print(
        phi_stationary(
            p__=np.array([[0.75, 0.25], [1.0, 0.0]])
        )
    )

    print("Q15)")
    coin_toss(
        target_purse=7,
        init_purse=3,
        stop_purse=0,
        n_samples=5_000,  # number of histories or simulations
        t_steps=200,  # Length of each simulation
        random_seed=12345,  # Seed for reproducibility
    )

    print("Q17)")
    print(
        hmm_backward_onestep(
            xi_prob_t_t=np.array([[0.1, 0.8, 0.1]]),
            xi_prob_t1_t=np.array([[0.275, 0.45, 0.275]]),
            xi_prob_t1_T=np.array([[0.25, 0.5, 0.25]]),
            p_transition=np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]),
        )
    )

    print("Q18)")
    print(sample_markov_chain(
        p__=np.array([
            [0.5, 0.4, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.4, 0.5]
        ]),
        n_samples=10000,
        random_seed=12345,
        initial_state=0  # Start from state 0 (state 1 in human indexing)
    ))

    print("Q19)")
    tauchen_method(rho=0.5, sigma=0.1, lambd=2., n_grid=3)

    print("Q20)")
    print(hmm_forward_onestep(
        p__ = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]),
        xi_t_t= np.array([[0.2, 0.4, 0.4]]),
    ))


def run_practice_graded():
    print("Q2)")
    print(
        HmmFilter.forward_alg(
            pi0=np.array([0.2, 0.8]),
            d_states=2,
            t_sample=1,
            p_transition=np.array([[0.75, 0.25], [0.25, 0.75]]),
            mu=np.array([-2, 2]),
            sigma=np.array([1., 1.]),
            Y=np.array([-1.5])
        )
    )

    print("Q3)")
    bond_forward_price(
        p__=_credit_rating_counts_to_p__(),
        CURR_RATING="B",
        FACE_VALUE=100.0,
        COUPON=4.0,
        D_RECOVERY=50.0,
    )

    print("Q4)")
    print(
            hmm_smoother_params(
            y_=np.array([1, -0.5]),
            xi_1_2=np.array([0.35, 0.65]),
            xi_2_2=np.array([0.75, 0.25])
        )
    )

    print("Q5)")
    print( phi_stationary(p__=np.array([[0.5, 0.5], [0.1, 0.9]])) )

    print("Q6)")
    HmmFilter.log_likelihood(
        p__=np.array([[0.8, 0.2], [0.2, 0.8]]),
        mu=np.array([-1., 1.]),
        sigma=np.array([0.8, 0.8]),
        pi0=np.array([0.2, 0.8]),
        y_input=np.array([-0.85, 0.4, -0.2]),
    )  # assert almost equal to -5.879

    HmmFilter.log_likelihood(
        p__=np.array([[0.75, 0.25], [0.25, 0.75]]),
        mu=np.array([-2., 2.]),
        sigma=np.array([1., 1.]),
        pi0=np.array([0.2, 0.8]),
        y_input=np.array([0.25, -0.3, 1.5]),
    )

    print("Q7)")
    rouwenhurst_method(
        rho=0.9,
        sigma=0.2,
        n_grid=3
    )

    print("Q8)")
    print(hmm_backward_onestep(
        xi_prob_t_t=np.array([[0.2, 0.6, 0.2]]),
        xi_prob_t1_t=np.array([[0.3, 0.4, 0.3]]),
        xi_prob_t1_T=np.array([[0.15, 0.7, 0.15]]),
        p_transition=np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]),
    ))

    print("Q10)")
    HmmFilter.log_likelihood(
        p__=np.array([[0.8, 0.2], [0.3, 0.7]]),
        mu=np.array([-3., 3.]),
        sigma=np.array([1., 1.]),
        pi0=np.array([0.1, 0.9]),
        y_input=np.array([1.5, 2]),
    )  # assert almost equal to -5.879

    print("Q11)")
    bond_forward_price(
        p__=_credit_rating_counts_to_p__(),
        CURR_RATING="AAA",
        FACE_VALUE=100.0,
        COUPON=2.0,
        D_RECOVERY=75.2,
    )

    print("Q12)")
    print(hmm_smoother_params(
        y_=np.array([2., -0.5]),
        xi_1_2=np.array([0.35, 0.65]),
        xi_2_2=np.array([0.75, 0.25])
    ))

    print("Q15)")
    coin_toss(
        target_purse=7,
        init_purse=3,
        stop_purse=0,
        n_samples=5_000,  # number of histories or simulations
        t_steps=200,  # Length of each simulation
        random_seed=12345,  # Seed for reproducibility
    )

    pass


if __name__ == "__main__":
    run_practice_graded()