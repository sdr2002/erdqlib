from matplotlib import pyplot as plt
from numpy.random import seed, rand
import numpy as np
from scipy.stats import norm


def tauchen_method(rho, sigma, lambd, n_grid):
    start_tauchen = -lambd * sigma / (1 - rho ** 2) ** 0.5
    end_tauchen = -start_tauchen

    zgrid_ = np.linspace(start_tauchen, end_tauchen, n_grid)
    zmid_points = (zgrid_[1:] + zgrid_[:-1]) / 2

    p__ = np.zeros((n_grid, n_grid))
    # P_i,1
    p__[:, 0] = norm.cdf((zmid_points[0] - rho * zgrid_) / sigma)
    # P_i,N_GRID
    p__[:, -1] = 1.0 - norm.cdf((zmid_points[-1] - rho * zgrid_) / sigma)

    # P_i,j for j in [2, N_GRID - 1]
    for i in range(0, n_grid):
        for j in range(1, n_grid - 1):
            lhs = norm.cdf((zmid_points[j] - rho * zgrid_[i]) / sigma)
            rhs = norm.cdf((zmid_points[j - 1] - rho * zgrid_[i]) / sigma)
            p__[i, j] = lhs - rhs

    print(f"p__:\n{p__}")
    print(f"zgrid_:\n{zgrid_}")

    return p__, zgrid_


def rouwenhurst_method(rho, sigma, n_grid):
    p_rouwen = (1 + rho) * 0.5  # chosen to be = pi
    q_rouwen = p_rouwen
    start_rouwen = -(((n_grid - 1) / (1 - rho ** 2)) ** 0.5) * sigma
    end_rouwen = -start_rouwen
    zgrid_ = np.linspace(start_rouwen, end_rouwen, n_grid)
    p__ = np.append(
        [[p_rouwen, 1.0 - p_rouwen]], [[1 - q_rouwen, q_rouwen]], axis=0
    )

    for i in range(2, n_grid):
        ''' Padding zero for one row and one column
        m1 = 
        | P     0_.T |
        | 0_    0    |
        m2 =
        | 0_.T  P_.T |
        | 0     0_   |
        m3 =
        | 0_    0    |
        | P     0_.T |
        m4 =
        | 0     0_   |
        | 0_.T  P    |
        '''
        m1 = np.append(p__, np.zeros((i, 1)), axis=1)
        m1 = np.append(m1, np.zeros((1, i + 1)), axis=0)
        m2 = np.append(np.zeros((i, 1)), p__, axis=1)
        m2 = np.append(m2, np.zeros((1, i + 1)), axis=0)
        m3 = np.append(p__, np.zeros((i, 1)), axis=1)
        m3 = np.append(np.zeros((1, i + 1)), m3, axis=0)
        m4 = np.append(np.zeros((i, 1)), p__, axis=1)
        m4 = np.append(np.zeros((1, i + 1)), m4, axis=0)

        p__ = p_rouwen * m1 + (1 - p_rouwen) * m2 + (1 - q_rouwen) * m3 + q_rouwen * m4
        # Mormalize the rows to make the sum over columns to 1
        p__[1:i, :] = 0.5 * p__[1:i, :]

    print(f"p__:\n{p__}")
    print(f"zgrid_:\n{zgrid_}")

    return p__, zgrid_


def ex_unconditional(
        RHO=0.975,
        SIGMA=0.1,
        N_GRID=9,
        LAMBDA=2.0,  # Used in Tauchen method
        N_TRANSITION=100
):
    transition_tauchen, zgrid_tauchen = tauchen_method(RHO, SIGMA, LAMBDA, N_GRID)
    transition_rouwen, zgrid_rouwen = rouwenhurst_method(RHO, SIGMA, N_GRID)

    # Find the stationary distributions by iteration
    p_stationary_tauchen = np.ones((N_GRID, 1)) / N_GRID
    p_stationary_rouwen = np.ones((N_GRID, 1)) / N_GRID
    # for t in range(1, N_TRANSITION):
    p_stationary_tauchen = np.linalg.matrix_power(transition_tauchen, N_TRANSITION-1).T @ p_stationary_tauchen
    p_stationary_rouwen = np.linalg.matrix_power(transition_rouwen, N_TRANSITION-1).T @ p_stationary_rouwen

    print(p_stationary_tauchen)
    print(p_stationary_rouwen)

    # Check if unconditional moments match
    tauchen_mean_stationary = np.dot(p_stationary_tauchen.T, zgrid_tauchen) / N_GRID
    tauchen_sd_stationary = np.sqrt(np.dot(p_stationary_tauchen.T, (zgrid_tauchen - tauchen_mean_stationary) ** 2))

    rouwen_mean_stationary = np.dot(p_stationary_rouwen.T, zgrid_rouwen) / N_GRID
    rouwen_sd_stationary = np.sqrt(np.dot(p_stationary_rouwen.T, (zgrid_rouwen - rouwen_mean_stationary) ** 2))

    print(
        "Checking the unconditional mean....",
        "Tauchen Mean:",
        tauchen_mean_stationary,
        "Rouwen Mean:",
        rouwen_mean_stationary,
        "Mean:",
        0,
    )
    print(
        "Checking the unconditional sd....",
        "Tauchen sd:",
        tauchen_sd_stationary,
        "Rouwen sd:",
        rouwen_sd_stationary,
        "sd:",
        SIGMA / (1 - RHO ** 2) ** 0.5,
    )

    return transition_tauchen, zgrid_tauchen, transition_rouwen, zgrid_rouwen, N_GRID


def ex_sample_paths(
        P_tauchen, zgrid_tauchen, P_rouwen, zgrid_rouwen, N_GRID,
        i_init=1, random_seed=12345, LEN_HIST=10000
):
    # Monte Carlo simulations to compare performance
    # seed random number generator
    seed(random_seed)

    histories_tauchen_ind = np.zeros((LEN_HIST), np.int8)
    histories_tauchen_z = np.zeros((LEN_HIST))
    histories_tauchen_ind[0] = i_init
    histories_tauchen_z[0] = zgrid_tauchen[histories_tauchen_ind[0]]

    histories_rouwen_ind = np.zeros((LEN_HIST), np.int8)
    histories_rouwen_z = np.zeros((LEN_HIST))
    histories_rouwen_ind[0] = i_init
    histories_rouwen_z[0] = zgrid_rouwen[histories_rouwen_ind[0]]

    randarray = rand(LEN_HIST)

    # for j in range(1, LEN_HIST):
    #     for r in range(0, N_GRID):
    #         if randarray[j] < np.cumsum(P_tauchen[histories_tauchen_ind[j - 1], :])[r]:
    #             histories_tauchen_z[j] = zgrid_tauchen[r]
    #             histories_tauchen_ind[j] = r
    #             break
    #     for r in range(0, N_GRID):
    #         if randarray[j] < np.cumsum(P_rouwen[histories_rouwen_ind[j - 1], :])[r]:
    #             histories_rouwen_z[j] = zgrid_rouwen[r]
    #             histories_rouwen_ind[j] = r
    #             break

    # Precompute cumulative transition matrices
    cum_P_tauchen = np.cumsum(P_tauchen, axis=1)
    cum_P_rouwen = np.cumsum(P_rouwen, axis=1)
    for j in range(1, LEN_HIST):
        prev_tauchen = histories_tauchen_ind[j - 1]
        prev_rouwen = histories_rouwen_ind[j - 1]
        # Find next state index using searchsorted
        next_tauchen = np.searchsorted(cum_P_tauchen[prev_tauchen], randarray[j])
        next_rouwen = np.searchsorted(cum_P_rouwen[prev_rouwen], randarray[j])
        histories_tauchen_ind[j] = next_tauchen
        histories_tauchen_z[j] = zgrid_tauchen[next_tauchen]
        histories_rouwen_ind[j] = next_rouwen
        histories_rouwen_z[j] = zgrid_rouwen[next_rouwen]

    plt.subplot(1, 2, 1)
    plt.plot(histories_tauchen_z[:])
    plt.plot(histories_rouwen_z[:])
    plt.title("Tauchen vs. Rouwenhorst")

    fig = plt.gcf()
    fig.set_size_inches(16, 5)
    plt.show()

    return histories_tauchen_z, histories_rouwen_z


def ex_sampled_stat(
        histories_tauchen_z, histories_rouwen_z,
        RHO, SIGMA, LEN_HIST=10000, T_EXCLUDE = 100
):
    # Compute mean, variance, and autocorrelation coefficient of both series
    # Exclude the initial periods (Burn-in)
    tauchen_mean = np.mean(histories_tauchen_z[T_EXCLUDE:LEN_HIST])
    rouwen_mean = np.mean(histories_rouwen_z[T_EXCLUDE:LEN_HIST])

    print("Real mean:", 0, "Tauchen mean:", tauchen_mean, "Rouwen mean:", rouwen_mean)

    tauchen_sd = np.std(histories_tauchen_z[T_EXCLUDE:LEN_HIST])
    rouwen_sd = np.std(histories_rouwen_z[T_EXCLUDE:LEN_HIST])

    print(
        "Real sd:",
        SIGMA / (1 - RHO ** 2) ** 0.5,
        "Tauchen sd:",
        tauchen_sd,
        "Rouwen_sd:",
        rouwen_sd,
    )

    tauchen_cov = np.cov(
        histories_tauchen_z[T_EXCLUDE: LEN_HIST - 1],  # noQA E203
        histories_tauchen_z[T_EXCLUDE + 1: LEN_HIST],  # noQA E203
    )
    rouwen_cov = np.cov(
        histories_rouwen_z[T_EXCLUDE: LEN_HIST - 1],  # noQA E203
        histories_rouwen_z[T_EXCLUDE + 1: LEN_HIST],  # noQA E203
    )

    tauchen_rho = tauchen_cov[0, 1] / tauchen_cov[0, 0]
    rouwen_rho = rouwen_cov[0, 1] / rouwen_cov[0, 0]

    print("Real rho:", RHO, "Tauchen rho:", tauchen_rho, "Rouwen rho:", rouwen_rho)


if __name__ == "__main__":
    P_tauchen, zgrid_tauchen, P_rouwen, zgrid_rouwen, N_GRID = ex_unconditional(
        RHO=0.975, SIGMA=0.1, N_GRID=9, LAMBDA=2.0, N_TRANSITION=100
    )
    histories_tauchen_z, histories_rouwen_z = ex_sample_paths(
        P_tauchen, zgrid_tauchen, P_rouwen, zgrid_rouwen, N_GRID,
        i_init=1, random_seed=12345, LEN_HIST=10000
    )
    ex_sampled_stat(
        histories_tauchen_z, histories_rouwen_z,
        RHO=0.975, SIGMA=0.1, LEN_HIST=10000
    )
