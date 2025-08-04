r"""
## **1. Hidden Markov Models**

A Hidden Markov Model (HMM) is a Markov process that is split into two components: an observable component and an unobservable or "hidden" component that follows a Markov process. HMMs naturally describe setups where a stochastic system is observed through noisy measurements, for instance stock prices, that are affected by an unobserved economic factor. However, this type of modelization is of broad application in many fields, from speech recognition to thermodynamics.

A basic HMM contains the following components:
* A set of $N$ states $\mathcal{S}=\{s_1,...,s_N\}$
* A transition probability matrix $P$
* A sequence of $T$, possibly vector-valued, observations $\mathcal{Y}_T=\{y_1,...,y_T\}$
* A sequence of observation marginal likelihoods $f(y_t|s_t=i)$ for each $i=1,..,N$.
* An initial probability distribution $\pi=\{\pi_1,...,\pi_N\}$.

An important assumption is that the hidden Markov process is independent of past observations $\mathcal{Y}_{t-1}$, i.e., $\mathbb{P}\{s_t=j|s_t=i,\mathcal{Y}_{t-1}\}=\mathbb{P}\{s_t=j|s_t=i\}=p_{ij}$.

This type of setup is also labeled in some instances as a Markov regime-switching model, where the observable output $y_t$ features a marginal distribution whose parameters change with the realization of an unobservable state. Let's see with an example an application of Hidden Markov Process.

Suppose that $y_t=\mu_t+\varepsilon_t$, where $\mu_t = \mu_j$ if $s_t=j$, $\mathcal{S}=\{1,...N\}$, $S_t$ is Markovian, and $\varepsilon_t$ is i.i.d. $N(0,\sigma^2)$.

We are interested in making inference about the probability of being in each state $s_i$ at each date $t$, as well as estimating the model parameters of the transition matrix $P$ and the vector $(\mu_1,...,\mu_N,\sigma)$. We collect all the parameters in a vector $\theta$ and let's assume for now that we know its value with certainty.

Let's build the log-likelihood function of the process:
$$
\begin{align}
\mathcal{L}(\theta) = \sum_{t=1}^T log\ f(y_t|\mathcal{Y}_{t-1};\theta)
\end{align}
$$
where
$$
\begin{align}
f(y_t|\mathcal{Y}_{t-1};\theta) = \sum_{i=1}^N \mathbb{P}(s_t=i|\mathcal{Y}_{t-1};\theta) \times \phi_{t}(i)
\end{align}
$$
where $\phi_{t}(i)=\phi\left(\frac{y_t-\mu_i}{\sigma}\right)$, and $\phi(.)$ denotes the standard normal probability density function. The main problem arises from the evaluation of $\mathbb{P}(s_t=i|\mathcal{Y}_{t-1};\theta)$, which would usually require an overwhelming amount of computations for each potential path $\mathcal{Y}_{t-1}$ and for each period $t$. This rules out estimating directly $\theta$ by Maximum Likelihood.

One potential algorithm to reach a solution is to denote $\xi_{t|t}(j)=\mathbb{P}(s_t=j|\mathcal{Y}_{t};\theta)$. To estimate the model, we can rely on a recursive algorithm that sets the optimal forecasts $\xi_{t+1|t}(j)=\mathbb{P}(s_{t+1}=j|\mathcal{Y}_{t};\theta)$:
$$
\begin{align}
& \xi_{t|t}(i) = \frac{\xi_{t|t-1}(i)\phi_{t}(i)}{f(y_{t}|\mathcal{Y}_{t-1};\theta)} \\
& \xi_{t+1|t}(j) = \sum_{i=1}^N p_{ij} \xi_{t|t}(i)
\end{align}
$$
This procedure is named the Hamilton filter, developed by Hamilton (1990). To initialize the recursion, we can set $\xi_{1|0}(i)=1/N$ or, if we have some guess for the initial distribution, $\xi_{1|0}(i)=\pi_i$. Another alternative is to include it in the vector of parameters to estimate, under the usual constraints for a probability distribution.

Then, to estimate the model, we would set initial parameters $\theta^0$ that would allow us to recover each $\xi_{t+1|t}(j)$ and evaluate the log-likelihood function, then iterate with new guess $\theta^1$ and so on until convergence.

Lastly, notice that we can make forecasts on the observable process $y_{t+1}$ by exploiting the expressions for $\xi_{t+1|t}(j)$ once we have estimated the model parameters.

We are going to bring to practice this estimation by simulating a time series governed by a hidden Markov process. We first build a simulated time series and then estimate the model by maximum likelihood, applying the Hamilton filter within the evaluation step. To keep things manageable, we are going to assume that the value of $\sigma$ is known. Therefore, we just need to estimate the vector $\theta=\{\mu_1,\mu_2,p_{11},p_{22}\}$.
"""
import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from typing import Tuple, List, Optional

import pandas as pd
from matplotlib import pyplot as plt

import numpy as np
import scipy.optimize as sco
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from scipy.stats import norm

from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)


def check_transition_matrix(p_transition: NDArray[Tuple[float, float]], d_states: int):
    """Check the validity of the transition matrix P."""
    if p_transition.shape != (d_states, d_states):
        raise ValueError(f"Transition matrix P must be {d_states}x{d_states}, got {p_transition.shape}")
    if not np.all(np.sum(p_transition, axis=1) != 0):
        raise ValueError("Transition matrix P must have rows summing to 1.")


def create_default_transition_matrix(p_diags: NDArray[Tuple[float]]) -> NDArray[Tuple[float, float]]:
    """
    Construct the transition probability matrix P from parameter vector theta.

    Args:
        p_diags: 1D array of transition probabilities to self (diagonal elements)

    Returns:
        NxN transition probability matrix P
    """
    if p_diags.ndim != 1:
        raise ValueError(f"p_selves must be a 1D array, got shape {p_diags.shape}")
    n_states: int = len(p_diags)
    remainders = (1.0 - p_diags) / (n_states - 1)
    p_transition: NDArray[Tuple[float, float]] = np.full(
        shape=(n_states, n_states),
        fill_value=remainders,
        dtype=float
    ).T
    p_transition -= np.diag(v=remainders)
    p_transition += np.diag(v=p_diags)

    return p_transition


def get_parameters_search_array(
        p_transition: NDArray[Tuple[float, float]], mean_values: NDArray[Tuple[float]]
) -> NDArray[Tuple[float]]:
    """Convert transition 2D-array P and 1D-array means into a single parameter array."""
    # Remove the last column of P (last state transitions to itself) as the column sum must be 1
    return np.concatenate([p_transition[:, :-1].reshape(-1), mean_values])


@dataclass
class HMMPath:
    """Data class to hold the path of a Hidden Markov Model."""
    states_st: NDArray[Tuple[int]]
    states_vt: NDArray[Tuple[float]]
    y_data: NDArray[Tuple[float]]

    def __post_init__(self):
        if self.y_data is None:
            raise ValueError("y_data cannot be None.")

        if not ((self.states_st is None) and (self.states_vt is None)):
            if not (not(self.states_st is None) and not(self.states_vt is None)):
                raise ValueError("Both states_st and states_vt must be provided or both must be None.")

            if not (self.states_st.ndim == 1 and self.states_vt.ndim == 1 and self.y_data.ndim == 1):
                raise ValueError("All arrays must be 1D.")
            if len(self.states_st) != len(self.states_vt):
                raise ValueError("State arrays must have the same length.")
            if len(self.states_st) <= len(self.y_data):
                raise ValueError("Length of state arrays must be greater than length of observations due to BURNIN period.")


@dataclass
class HMMParameters:
    mu: NDArray[Tuple[float]]
    sigma: NDArray[Tuple[float]]
    p_transition: NDArray[Tuple[float, float]]
    pi: NDArray[Tuple[float]]

    def calculate_differential(self, new_: "HMMParameters", eps: float=1e-2) -> NDArray[Tuple[float]]:
        """Calculate the difference ratios between current and previously estimated parameter"""
        diff: NDArray[Tuple[float]] = np.zeros((4), dtype=float)
        diff[0] = np.sum(np.absolute(new_.mu - self.mu)) / (np.min(self.mu) + eps * 1e-2)
        diff[1] = np.sum(np.absolute(new_.sigma - self.sigma)) / (np.min(self.sigma) + eps * 1e-2)
        diff[2] = np.sum(np.absolute(np.subtract(new_.p_transition, self.p_transition))) / (np.min(self.p_transition) + eps * 1e-2)
        diff[3] = np.sum(np.absolute(new_.pi - self.pi)) / (np.min(self.pi) + eps * 1e-2)
        return diff

    def copy(self) -> "HMMParameters":
        """Create a copy of the HMMParameters instance."""
        return HMMParameters(
            mu=np.copy(self.mu),
            sigma=np.copy(self.sigma),
            p_transition=np.copy(self.p_transition),
            pi=np.copy(self.pi)
        )

    def log_iter(self, logger, ite: int = None, diffs_: NDArray[Tuple[float]] = None):
        """Log the current iteration and parameter differences."""
        head: str = f"Iter[{ite}]" if ite is not None else "Initial"
        msg = (
            f"{head}: mu={np.array2string(self.mu, precision=3, separator=',')}, "
            f"sigma={np.array2string(self.sigma, precision=3, separator=',')}, "
            f"p={np.array2string(self.p_transition, precision=3, separator=',', prefix='', suffix='').replace('\n', ';')}, "
            f"pi0={np.array2string(self.pi, precision=3, separator=',', suppress_small=False)}, "
        )
        if diffs_ is not None:
            msg += f"diff[mu_,sigma_,p__,pi0_]={np.array2string(diffs_, precision=3, separator=',')}"

        logger.info(msg)

    def get_d_states(self) -> int:
        return len(self.mu)

    def check_parameter_shapes(self):
        d_states: int = self.get_d_states()
        if self.p_transition.shape != (d_states, d_states):
            raise ValueError(f"p_transition must be a {d_states}x{d_states} matrix, got {self.p_transition.shape}")
        if self.mu.shape != (d_states,):
            raise ValueError(f"mu must be a 1D array of length {d_states}, got {self.mu.shape}")
        if self.sigma.shape != (d_states,):
            raise ValueError(f"sigma must be a 1D array of length {d_states}, got {self.sigma.shape}")
        if self.pi.shape != (d_states,):
            raise ValueError(f"pi must be a 1D array of length {d_states}, got {self.pi.shape}")


class HmmFilter:
    """Hidden Markov Model implementation supporting N-state space."""

    @staticmethod
    def generate_markov_sequence(
            P: NDArray[Tuple[float, float]],
            state_arr: List[float],
            len_path: int,
            sigma: float,
            len_burnin: int,
            state_initial: int = 0,
            seed_val: int = 12345,
    ) -> HMMPath:
        """
        Generate a sample path from an N-state Markov chain with Gaussian noise observations.
        """
        np.random.seed(seed_val)

        states_st: NDArray[Tuple[int]] = np.zeros(len_path, dtype=int)
        states_vt: NDArray[Tuple[float]] = np.zeros(len_path, dtype=float)
        T_SAMPLE: int = len_path - len_burnin
        Yarray: NDArray[Tuple[float]] = np.zeros(T_SAMPLE, dtype=float)

        # Set first state
        states_st[0] = state_initial
        states_vt[0] = state_arr[state_initial]

        # Generate random values
        randarray = np.random.rand(len_path)
        rnandarray = np.random.normal(0.0, sigma, size=len_path)

        # Generate states following Markov property - P(s_t|s_{t-1}) = p_{ij} - for N-state case
        for tt in range(1, len_path):
            '''
            original implementation was to compare diagonal element of P with random value to choose go off-the-diagonal
              each time, which only works for 2-state case.
            '''
            index_prev = states_st[tt - 1]
            # Given P[index_prev] = [p₀, p₁, ..., pₙ], where pᵢ is the probability of transitioning from state index_prev to state i
            # computes the cumulative distribution function (CDF) for transitions from the current state:
            cumulative_probs = np.cumsum(P[index_prev])
            # implements the inverse CDF sampling
            next_state = np.searchsorted(cumulative_probs, randarray[tt])
            # assigns the sampled state
            states_st[tt] = next_state

            # Get μ_t based on current state
            states_vt[tt] = state_arr[states_st[tt]]

            # Generate observation y_t = μ_t + ε_t after burn-in period
            if tt >= len_burnin:
                Yarray[tt - len_burnin] = states_vt[tt] + rnandarray[tt]

        return HMMPath(states_st=states_st, states_vt=states_vt, y_data=Yarray)

    @classmethod
    def forward_alg(
            cls,
            pi0: NDArray[Tuple[float]],
            d_states: int,
            t_sample: int,
            p_transition: NDArray[Tuple[float, float]],
            mu: NDArray[Tuple[float]],
            sigma: float | NDArray[Tuple[float]],
            Y: NDArray[Tuple[float]]
    ) -> Tuple[NDArray[Tuple[float, float]], NDArray[Tuple[float, float]]]:
        """Forward algorithm (Hamilton filter) for N-state HMM."""
        if type(sigma) is float:
            sigma: NDArray[Tuple[float]] = np.full(shape=(d_states,), fill_value=sigma, dtype=float)
        if sigma.ndim > 1:
            raise ValueError(f"sigma must be a 1D array, got {sigma}")
        # Initialize arrays for filtered (ξ_{t|t}) and predicted (ξ_{t+1|t}) probabilities
        xi_prob_t: NDArray[Tuple[float, float]] = np.zeros((t_sample, d_states), dtype=float)
        xi_prob_t1: NDArray[Tuple[float, float]] = np.zeros((t_sample, d_states), dtype=float)

        # Forward recursion for t=0...T-1
        for tt in range(0, t_sample):
            pi_t: NDArray[Tuple[float]] = np.array(pi0, dtype=float) if tt == 0 else xi_prob_t1[tt - 1]

            # Calculate f(y_t|Y_{t-1}) = ∑_i ξ_{t|t-1}(i) × ϕ_t(i)
            likelihood_y: float = cls.likelihood(xi_prob=pi_t, mu=mu, sigma=sigma, y=float(Y[tt]))
            # Calculate normal PDFs ϕ_t(i) for each state
            phi: NDArray[Tuple[float]] = np.array(norm.pdf((Y[tt] - mu) / sigma))
            # Update ξ_{t|t}(i) = ξ_{t|t-1}(i) × ϕ_t(i) / f(y_t|Y_{t-1})
            xi_prob_t[tt] = np.multiply(pi_t, phi) / likelihood_y
            # Predict ξ_{t+1|t}(j) = ∑_i p_{ij} × ξ_{t|t}(i)
            xi_prob_t1[tt] = np.dot(xi_prob_t[tt], p_transition)

        return xi_prob_t, xi_prob_t1

    @staticmethod
    def get_p_transition_from_search_array(parameters_search: NDArray[Tuple[float]]) -> NDArray[Tuple[float, float]]:
        """
        Convert a search array into a transition matrix P.

        Args:
            parameters_search: 1D array containing transition probabilities and means in n_states*(n_states-1) + n_states format.

        Returns:
            Transition probability matrix P
        """
        if parameters_search.ndim != 1:
            raise ValueError(f"parameters_search must be a 1D array, got shape {parameters_search.shape}")
        n_states: int = int(np.sqrt(len(parameters_search)))
        if n_states ** 2 != len(parameters_search):
            raise ValueError(f"parameters_search length {len(parameters_search)} is not a perfect square for n_states.")
        p_transition: NDArray[Tuple[float, float]] = create_default_transition_matrix(
            p_diags=parameters_search[:n_states]
        )
        return p_transition

    @staticmethod
    def get_mu_from_search_array(parameters_search: NDArray[Tuple[float]]) -> NDArray[Tuple[float]]:
        """
        Extract the means from the search array.

        Args:
            parameters_search: 1D array containing transition probabilities and means in n_states*(n_states-1) + n_states format.

        Returns:
            1D array of means for each state.
        """
        if parameters_search.ndim != 1:
            raise ValueError(f"parameters_search must be a 1D array, got shape {parameters_search.shape}")
        n_states: int = int(np.sqrt(len(parameters_search)))
        if n_states ** 2 != len(parameters_search):
            raise ValueError(f"parameters_search length {len(parameters_search)} is not a perfect square for n_states.")
        return parameters_search[-n_states:]

    @staticmethod
    def likelihood(
            xi_prob: NDArray[Tuple[float] | Tuple[float, float]],
            mu: NDArray[Tuple[float]],
            sigma: float | NDArray[Tuple[float]],
            y: float | NDArray[Tuple[float]]
    ) -> float:
        """
        Calculate the likelihood of observing value y given state probabilities.

        Args:
            xi_prob: State probabilities
            mu: Mean values for each state
            sigma: Standard deviation of the noise
            y: Observed value

        Returns:
            Likelihood value
        """
        # Calculate ϕ_t(i) = ϕ((y_t-μ_i)/σ) for each state i
        phi = norm.pdf((y - mu) / sigma)

        # Compute f(y_t|ξ_t) = ∑_i ξ_{t|t-1}(i) × ϕ_t(i)
        y_like = np.dot(xi_prob, phi)

        return float(y_like)

    @classmethod
    def log_likelihood(
            cls,
            p__: NDArray[Tuple[float, float]],
            mu: NDArray[Tuple[float]],
            sigma: float | NDArray[Tuple[float]],
            pi0: NDArray[Tuple[float]],
            y_input: NDArray[Tuple[float]],
            mute: bool = False,
            ind_iter: List[int] = None
    ) -> float:
        """Calculate the negative log likelihood for the HMM parameters."""

        '''
        TODO: below may output 7.447319519091295 just like the lecture note but got 7.564122
        
        HMM.log_likelihood(
            parameters_in_search=np.concat(
                [
                    np.array([[0.75, 0.25], [0.25, 0.75]])[:,:-1].reshape(-1),
                    np.array([-2., 2.])
                ],
                axis=0
            ),
            pi0=np.array([0.2, 0.8]),
            sigma=1.,
            Y=np.array([0.25, -0.3, 1.5]),
        )
        '''
        # Extract transition matrix and means
        d_states: int = len(pi0)
        t_sample: int = len(y_input)
        expected_dim = (d_states, d_states)
        if p__.shape != expected_dim:
            raise ValueError(f"params.shape must be {expected_dim} but got {p__.shape}")
        if mu.shape != (d_states,):
            raise ValueError(f"mu must be a 1D array of length {d_states}, got {mu.shape}")
        if type(sigma) is float:
            sigma = np.full(shape=(d_states,), fill_value=sigma, dtype=float)
        if sigma.shape != (d_states,):
            raise ValueError(f"sigma must be a 1D array of length {d_states}, got {sigma.shape}")

        P: NDArray[Tuple[float, float]] = p__
        mu: NDArray[Tuple[float]] = mu

        # Run forward algorithm
        xi_prob_t, xi_prob_t1 = cls.forward_alg(
            pi0=np.array(pi0),
            d_states=len(pi0),
            t_sample=t_sample,
            p_transition=P,
            mu=mu,
            sigma=sigma,
            Y=y_input
        )

        # Calculate likelihood for each observation
        y_like = np.zeros(t_sample)
        for tt in range(t_sample):
            xi_prob = pi0 if tt == 0 else xi_prob_t1[tt - 1]

            y_like[tt] = cls.likelihood(
                xi_prob=xi_prob,
                mu=mu,
                sigma=sigma,
                y=y_input[tt]
            )

        sum_log_likelihood = np.sum(np.log(y_like))

        if mute:
            return sum_log_likelihood

        log_msg: str = f"Iter{ind_iter} - " \
                       f"Log-likelihood parameters: Diag(P)={np.diag(P)}, Mu={mu}, " \
                       f"sigma={sigma} | (-)LogLikelihood: {-sum_log_likelihood:.6f}"
        if ind_iter:
            if ind_iter[0] % 10 == 0:
                LOGGER.info(log_msg)
            ind_iter[0] += 1
        else:
            LOGGER.info(log_msg)

        return sum_log_likelihood

    @classmethod
    def optimise_filter(
            cls,
            pi0: NDArray[Tuple[float]],
            sigma: float,
            Y: NDArray[Tuple[float]],
            n_states: int = 2,
            bounds: List[Tuple[float, float]] = None,
            initial_guess: NDArray[Tuple[float]] = None,
            method: str = "trust-constr",
            tol: float = 1e-2,
            mute: bool = False
    ) -> sco.OptimizeResult:
        """Optimize the HMM parameters by minimizing negative log likelihood."""
        # Set up default bounds and initial guess based on number of states
        y_min: float = float(np.percentile(Y, 0.1))
        y_max: float = float(np.percentile(Y, 99.9))
        if bounds is None:
            # For N-state case: transition probs in [0,1] for every element except the last column
            #   in n_states x (n_states - 1) dimension
            p_bounds = [(0.0, 1.0)] * n_states * (n_states - 1)

            # For N-state case: means spread evenly between min and max of Y in n_states dimension
            mu_step = (y_max - y_min) / max(n_states - 1, 1)
            mu_bounds = []
            for i in range(n_states):
                center = y_min + i * mu_step
                mu_bounds.append((center - mu_step / 2, center + mu_step / 2))
            bounds = p_bounds + mu_bounds

        if initial_guess is None:
            # For N-state case: high diagonal probs, means spread out
            ''' Original implementation
            diag_probs = [0.7] * n_states
            means = np.linspace(-1.5, 1.5, n_states)
            initial_guess = np.concatenate([diag_probs, means])            
            '''
            p_non_diag: float = 0.3 / (n_states - 1)
            P: NDArray[Tuple[float, float]] = np.full(shape=(n_states, n_states), fill_value=p_non_diag, dtype=float)
            P -= np.diag(np.full(shape=n_states, fill_value=p_non_diag, dtype=float))
            P += np.diag(np.full(shape=n_states, fill_value=0.7, dtype=float))

            means: NDArray[Tuple[float]] = np.percentile(
                a=Y,
                q=np.linspace(start=0, stop=100, num=n_states + 2, dtype=float)[1:-1]
            ).astype(float)

            initial_guess = get_parameters_search_array(p_transition=P, mean_values=means)

        LOGGER.info(
            f"Starting optimization with {n_states} states ({n_states}x{n_states - 1} P & {n_states} means), "
            f"initial guess: {initial_guess}"
        )

        ind_iter: List[int] = [0]
        result = sco.minimize(
            fun=lambda params_on_search, sig_=sigma, pi0_=pi0, y_=Y: - cls.log_likelihood(
                p__=cls.get_p_transition_from_search_array(parameters_search=params_on_search),
                mu=cls.get_mu_from_search_array(parameters_search=params_on_search),
                sigma=sig_,
                pi0=pi0_,
                y_input=y_,
                mute=mute,
                ind_iter=ind_iter
            ),
            x0=initial_guess,
            method=method,
            bounds=bounds,
            tol=tol
        )

        if not mute:
            if not result.success:
                LOGGER.warning(f"Optimization failed: {result.message}")
            else:
                LOGGER.info(f"Optimization successful: {result.message}")

                # Extract and display results
                parameters_on_search: NDArray[Tuple[float]] = result.x
                P_hat = cls.get_p_transition_from_search_array(parameters_on_search)
                mu_hat = parameters_on_search[n_states:2 * n_states]

                LOGGER.info(f"Estimated P:\n{P_hat}")
                LOGGER.info(f"Estimated mu:\n{mu_hat}")

        return result

    @staticmethod
    def plot_histogram(Yarray: NDArray[Tuple[float]], bins: int = 50) -> None:
        """
        Plot a histogram of the observed sample data.

        Args:
            Yarray: Array of observed values
            bins: Number of bins for the histogram
        """
        # Visualize the empirical distribution of the observations {y_1,...,y_T}
        plt.hist(Yarray, bins, density=True)
        plt.title("Histogram of Observed Sample")
        plt.show()

    @staticmethod
    def plot_estimation(
            xi_prob_t: NDArray[Tuple[float] | Tuple[float, float]],
            states_vt: NDArray[Tuple[float]],
            observes_vt: NDArray[Tuple[float]],
            start: int = 400,
            end: int = 600
    ) -> None:
        """
        Plot the estimated state probabilities, actual states, and observed values.

        Args:
            xi_prob_t: Estimated state probabilities
            states_vt: State values
            observes_vt: Observed values sequence
            start: Starting index for the plot window
            end: Ending index for the plot window
        """
        plt.subplot(3, 1, 1)
        for i in range(xi_prob_t.shape[1]):
            plt.plot(xi_prob_t[start:end, i], label=f"State {i} estimated probability")
        plt.legend(loc="lower left")
        plt.subplot(3, 1, 2)
        plt.plot(states_vt[start:end])
        plt.title("Actual realization of state")
        plt.subplot(3, 1, 3)
        plt.plot(observes_vt[start:end])
        plt.title("Observed time series")
        fig = plt.gcf()
        fig.set_size_inches(16, 12)
        plt.show()


@dataclass
class HmmExpectationPath:
    xi_prob_t: NDArray[Tuple[float, float]]
    xi_prob_t1: NDArray[Tuple[float, float]]
    xi_prob_T: NDArray[Tuple[float, float]]


class HmmSmoother(HmmFilter):

    @staticmethod
    def backward_alg(
            xi_prob_t: NDArray[Tuple[float, float]],
            xi_prob_t1: NDArray[Tuple[float, float]],
            d_states: int,
            t_sample: int,
            p_transition: NDArray[Tuple[float, float]]
    ) -> NDArray[Tuple[float, float]]:
        """Backward algorithm (Kim smoothing) for N-state HMM.

        Implementation of the Kim smoother to compute smoothed probabilities ξ_{t|T}(j) that condition
        on the entire observed sequence Y_1:T rather than just Y_1:t.

        Args:
            xi_prob_t: Forward filtered probabilities ξ_{t|t}(j) = P(s_t=j|Y_1:t)
            xi_prob_t1: One-step ahead predictions ξ_{t+1|t}(j) = P(s_t+1=j|Y_1:t)
            d_states: Number of states in the HMM
            t_sample: Length of observation sequence
            p_transition: Transition probability matrix where P[i,j] = P(s_t+1=j|s_t=i)

        Returns:
            xi_prob_T: Smoothed state probabilities ξ_{t|T}(j) = P(s_t=j|Y_1:T)
        """
        # Initialize array for smoothed probabilities ξ_{t|T}(j)
        xi_prob_T: NDArray[Tuple[float, float]] = np.zeros((t_sample, d_states), dtype=float)

        # Backward recursion to compute smoothed probabilities
        for tt in reversed(range(t_sample)):
            if tt == t_sample - 1:
                # Initialize last time step: ξ_{T|T}(j) = ξ_{T|T}(j) (filtered = smoothed at t=T)
                xi_prob_T[-1, :] = xi_prob_t[-1, :]
                continue
            # Calculate ratio ξ_{t+1|T}(j)/ξ_{t+1|t}(j) for all states j
            xi_T_xi_t = np.divide(xi_prob_T[tt + 1, :], xi_prob_t1[tt, :])
            '''
            # For each state s at time t
            for ss in range(0, d_states):
                # Kim smoothing equation: ξ_{t|T}(s) = ξ_{t|t}(s) * Σ_j [P_{s,j} * ξ_{t+1|T}(j)/ξ_{t+1|t}(j)]
                # This updates the probability of being in state s at time t given the entire sequence
                xi_prob_T[tt, ss] = xi_prob_t[tt, ss] * np.dot(p_transition[ss, :], xi_T_xi)
            '''
            # Vectorized Kim smoothing equation for all states at time t
            # ξ_{t|T}(s) = ξ_{t|t}(s) * Σ_j [P_{s,j} * ξ_{t+1|T}(j)/ξ_{t+1|t}(j)]
            xi_prob_T[tt] = xi_prob_t[tt] * np.dot(p_transition, xi_T_xi_t)
        return xi_prob_T

    @classmethod
    def em_expect(
            cls,
            params_result: HMMParameters,
            d_states: int,
            t_sample: int,
            y_data: NDArray[Tuple[float]],
    ) -> HmmExpectationPath:
        xi_prob_t, xi_prob_t1 = cls.forward_alg(
            pi0=params_result.pi,
            d_states=d_states,
            t_sample=t_sample,
            p_transition=params_result.p_transition,
            mu=params_result.mu,
            sigma=params_result.sigma,
            Y=y_data
        )  # type: NDArray[Tuple[float, float]], NDArray[Tuple[float, float]]

        xi_prob_T: NDArray[Tuple[float, float]] = cls.backward_alg(
            xi_prob_t=xi_prob_t,
            xi_prob_t1=xi_prob_t1,
            d_states=d_states,
            t_sample=t_sample,
            p_transition=params_result.p_transition
        )

        return HmmExpectationPath(xi_prob_t=xi_prob_t, xi_prob_t1=xi_prob_t1, xi_prob_T=xi_prob_T)

    @staticmethod
    def create_joint_transition_tensor(
            xi_prob_t: NDArray[Tuple[float, float]],
            xi_prob_t1: NDArray[Tuple[float, float]],
            xi_prob_T: NDArray[Tuple[float, float]],
            p_hat: NDArray[Tuple[float, float]],
            d_states: int,
            t_sample: int
    ) -> NDArray[Tuple[float, float, float]]:
        """
        Compute joint transition tensor P_hat_T for Pr(s_t+1 = j, s_t = i) in HMM EM algorithm.

        Args:
            xi_prob_t: Forward filtered probabilities, shape (t_sample, d_states)
            xi_prob_t1: One-step ahead predictions, shape (t_sample, d_states)
            xi_prob_T: Smoothed state probabilities, shape (t_sample, d_states)
            p_hat: Transition probability matrix, shape (d_states, d_states)
            d_states: Number of states
            t_sample: Number of time samples

        Returns:
            P_hat_T: Joint transition tensor, shape (d_states, d_states, t_sample)
        """
        # Initialize the joint transition probability tensor P(s_t=i,s_{t+1}=j|Y_1:T)
        # This tensor holds the smoothed joint probabilities of state transitions
        P_hat_T: NDArray[Tuple[float, float, float]] = np.zeros((d_states, d_states, t_sample), dtype=float)

        # Compute joint transition probabilities for all time points (excluding t=0)
        for tt in range(1, t_sample):
            for ss in range(0, d_states):
                for ss2 in range(0, d_states):
                    # Calculate P(s_t=i,s_{t+1}=j|Y_1:T) using Kim smoothing equations:
                    # P(s_t=i,s_{t+1}=j|Y_1:T) = p_{ij} * ξ_{t|t}(i) * ξ_{t+1|T}(j) / ξ_{t+1|t}(j)
                    #
                    # This combines:
                    # - p_{ij}: The transition probability from state i to j
                    # - ξ_{t|t}(i): Filtered probability of being in state i at time t
                    # - ξ_{t+1|T}(j): Smoothed probability of being in state j at time t+1
                    # - 1/ξ_{t+1|t}(j): Normalization by predicted probability
                    P_hat_T[ss, ss2, tt] = (
                            p_hat[ss, ss2]  # p_{ij}: transition probability from i to j
                            * xi_prob_t[tt - 1, ss]  # ξ_{t|t}(i): filtered probability at t
                            * xi_prob_T[tt, ss2]  # ξ_{t+1|T}(j): smoothed probability at t+1
                            / xi_prob_t1[tt - 1, ss2]  # 1/ξ_{t+1|t}(j): inverse of predicted probability
                    )
        return P_hat_T

    @staticmethod
    def em_maximise(
            xi_prob: NDArray[Tuple[float, float]],
            P: NDArray[Tuple[float, float, float]],
            d_states: int,
            Y: NDArray[Tuple[float]]
    ) -> HMMParameters:
        """Maximization step of the EM algorithm for HMM parameter estimation.

        Args:
            xi_prob: Smoothed state probabilities ξ_{t|T}(j) = P(s_t=j|Y_1:T)
            P: Joint state probabilities P[i,j,t] = P(s_t=i,s_{t+1}=j|Y_1:T)
            d_states: Number of states
            Y: Observed data sequence

        Returns:
            mu_hat: Updated state-specific means
            sigma_hat: Updated state-specific standard deviations
            P_hat: Updated transition probability matrix
            pi_hat: Updated initial state probability distribution
        """
        # Initialize arrays for updated parameter estimates
        mu_hat: NDArray[Tuple[float]] = np.zeros((d_states,), dtype=float)
        sigma_hat: NDArray[Tuple[float]] = np.zeros((d_states,), dtype=float)
        P_hat: NDArray[Tuple[float, float]] = np.zeros((d_states, d_states), dtype=float)
        pi_hat: NDArray[Tuple[float]] = np.zeros((d_states,), dtype=float)

        for ss in range(0, d_states):
            # Calculate μ_i^{(k)} = ∑_{t=1}^T ξ_{t|T}^{(k-1)}(i) y_t / ∑_{t=1}^T ξ_{t|T}^{(k-1)}(i)
            # This is a weighted average of observations where weights are smoothed state probabilities
            xi_y = np.dot(xi_prob[:, ss], Y)  # Numerator: ∑_{t=1}^T ξ_{t|T}(i) y_t
            mu_hat[ss] = float(xi_y.item() if np.ndim(xi_y) == 0 else xi_y) / float(
                np.sum(xi_prob[:, ss]))  # Divide by ∑_{t=1}^T ξ_{t|T}(i)

            # Calculate σ_i^{(k)} = √(∑_{t=1}^T ξ_{t|T}^{(k-1)}(i) (y_t-μ_i^{(k)})^2 / ∑_{t=1}^T ξ_{t|T}^{(k-1)}(i))
            # This is the weighted standard deviation where weights are smoothed state probabilities
            xi_y_mu2 = np.dot(xi_prob[:, ss], (Y - mu_hat[ss]) ** 2)  # Numerator: ∑_{t=1}^T ξ_{t|T}(i) (y_t-μ_i)^2
            sigma_hat[ss] = float(
                (xi_y_mu2 / np.sum(xi_prob[:, ss])).item()) ** 0.5  # Divide by ∑_{t=1}^T ξ_{t|T}(i) and take sqrt

            # Calculate p_{ij}^{(k)} = ∑_{t=2}^T P(s_t=i,s_{t+1}=j|Y_1:T) / ∑_{t=2}^T ∑_j P(s_t=i,s_{t+1}=j|Y_1:T)
            # This estimates transition probability from state i to j using joint smoothed probabilities
            '''
            for ss2 in range(0, N):
                P_hat[ss, ss2] = np.sum(P[ss, ss2, 1:]) / np.sum(P[ss, :, 1:])
            '''
            P_hat[ss, :] = np.sum(P[ss, :, 1:], axis=1) / np.sum(np.sum(P[ss, :, 1:], axis=0))
            # May want to normalize P_hatt across all possible next states - np.sum(xi_prob[0:T-1,ss]) or np.sum(P[ss,:,1:T])
            # LOGGER.info( np.sum(xi_prob[0:T-1,ss]), np.sum(P[ss,:,1:T]))

            # Set π_i^{(k)} = ξ_{1|T}^{(k-1)}(i), the smoothed probability of starting in state i
            pi_hat[ss] = xi_prob[0, ss]

        return HMMParameters(
            mu=mu_hat, sigma=sigma_hat, p_transition=P_hat, pi=pi_hat
        )

    @classmethod
    def plot_probabilities(
            cls,
            time_index: pd.DatetimeIndex,
            xi_prob_T: NDArray[Tuple[float, float]],
            y_data: NDArray[Tuple[float]],
            hmm_parameters: HMMParameters,
            transparency: float = 0.3
    ):
        d_states: int = xi_prob_T.shape[1]
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 6))

        # Plot state 0 probability
        for d in range(d_states):
            ax1.plot(
                time_index, xi_prob_T[:, d],
                alpha=transparency, label=f"State {d}"
            )
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State Probability")
        ax1.tick_params(axis="y")
        ax1.set_title("State Probabilities")

        # Plot the index of the maximum probability state between the two states of xi_prob_T on ax1 twinx
        ax1_2 = ax1.twinx()
        max_state = np.argmax(xi_prob_T, axis=1)
        ax1_2.plot(
            time_index, max_state,
            color='black', marker='x', linestyle='None', alpha=0.3, label="Max State Index"
        )
        ax1_2.set_ylabel("Argmax state")
        ax1_2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax1.legend(loc="upper left")

        # With ax2, plot the VIX index
        ax2.plot(time_index, y_data, alpha=transparency, label="VIX Index")
        # plot the means of the max_state per step
        ax2.plot(time_index, hmm_parameters.mu[max_state], alpha=transparency, label="Mean of Max State")
        ax2.fill_between(
            time_index,
            hmm_parameters.mu[max_state] - hmm_parameters.sigma[max_state],
            hmm_parameters.mu[max_state] + hmm_parameters.sigma[max_state],
            color="gray",
            alpha=0.2,
            label="Mean ± Sigma"
        )
        ax2.legend(loc="upper left")

        # Use non-blocking mode
        plt.show()


    @classmethod
    def optimise_smoother(
            cls,
            df_vix: pd.DataFrame,
            p_transition=np.array([
                [0.75, 0.25],
                [0.3, 0.7]
            ]),
            mu_initial=np.array([2, 4]),
            sigma_initial=np.array([0.1, 0.1]),
            pi_initial=np.array([0.5, 0.5]),
            # Determine maximum number of iterations until convergence and convergence tolerance
            itemax=25,
            itetol=5e-2,

            skip_plot: bool = True,
            test_code: Optional[int] = None
    ):
        y_data: NDArray[Tuple[float]] = df_vix.to_numpy().reshape(-1)  # T x 1 2D array of vix

        # SET INITIAL GUESSES
        d_states: int = p_transition.shape[0]
        t_sample = len(y_data)
        params_result: HMMParameters = HMMParameters(
            p_transition=p_transition.copy(),
            mu= mu_initial.copy(),
            sigma=sigma_initial.copy(),
            # Roll transition to initial state probabilities to make it stationary, if exists
            pi=np.dot(pi_initial.copy(), np.linalg.matrix_power(p_transition, 100))
        )
        params_result.check_parameter_shapes()
        params_result.log_iter(logger=LOGGER)

        e_path: HmmExpectationPath = None # type: ignore
        for ite in range(itemax):
            # Expectation step
            e_path: HmmExpectationPath = cls.em_expect(
                params_result=params_result,
                d_states=d_states,
                t_sample=t_sample,
                y_data=y_data
            )
            if test_code == 1:
                if ite == 0:
                    check_em_expect(
                        e_path=e_path,
                        params_result=params_result,
                        d_states=d_states,
                        t_sample=t_sample,
                        y_data=y_data
                    )

            if not skip_plot:
                cls.plot_probabilities(
                    xi_prob_T=e_path.xi_prob_T,
                    time_index=df_vix.index,
                    y_data=y_data,
                    hmm_parameters=params_result
                )
                skip_plot = True

            # Maximisation step
            opt_params: HMMParameters = cls.em_maximise(
                xi_prob=e_path.xi_prob_T,
                P=cls.create_joint_transition_tensor(
                    xi_prob_t=e_path.xi_prob_t,
                    xi_prob_t1=e_path.xi_prob_t1,
                    xi_prob_T=e_path.xi_prob_T,
                    p_hat=params_result.p_transition,
                    d_states=d_states,
                    t_sample=t_sample
                ),
                d_states=d_states,
                Y=y_data
            )
            params_diff: NDArray[Tuple[float]] = params_result.calculate_differential(opt_params)
            params_result.log_iter(
                logger=LOGGER,
                ite=ite,
                diffs_=params_diff
            )
            # to the next iteration
            params_result = opt_params.copy()
            # Convert 2D arrays to compact string representation for logging
            # convergence check
            if np.max(params_diff) < itetol:
                LOGGER.info(f"Convergence reached at iteration {ite} with diff={params_diff} < itertol={itetol}")
                break

        if test_code == 1:
            np.testing.assert_array_almost_equal(
                actual=params_result.mu, desired=np.array([2.49921924, 3.04007959]), decimal=8
            )
            np.testing.assert_array_almost_equal(
                actual=params_result.sigma, desired=np.array([0.09329688, 0.29801484]), decimal=8
            )
            np.testing.assert_array_almost_equal(
                actual=params_result.p_transition, desired=np.array([[0.98050049, 0.01949951], [0.0062869, 0.9937131]]),
                decimal=8
            )
            np.testing.assert_array_almost_equal(
                actual=params_result.pi, desired=np.array([2.91218071e-56, 1.00000000e+00]), decimal=8
            )

        if e_path.xi_prob_T is None:
            raise ValueError("xi_prob_T should not be None, check the backward algorithm implementation.")
        cls.plot_probabilities(
            xi_prob_T=e_path.xi_prob_T,
            time_index=df_vix.index,
            y_data=y_data,
            hmm_parameters=params_result
        )

        return HMMParameters(
            mu=params_result.mu,
            sigma=params_result.sigma,
            p_transition=params_result.p_transition,
            pi=params_result.pi
        )


def original_forward_alg(
        pi0: NDArray[Tuple[float]],
        d_states: int,
        t_sample: int,
        p_transition: NDArray[Tuple[float, float]],
        mu: NDArray[Tuple[float]],
        sigma: float | NDArray[Tuple[float]],
        Y: NDArray[Tuple[float]]
) -> Tuple[NDArray[Tuple[float, float]], NDArray[Tuple[float, float]]]:
    """Hamilton filter for a hidden Markov model with Gaussian emissions - see the equivalence in HMM.forward_alg"""
    xi_prob_t: NDArray[Tuple[float, float]] = np.zeros((t_sample, d_states), dtype=float)
    xi_prob_t1: NDArray[Tuple[float, float]] = np.zeros((t_sample, d_states), dtype=float)

    # Case t=1
    y_like = HmmFilter.likelihood(pi0, mu, sigma, Y[0])
    phi: NDArray[Tuple[float]] = None  # type: ignore
    for ss in range(0, d_states):
        phi = np.zeros((d_states), dtype=float)
        for ss2 in range(0, d_states):
            phi[ss2] = norm.pdf(np.squeeze((Y[0] - mu[ss2]) / sigma[ss2]))
    if phi is None:
        raise ValueError("phi should not be None, check the likelihood calculation.")

    xi_prob_t[0, :] = np.multiply(pi0, phi) / y_like
    for ss in range(0, d_states):
        xi_prob_t1[0, ss] = np.dot(p_transition[:, ss], xi_prob_t[0, :])

    for tt in range(1, t_sample):
        y_like = HmmFilter.likelihood(xi_prob_t1[tt - 1, :], mu, sigma, Y[tt])
        for ss in range(0, d_states):
            phi: NDArray[Tuple[float]] = np.zeros((d_states), dtype=float)
            for ss2 in range(0, d_states):
                phi[ss2] = norm.pdf(np.squeeze((Y[tt] - mu[ss2]) / sigma[ss2]))
        xi_prob_t[tt, :] = np.multiply(xi_prob_t1[tt - 1, :], phi) / y_like
        for ss in range(0, d_states):
            xi_prob_t1[tt, ss] = np.dot(p_transition[:, ss], xi_prob_t[tt, :])

    return xi_prob_t, xi_prob_t1


class DataColumn(StrEnum):
    DATE = "DATE"
    OPEN = "OPEN"
    HIGH = "HIGH"
    LOW = "LOW"
    CLOSE = "CLOSE"


def load_vix_data(
        file_path: str = "./VIX_History_L4.csv",
        vix_col: str = DataColumn.CLOSE,
        date_i: dt.date = dt.date(1990, 1, 1),
        date_f: dt.date = dt.date(2022, 6, 30),
        skip_plot: bool = True
) -> pd.DataFrame:
    PData = pd.read_csv(file_path)

    YDatapd = PData.set_index(DataColumn.DATE)[[vix_col]]
    YDatapd.index = pd.to_datetime(YDatapd.index)
    if date_i and date_f:
        YDatapd = YDatapd.loc[date_i: date_f]
    YDatapd.loc[:, vix_col] = np.log(YDatapd[vix_col])
    YData = YDatapd.to_numpy()
    if not skip_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
        ax1.plot(YDatapd.index, YData)
        ax1.set_ylabel("log(VIX)")
        ax1.set_title("VIX time series")

        ax2.hist(YData, bins=50, orientation='horizontal', color='gray', alpha=0.3, density=True)
        ax2.set_ylabel("Histogram (density)", labelpad=10)
        ax2.yaxis.set_label_position("right")
        ax2.set_xlabel("Density")
        ax2.set_title("Distribution")
        ax2.yaxis.tick_right()

        plt.show()

    return YDatapd


def run_calibration_on_samples_from_true_hmm(
        data: HMMPath,
        p_transition: NDArray[Tuple[float, float]],
        state_values: List[float],
        len_path: int,
        sigma: float,
        len_burnin: int,
        pi_initial: NDArray[Tuple[float]],
        mu_initial: NDArray[Tuple[float]],
        test_code: Optional[int] = None  # None if not testing, else for testing purposes
):
    """Run calibration of HMM parameters on synthetic data generated from a true HMM."""
    d_states: int = len(state_values)
    check_transition_matrix(p_transition=p_transition, d_states=d_states)

    t_sample: int = len_path - len_burnin
    xi_prob_t, _xi_prob_t1 = HmmFilter.forward_alg(
        pi0=pi_initial,
        d_states=d_states,
        t_sample=t_sample,
        p_transition=p_transition,
        mu=mu_initial,
        sigma=sigma,
        Y=data.y_data
    )
    LOGGER.info(f"xi_prob_t:\n{xi_prob_t}")
    if test_code == 1:
        np.testing.assert_array_almost_equal(actual=xi_prob_t[0], desired=np.array([9.94e-04, 9.99e-01]), decimal=3)
        np.testing.assert_array_almost_equal(actual=xi_prob_t[-1], desired=np.array([4.067e-05, 1.000e+00]), decimal=3)
        np.testing.assert_array_almost_equal(actual=_xi_prob_t1[0], desired=np.array([0.15, 0.85]), decimal=3)
        np.testing.assert_array_almost_equal(actual=_xi_prob_t1[-1], desired=np.array([0.15, 0.85]), decimal=3)

    # Optimize parameters
    lik_model: OptimizeResult = HmmFilter.optimise_filter(
        pi0=pi_initial,
        sigma=sigma,
        Y=data.y_data,
        n_states=d_states
    )
    # Extract estimated parameters
    optimised_parameters: NDArray[Tuple[float, ...], float] = lik_model.x
    if test_code == 1:
        np.testing.assert_array_almost_equal(
            actual=optimised_parameters,
            desired=np.array([ 0.481,  0.85 , -1.031,  1.007]),
            decimal=3
        )
        np.testing.assert_approx_equal(actual=lik_model.fun, desired=1868.35956, significant=9)
    LOGGER.info(f"opt_params_hat: {optimised_parameters}")

    p_hat: NDArray[Tuple[float, float]] = HmmFilter.get_p_transition_from_search_array(parameters_search=optimised_parameters)
    mu_hat: NDArray[Tuple[float]] = HmmFilter.get_mu_from_search_array(parameters_search=optimised_parameters)
    # Run forward algorithm with estimated parameters
    xi_t, _xi_prob_t1 = HmmFilter.forward_alg(
        pi0=pi_initial,
        d_states=d_states,
        t_sample=t_sample,
        p_transition=p_hat,
        mu=mu_hat,
        sigma=sigma,
        Y=data.y_data
    )
    if test_code == 1:
        np.testing.assert_array_almost_equal(actual=xi_prob_t[0], desired=np.array([9.94e-04, 9.99e-01]), decimal=3)
        np.testing.assert_array_almost_equal(actual=xi_prob_t[-1], desired=np.array([4.067e-05, 1.000e+00]), decimal=3)
        np.testing.assert_array_almost_equal(actual=_xi_prob_t1[0], desired=np.array([0.151, 0.849]), decimal=3)
        np.testing.assert_array_almost_equal(actual=_xi_prob_t1[-1], desired=np.array([0.15, 0.85]), decimal=3)

    # Plot results
    HmmFilter.plot_histogram(data.y_data)
    HmmFilter.plot_estimation(
        xi_prob_t=xi_t,
        states_vt=data.states_vt,
        observes_vt=data.y_data
    )


def ex_hmm_filter_2d(skip_plot_sampling: bool = True):
    # Define model parameters
    state_v = [-1.0, 1.0]
    d_states: int = len(state_v)

    # Create transition matrix
    p_transition = np.array([
        [0.45, 0.55],
        [0.15, 0.85]
    ])
    check_transition_matrix(p_transition=p_transition, d_states=d_states)

    LEN_HIST: int = 1500
    SIGMA: float = 0.5
    BURNIN: int = 500

    # Initial parameters hypothesis
    pi_initial: NDArray[Tuple[float]] = np.array([1 / d_states] * d_states, dtype=float)
    mu_initial: NDArray[Tuple[float]] = np.array([-1, 1], dtype=float)  # Initial means for each state

    sampled_path: HMMPath = HmmFilter.generate_markov_sequence(
        P=p_transition,
        state_arr=state_v,
        len_path=LEN_HIST,
        len_burnin=BURNIN,
        sigma=SIGMA,
        state_initial=0,
        seed_val=12345
    )
    # Plot synthetic data
    if not skip_plot_sampling:
        HmmFilter.plot_histogram(sampled_path.y_data)

    run_calibration_on_samples_from_true_hmm(
        data=sampled_path,
        p_transition=p_transition,
        state_values=state_v,
        len_path=LEN_HIST,
        sigma=SIGMA,
        len_burnin=BURNIN,
        pi_initial=pi_initial,
        mu_initial=mu_initial,
        test_code=1
    )


def ex_hmm_filter_3d(skip_plot_sampling: bool = True):
    # Define model parameters
    state_v = [-2.0, 0.0, 2.0]
    d_states: int = len(state_v)

    # Create transition matrix
    p_transition = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7]
    ])
    check_transition_matrix(p_transition=p_transition, d_states=d_states)

    LEN_HIST = 1500
    SIGMA = 0.5
    BURNIN = 500

    # Initial parameters hypothesis
    pi_initial: NDArray[Tuple[float]] = np.array([1 / d_states] * d_states, dtype=float)
    mu_initial: NDArray[Tuple[float]] = np.array([-1, 1, 2], dtype=float)  # Initial means for each state

    sampled_path: HMMPath = HmmFilter.generate_markov_sequence(
        P=p_transition,
        state_arr=state_v,
        len_path=LEN_HIST,
        len_burnin=BURNIN,
        sigma=SIGMA,
        state_initial=0,
        seed_val=12345
    )
    # Plot synthetic data
    if not skip_plot_sampling:
        HmmFilter.plot_histogram(sampled_path.y_data)

    run_calibration_on_samples_from_true_hmm(
        data=sampled_path,
        p_transition=p_transition,
        state_values=state_v,
        len_path=LEN_HIST,
        sigma=SIGMA,
        len_burnin=BURNIN,
        pi_initial=pi_initial,
        mu_initial=mu_initial
    )

def check_em_expect(
        e_path: HmmExpectationPath,
        params_result: HMMParameters,
        d_states: int,
        t_sample: int,
        y_data: NDArray[Tuple[float]]
):
    orig_xi_prob_t, orig_xi_prob_t1 = original_forward_alg(
        pi0=params_result.pi, d_states=d_states, t_sample=t_sample,
        p_transition=params_result.p_transition, mu=params_result.mu, sigma=params_result.sigma,
        Y=y_data
    )  # type: NDArray[Tuple[float, float]], NDArray[Tuple[float, float]]
    np.testing.assert_array_almost_equal(actual=e_path.xi_prob_t, desired=orig_xi_prob_t)
    np.testing.assert_array_almost_equal(actual=e_path.xi_prob_t1, desired=orig_xi_prob_t1)

    np.testing.assert_array_almost_equal(
        actual=e_path.xi_prob_T[0], desired=np.array([1.00000000e+00, 1.79325439e-14])
    )
    np.testing.assert_array_almost_equal(
        actual=e_path.xi_prob_T[3], desired=np.array([0.49315249, 0.50684751])
    )


def ex_hmm_smoother_2d_with_vix():
    HmmSmoother.optimise_smoother(
        load_vix_data(skip_plot=True),
        skip_plot=False,
        test_code=1
    )


# Example usage
if __name__ == "__main__":
    np.set_printoptions(precision=3)

    ex_hmm_filter_2d()
    # ex_hmm_filter_3d()

    ex_hmm_smoother_2d_with_vix()
