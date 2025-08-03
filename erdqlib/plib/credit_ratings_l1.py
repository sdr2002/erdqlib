import numpy as np
from numpy.random import rand
from typing import Dict

from erdqlib.tool.logger_util import create_logger

LOGGER = create_logger(__name__)
np.set_printoptions(precision=2)


def simulate_default_times(
        random_seed: int,
        transition_matrix: np.ndarray,
        state_map: Dict[str, int],
        state0: str,
        n_paths: int = 1000,
        n_steps: int = 100
) -> np.ndarray:
    np.random.seed(random_seed)

    LOGGER.info(f"transition_matrix: {transition_matrix}")
    randarray: np.ndarray = rand(n_paths, n_steps)
    default_time: np.ndarray = np.zeros(n_paths)
    histories: np.ndarray = np.zeros((n_paths, n_steps), dtype=np.int8)
    histories[:, 0] = state_map[state0]

    for i in range(n_paths):
        # Simulate each independent credit rating history
        for j in range(1, n_steps):
            # For each time step (except the initial state)
            prev_state: int = int(histories[i, j - 1])  # Get the previous rating/state index
            cumsum_probs: np.ndarray = np.cumsum(
                transition_matrix[prev_state])  # Cumulative transition probabilities for this state
            rand_val: float = float(randarray[i, j])  # Random value for this path and time step
            next_state: int = int(
                np.searchsorted(cumsum_probs, rand_val))  # Determine next state by where rand_val fits in cumsum_probs
            histories[i, j] = next_state  # Record the new state (rating) for this time step
            if next_state == state_map["D"]:
                # If the new state is 'D' (default), stop further transitions for this path
                break
        # After simulating this path, check if default ('D') was reached
        if np.any(histories[i, :] == state_map["D"]):
            where_default: int = int(
                np.where(histories[i, :] == state_map["D"])[0][0])  # Find the first time step where default occurred
            default_time[i] = where_default  # Record the time to default for this path
            # (default_sum is not used here, but could be incremented to count number of defaults)
        else:
            default_time[i] = 0.0  # If no default, set time to default as 0

    LOGGER.info(f"Default time: {float(np.sum(default_time)) / np.sum(default_time > 0, dtype=float)}")
    return default_time


if __name__ == "__main__":
    # Example usage
    P0 = np.array(
        [
            [87.06, 9.06, 0.53, 0.05, 0.11, 0.03, 0.05, 0.0, 3.11],
            [0.48, 87.23, 7.77, 0.47, 0.05, 0.06, 0.02, 0.02, 3.89],
            [0.03, 1.6, 88.58, 5.0, 0.26, 0.11, 0.02, 0.05, 4.35],
            [0, 0.09, 3.25, 86.49, 3.56, 0.43, 0.1, 0.16, 5.92],
            [0.01, 0.03, 0.11, 4.55, 77.82, 6.8, 0.55, 0.63, 9.51],
            [0.0, 0.02, 0.07, 0.15, 4.54, 74.6, 4.96, 3.34, 12.33],
            [0.0, 0.0, 0.1, 0.17, 0.55, 12.47, 43.11, 28.3, 15.31],
        ]
    )
    P0 = np.append(P0, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0]], axis=0)
    # Normalize transition matrix, ignoring NR type
    transition_matrix: np.ndarray = P0[:, :-1]
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
    state_map = {
        "AAA": 0,
        "AA": 1,
        "A": 2,
        "BBB": 3,
        "BB": 4,
        "B": 5,
        "CCC": 6,
        "D": 7,
    }

    simulate_default_times(
        random_seed=12345,
        transition_matrix=transition_matrix,
        state_map=state_map,
        state0="CCC",
        n_paths=1000,
        n_steps=100
    )
