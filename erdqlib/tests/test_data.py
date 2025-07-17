from pathlib import Path

import numpy as np


def save_nparray_to_txt(arr: np.ndarray, path: Path):
    """
    Save a numpy array to a text file.

    :param path: Path to the text file.
    :return: None
    """
    if not path.parent.exists():
        raise FileNotFoundError(f"Directory {path.parent} does not exist.")

    np.savetxt(path, arr, delimiter=',', fmt='%.18e')  # Save an empty array as a placeholder
    return


def load_nparray_from_txt(path: Path) -> np.ndarray:
    """
    Load a numpy array from a text file.

    :param path: Path to the text file.
    :return: Numpy array loaded from the text file.
    """
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    return np.loadtxt(path, delimiter=',', dtype=float)  # Assuming the data is comma-separated


def compare_arrays_and_update(actual: np.ndarray, expected_path: Path, rtol: float=1e-6, atol: float=1e-6):
    """
    Compare an actual numpy array with an expected one loaded from a file.
    If they do not match, save the actual array to a new file with 'new.' prefix.

    :param actual: Actual numpy array to compare.
    :param expected_path: Path to the expected numpy array file.
    :return: None
    """
    new_expected_path = expected_path.parent / f"new.{expected_path.name}"

    if not expected_path.exists():
        save_nparray_to_txt(actual, new_expected_path)
        raise FileNotFoundError(f"{expected_path} does not exist. Saved actual array to {new_expected_path}.")

    expected = load_nparray_from_txt(expected_path)
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        save_nparray_to_txt(actual, new_expected_path)
        raise AssertionError(f"Arrays do not match. Saved actual array to {new_expected_path}.")
