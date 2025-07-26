from pathlib import Path

import numpy as np


def save_nparray_to_txt(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, delimiter=',', fmt='%.9e')  # Save an empty array as a placeholder
    return


def load_nparray_from_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=',', dtype=np.float64)  # Assuming the data is comma-separated


def compare_arrays_and_update(actual: np.ndarray, expected_path: Path, rtol: float=1e-6, atol: float=1e-6):
    """
    Compare an actual numpy array with an expected one loaded from a file.
    If they do not match, save the actual array to a new file with 'new.' prefix.

    :param actual: Actual numpy array to compare.
    :param expected_path: Path to the expected numpy array file.
    :param rtol: Relative tolerance for comparison.
    :param atol: Absolute tolerance for comparison.
    :return: None
    """
    new_expected_path = expected_path.parent / f"new.{expected_path.name}"

    if not expected_path.exists():
        save_nparray_to_txt(actual, new_expected_path)
        raise FileNotFoundError(f"{expected_path} does not exist. Saved actual array to {new_expected_path}.")

    expected = load_nparray_from_txt(expected_path)
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        save_nparray_to_txt(actual, new_expected_path)
        raise AssertionError(f"Arrays do not match. Saved actual array to {new_expected_path}.")


# # TODO correctly implement the dataframe comparison tool: naive save/load roundtrip fails
# def load_dataframe_from_csv(path: Path) -> pd.DataFrame:
#     return pd.read_csv(path, dtype=float)  # Assuming the data is float type
#
#
# def save_dataframe_to_csv(df: pd.DataFrame, path: Path):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(path, index=False)  # Save an empty DataFrame as a placeholder
#     return
#
#
# def compare_dataframe_and_update(actual: pd.DataFrame, expected_path: Path):
#     """
#     Compare an actual numpy array with an expected one loaded from a file.
#     If they do not match, save the actual array to a new file with 'new.' prefix.
#
#     :param actual: Actual numpy array to compare.
#     :param expected_path: Path to the expected numpy array file.
#     :param rtol: Relative tolerance for comparison.
#     :param atol: Absolute tolerance for comparison.
#     :return: None
#     """
#     new_expected_path = expected_path.parent / f"new.{expected_path.name}"
#
#     if not expected_path.exists():
#         save_dataframe_to_csv(actual, new_expected_path)
#         raise FileNotFoundError(f"{expected_path} does not exist. Saved actual array to {new_expected_path}.")
#
#     expected = load_dataframe_from_csv(expected_path)
#     if not pd.testing.assert_frame_equal(actual, expected, check_exact=False):
#         save_dataframe_to_csv(actual, new_expected_path)
        raise AssertionError(f"Arrays do not match. Saved actual array to {new_expected_path}.")
