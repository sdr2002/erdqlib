from pathlib import Path


def get_path_from_package(path_alias: str) -> str:
    """Get the path to the erdqlib library.

    For instance, if the path_alis is given as "erdqlib@bar/tests/src/mc/data/test_vasicek_paths_values.csv",
      and the path of the "erdqlib" package is in "/home/foo/erdqlib",
      then the returned path will be "/home/foo/erdqlib/bar/tests/src/mc/data/test_vasicek_paths_values.csv".
    """
    if "@" in path_alias:
        package_name, relative_path = path_alias.split("@", 1)
        package_path = Path(__import__(package_name).__file__).parent
        return str(package_path / relative_path)
    else:
        return str(Path(path_alias).resolve())