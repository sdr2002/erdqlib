import os
import logging
from dotenv import load_dotenv
from pathlib import Path

LOGGER = logging.getLogger(__file__)


def get_erdqlib_package_root() -> Path:
    return Path(__file__).parent.parent


def get_erdqlib_project_root() -> Path:
    return get_erdqlib_package_root().parent


def load_env_from_dotenv():
    dotenv_path = os.path.join(get_erdqlib_project_root(), '.env')
    print(f"Loading .env file from {dotenv_path}")

    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f".env file not found at {dotenv_path}")
    load_dotenv()
    print(f"Loaded .env file from {dotenv_path}")


if __name__ == "__main__":
    load_env_from_dotenv()
    print(os.getenv('OPENAI_API_KEY'))