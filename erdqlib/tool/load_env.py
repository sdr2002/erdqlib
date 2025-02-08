import os
import logging
from dotenv import load_dotenv
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def get_erdqlib_package_root() -> Path:
    return Path(__file__).parent.parent


def get_erdqlib_project_root() -> Path:
    return get_erdqlib_package_root().parent


def load_env_from_dotenv():
    dotenv_path = os.path.join(get_erdqlib_project_root(), '.env')
    LOGGER.info(f"Loading .env file from {dotenv_path}")

    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f".env file not found at {dotenv_path}")
    load_dotenv()
    LOGGER.info(f"Loaded .env file from {dotenv_path}")


if __name__ == "__main__":
    # Configure logger to display info messages on the console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    load_env_from_dotenv()
    LOGGER.info(os.getenv('OPENAI_API_KEY'))