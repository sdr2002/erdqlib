import logging


def create_logger(path, logging_level=logging.INFO):
    logging.basicConfig(
        level=logging_level,
        format='\033[37m[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s\033[0m',
        datefmt='%H:%M:%S'
    )

    return logging.getLogger(path)