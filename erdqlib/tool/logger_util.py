import logging


def create_logger(path, logging_level=logging.INFO):
    logger = logging.getLogger(path)
    logger.setLevel(logging_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    formatter = logging.Formatter(
        '\033[40m[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s\033[0m',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger