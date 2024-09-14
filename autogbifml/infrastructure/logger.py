import logging
import sys

__LOGGER = None


def init_logger(name):
    # setup logging
    __LOGGER = logging.getLogger(name)
    __LOGGER.propagate = False

    # create console handler and set level to info
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(message)s")
    stdout.setFormatter(formatter)

    __LOGGER.addHandler(stdout)

    # set level to info
    __LOGGER.setLevel(logging.INFO)

    return __LOGGER
