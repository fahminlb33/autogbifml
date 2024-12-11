import sys
import logging


class BaseCommand:
    def __init__(self) -> None:
        # setup logging
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.propagate = True

        # create console handler and set level to info
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter(
            "%(name)s: %(asctime)s | %(levelname)s | %(message)s"
        )
        stdout.setFormatter(formatter)

        self.logger.addHandler(stdout)

        # set level to info
        self.logger.setLevel(logging.DEBUG)
