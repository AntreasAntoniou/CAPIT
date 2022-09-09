import logging

from rich.logging import RichHandler
from rich.traceback import install


def get_logger(name=__name__, set_default_handler=False) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    if set_default_handler:
        logger.setLevel(logging.INFO)
        ch = RichHandler()
        ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter("%(levelname)s - %(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    install(show_locals=False, extra_lines=1, word_wrap=True, width=350)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup

    return logger
