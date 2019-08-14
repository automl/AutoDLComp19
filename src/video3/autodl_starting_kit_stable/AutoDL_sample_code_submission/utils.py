import os
import sys
import logging


BASEDIR = os.path.dirname(os.path.abspath(__file__))


def get_logger():
    """Set logging format to something like:
            2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, 'INFO')
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    fileout_handler = logging.FileHandler(os.path.join(BASEDIR, 'run.log'), mode='w')
    fileout_handler.setLevel(logging.DEBUG)
    fileout_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(fileout_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


LOGGER = get_logger()
