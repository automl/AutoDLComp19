import os
import sys
import logging
import hjson


BASEDIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    def __init__(self, config_path):
        with open(config_path) as config_file:
            self.__dict__ = hjson.load(config_file)

    def write(self, save_path):
        with open(save_path, "w") as save_file:
            save_file.write(hjson.dumps(self.__dict__))


def get_logger():
    """Set logging format to something like:
            2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    conf = Config(os.path.join(BASEDIR, "config.hjson"))
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, conf.log_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


LOGGER = get_logger()
