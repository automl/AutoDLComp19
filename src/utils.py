import datetime
import hjson


def print_log(*content):
    """Logging function. (could've also used `import logging`.)"""
    now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    print("MODEL INFO: " + str(now) + " ", end="")
    print(*content)


class Config:
    def __init__(self, config_path):
        with open(config_path) as config_file:
            self.__dict__ = hjson.load(config_file)
