import datetime


def print_log(*content):
    """Logging function. (could've also used `import logging`.)"""
    now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    print("MODEL INFO: " + str(now) + " ", end="")
    print(*content)
