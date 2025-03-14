import os, threading


def get_root_dir():
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.getcwd() + '/'
    return root_dir


class GlobalConfig:
    debug = False

    root_dir = get_root_dir()
    device = 'cuda:0'
    conf_threshold = 0.33
    half = False
    sz_wh = (1280, 578)
