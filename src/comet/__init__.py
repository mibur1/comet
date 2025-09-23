from . import connectivity
from . import multiverse
from . import graph
from . import utils
from . import cifti
from . import bids

__all__ = ["connectivity", "multiverse", "graph", "utils", "cifti", "bids"]

def launch_gui(*args, **kwargs):
    from .gui import run
    return run(*args, **kwargs)