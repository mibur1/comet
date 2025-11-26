# Ignore some warnings
import warnings
warnings.filterwarnings("ignore", message=r".*invalid escape sequence '\\<'", 
                        category=SyntaxWarning)
warnings.filterwarnings("ignore", message=r"invalid value encountered in divide",
                        category=RuntimeWarning, module=r".*bct\.algorithms\.centrality")
warnings.filterwarnings( "ignore", message=r'.*"is not" with \'tuple\' literal.*', 
                        category=SyntaxWarning)
warnings.filterwarnings("ignore", message=r"Starting a Matplotlib GUI outside of the main thread",
                        category=UserWarning)

# Submodule imports
from . import connectivity
from . import multiverse
from . import graph
from . import utils
from . import cifti
from . import bids

__all__ = ["connectivity", "multiverse", "graph", "utils", "cifti", "bids"]

# GUI launch function
def launch_gui(*args, **kwargs):
    from .gui import run
    return run(*args, **kwargs)