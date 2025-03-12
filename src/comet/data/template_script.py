"""
Running Multiverse analysis

Multiverse analysis requires a Python script to be created by the user.
An initial template for this can be created through the GUI, with forking paths being stored in a dict and later used through double curly braces in the template function.

This example shows how one would create and run a multiverse analysis which will generate 4 (2*2) Python scripts (universes) calculating the sum of two numbers.
"""

from comet.multiverse import Multiverse

forking_paths = {
    "number_1": [1, 2],
    "number_2": [3, 4]
}

def analysis_template():
    import os 
    
    # The result of each universe is the addition of two numbers
    addition = {{number_1}} + {{number_2}}

    # Save results
    result = {"addition": addition}
    comet.utils.save_universe_results(result, universe=os.path.abspath(__file__))
