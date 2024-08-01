"""
Running Multiverse analysis

Multiverse analysis requires a Python script to be created by the user.
An initial template for this can be created through the GUI, with forking paths being stored in a dict and later used through double curly braces in the template function.

This example shows how one would create and run a multiverse analysis which will generate 3 Python scripts (universes) printing the numbers 1, 2, and 3, respectively.
"""

from comet.multiverse import Multiverse

forking_paths = {
    "numbers": [1, 2, 3]
}

def analysis_template():
    print({{numbers}})

multiverse = Multiverse(name="multiverse_example")
multiverse.create(analysis_template, forking_paths)
multiverse.summary()
#multiverse.run()
