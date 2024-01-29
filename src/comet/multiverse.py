import os
import sys
import csv
import inspect
import itertools
import subprocess
import pandas as pd
from tqdm import tqdm
from jinja2 import Template
from joblib import Parallel, delayed

import glob
import pickle
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# This function creates all the universe scripts
def create(analysis_template, forking_paths):
    # Get the directory of the calling script
    calling_script_dir = os.getcwd() if in_notebook else os.path.dirname(sys.path[0])

    save_script_dir = os.path.join(calling_script_dir, "universes")
    if not os.path.exists(save_script_dir):
        os.mkdir(save_script_dir)

    # Template creation
    template_code = inspect.getsource(analysis_template) # Extract the source code
    template_body = "\n".join(template_code.splitlines()[1:])# Remove the function definition

    # Determine the indentation level of the first line of the function body and remove it from all lines
    first_line = template_body.splitlines()[0]
    indentation_level = len(first_line) - len(first_line.lstrip())
    adjusted_lines = [line[indentation_level:] if len(line) > indentation_level else line for line in template_body.splitlines()]
    adjusted_template_body = "\n".join(adjusted_lines)
    
    # Create jinja template
    jinja_template = Template(adjusted_template_body)

    # Generate all unique combinations of forking paths
    keys, values = zip(*forking_paths.items())
    all_universes = list(itertools.product(*values))

    # Generate CSV file with the parameters of all universes
    csv_path = os.path.join(calling_script_dir, "universes")
    with open(f"{csv_path}/multiverse_summary.csv", "w", newline='') as csvfile:
        fieldnames = ['Universe'] + list(keys)  # 'Universe' as the first column
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, combination in enumerate(all_universes, start=1):
            context = {'Universe': f"Universe_{i}"}

            for key, value in zip(keys, combination):
                if isinstance(value, dict):
                    context[key] = value.get('name', '')
                else:
                    context[key] = value

            writer.writerow(context)

    # Create Python scripts for each combination
    for i, combination in enumerate(all_universes, start=1):
        context = {key: format_type(value) for key, value in zip(keys, combination)}
        rendered_content = jinja_template.render(**context)

        # Write to Python script
        save_path = os.path.join(save_script_dir, f"universe_{i}.py")
        with open(save_path, "w") as file:
            file.write(rendered_content)

# This function can run all (or individual) universes
def run(path=None, universe_number=None, parallel=1):
    if path is None:
        calling_script_dir = os.getcwd() if in_notebook else os.path.dirname(sys.path[0])
        universe_dir = os.path.join(calling_script_dir, "universes")
        path = universe_dir

    sorted_files = sorted(os.listdir(path))

    # Function for parallel processing, called by joblib.delayed
    def execute_script(file):
        print(f"Starting {file}")
        subprocess.run(["python", os.path.join(path, file)], check=True)

    if universe_number is None:
        print("Starting multiverse analysis for all universes...")
        Parallel(n_jobs=parallel)(delayed(execute_script)(file) for file in sorted_files if file.endswith(".py"))
    else:
        print(f"Starting analysis for universe {universe_number}...")
        subprocess.run(["python", os.path.join(path, f"universe_{universe_number}.py")], check=True)

# Prints a summary of all universes
def summary(type=None):
    calling_script_dir = os.getcwd() if in_notebook else os.path.dirname(sys.path[0])
    csv_path = os.path.join(calling_script_dir, "universes/multiverse_summary.csv")
    multiverse_summary = pd.read_csv(csv_path)
    print(multiverse_summary.head()) if type == "head" else print(multiverse_summary)
    print("")

# This handles the types of the decision points to generate a working template script
def format_type(value):
    if isinstance(value, str):
        return f"'{value}'"  # Strings are wrapped in quotes
    elif isinstance(value, int):
        return str(value)  # Integers are converted directly
    elif isinstance(value, float):
        return str(value)  # Floats are converted directly
    elif isinstance(value, bool):
        return "True" if value else "False" # Booleans are converted to their literal representations
    elif isinstance(value, dict):
        return handle_dict(value) # Dictionaries are handeled in a separate function
    elif isinstance(value, type):
        return value.__name__  # If the forking path is a class, we return the name of the class
    elif callable(value):     
        return value.__name__ # If the forking path is a function, we return the name of the function
    else:
        raise TypeError(f"Unsupported type for {value} which is of type {type(value)}")

# This handles the decision points that require class/function imports
def handle_dict(value):
    function_call = ""

    if "connectivity" in value.keys():
        dfc_class = value["connectivity"]
        input_data = value["input_data"]
        dfc_args = value["args"]
        function_call = f"comet.methods.{dfc_class}({input_data}, **{dfc_args}).connectivity()"
    
    if "graph" in value.keys():
        graph_function = value["graph"]
        input_data = value["input_data"]
        
        # If we have args, we need to handle them for indviduak 
        try:
            graph_args = value["args"].copy()  # Copy to avoid mutating the original

            # Measures that need a community affiliation vector
            if graph_function in("bct.participation_coef", "bct.participation_coef_sparse", "bct.participation_coef_sign", \
                                 "bct.agreement_wei", "bct.diversity_coef_sign", "bct.gateway_coef_sign", "bct.module_degree_zscore"):
                ci = graph_args["ci"]
                del graph_args["ci"]
                function_call = f"{graph_function}({input_data}, ci={ci}[0], **{graph_args})"
            # Measures that have additional numerical or string arguments
            else:
                function_call = f"{graph_function}({input_data}, **{graph_args})"
        
        # Measures that only need an adjacency matrix as input
        except KeyError:
            function_call = f"{graph_function}({input_data})"

    return function_call

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except Exception:
        return False
    return True
