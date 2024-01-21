import os
import json
import inspect
import itertools
import subprocess
from tqdm import tqdm
from jinja2 import Template


def create(analysis_template, forking_paths):
    # Get the directory of the calling script
    calling_script_file = inspect.stack()[1].filename
    calling_script_dir = os.path.dirname(os.path.abspath(calling_script_file))
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
    all_universes = itertools.product(*values)

    # Create scripts for each combination
    for i, combination in enumerate(all_universes, start=1):
        context = {key: format_type(value) for key, value in zip(keys, combination)}
        rendered_content = jinja_template.render(**context)

        # Write to Python script
        save_path = os.path.join(save_script_dir, f"universe_{i}.py")
        with open(save_path, "w") as file:
            file.write("\n".join(rendered_content.split('\n')))


def run(path=None, universe_number=None):
    if path is None:
        calling_script_file = inspect.stack()[1].filename
        calling_script_dir = os.path.dirname(os.path.abspath(calling_script_file))
        universe_dir = os.path.join(calling_script_dir, "universes")
        path = universe_dir

    sorted_files = sorted(os.listdir(path))

    if universe_number is None:
        print("Performing multiverse analysis for all universes, please wait...")
        for file in tqdm(sorted_files):
            if file.endswith(".py"):
                subprocess.run(["python", os.path.join(path, file)], check=True)
    else:
        print(f"Performing analysis for universe {universe_number}, please wait...")
        subprocess.run(["python", os.path.join(path, f"universe_{universe_number}.py")], check=True)


def format_type(value):
    if isinstance(value, str):
        return f"'{value}'"  # Strings are wrapped in quotes
    elif isinstance(value, int):
        return str(value)  # Integers are converted directly
    elif isinstance(value, float):
        return str(value)  # Floats are converted directly
    elif isinstance(value, bool):
        return "True" if value else "False"  # Booleans are converted to their literal representations
    elif isinstance(value, dict):  # Dictionaries
        return handle_dict(value) # Function to handle dictionaries which are used for classes and functions
    elif isinstance(value, type):  # Check if the value is a class
        return value.__name__  # Return the name of the class
    elif callable(value):  # Check if the value is a function
        return value.__name__  # Return the name of the function
    else:
        return "'<unsupported type>'"
    
def handle_dict(value):
    # Check if the dictionary is a class
    if "class" in value.keys():
        dfc_class = value["class"]
        dfc_data = value["input_data"]
        dfc_args = value["args"]
        function_call = f"methods.{dfc_class}({dfc_data}, **{dfc_args}).connectivity()"
    return function_call
