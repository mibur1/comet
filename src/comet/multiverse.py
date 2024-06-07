import os
import sys
import csv
import glob
import pickle
import inspect
import itertools
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import scipy.stats as stats
from jinja2 import Template
from matplotlib import transforms
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
from joblib import Parallel, delayed

def in_notebook():
    """
    Helper function to check if the code is running in a Jupyter notebook
    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except Exception:
        return False
    return True

class Multiverse:
    """
    Multiverse class for creating, running, and visualizing the multiverse analysis.

    This class provides functionality to create multiple analysis scripts based on different decision paths,
    run them in parallel, and visualize the results as a network or specification curve.

    Attributes
    ----------
    name : str
        Name of the multiverse analysis. Default is "multiverse".
    """

    def __init__(self, name="multiverse"):
        self.name = name

    # Public methods
    def create(self, analysis_template, forking_paths, invalid_paths=None):
        """
        Create the individual universe scripts

        Parameters
        ----------
        analysis_template : function
            Function containing the analysis template

        forking_paths : dict
            Dictionary containing the forking paths

        invalid_paths : list
            List of invalid paths that should be excluded from the multiverse
        """

        # Get the directory of the calling script
        calling_script_dir = os.getcwd() if in_notebook else os.path.dirname(sys.path[0])

        # Create directories
        multiverse_dir = os.path.join(calling_script_dir, self.name)
        os.makedirs(multiverse_dir, exist_ok=True)
        results_dir = os.path.join(multiverse_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

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

        # Remove universes that contain invalid paths
        if invalid_paths is not None:
            valid_universes = [combination for combination in all_universes if not self._check_paths(combination, invalid_paths)]
        else:
            valid_universes = all_universes

        # Create Python scripts for each combination
        for i, combination in enumerate(valid_universes, start=1):
            context = {key: self._format_type(value) for key, value in zip(keys, combination)}
            rendered_content = jinja_template.render(**context)

            # Write to Python script
            save_path = os.path.join(multiverse_dir, f"universe_{i}.py")
            with open(save_path, "w") as file:
                file.write(rendered_content)

        # Generate CSV file with the decisions of all universes
        self._create_csv(results_dir, valid_universes, keys)

        # Save forking paths
        with open(f"{results_dir}/forking_paths.pkl", "wb") as file:
            pickle.dump(forking_paths, file)

        return

    def run(self, path=None, universe_number=None, parallel=1):
        """
        Run either an individual universe or the entire multiverse

        Parameters
        ----------
        path : string
            Path of the multiverse directory

        universe_number : int
            Number of the universe to run. Default is None, which runs all universes

        parallel : int
            Number of universes to run in parallel
        """

        if path is None:
            calling_script_dir = os.getcwd() if in_notebook else os.path.dirname(sys.path[0])
            multiverse_dir = os.path.join(calling_script_dir, self.name)
            os.makedirs(multiverse_dir, exist_ok=True)
            path = multiverse_dir

        sorted_files = sorted(os.listdir(path))

        # Function for parallel processing, called by joblib.delayed
        def execute_script(file):
            print(f"Starting {file}")
            subprocess.run([sys.executable, os.path.join(path, file)], check=True, env=os.environ.copy())

        if universe_number is None:
            print("Starting multiverse analysis for all universes...")
            Parallel(n_jobs=parallel)(delayed(execute_script)(file) for file in sorted_files if file.endswith(".py"))
        else:
            print(f"Starting analysis for universe {universe_number}...")
            file = f"universe_{universe_number}.py"
            execute_script(file)

        return

    def summary(self, universe=range(1,5)):
        """
        Print the multiverse summary to the terminal/notebook

        Parameters
        ----------
        universe : int or range
            The universe number(s) to display. Default is range(1,5)
        """

        multiverse_summary = self._read_csv()

        if isinstance(universe, int):
            multiverse_summary = multiverse_summary.iloc[universe-1]
        elif isinstance(universe, range):
            multiverse_summary = multiverse_summary.iloc[universe.start-1:universe.stop]

        if in_notebook:
            from IPython.display import display
            display(multiverse_summary)
        else:
            print(multiverse_summary + "\n")

    def visualize(self, universe=None, cmap="Set2", node_size=1500, figsize=(8,5)):
        """
        Visualize the multiverse as a network

        Parameters
        ----------
        universe : int
            The universe to highlight in the network. Default is None

        cmap : str
            Colormap to use for the nodes. Default is "Set2"

        node_size : int
            Size of the nodes. Default is 1500

        figsize : tuple
            Size of the figure. Default is (8,5)
        """

        multiverse_summary = self._read_csv()

        # Function that recursively add nodes and edges excluding single-option parameters
        def add_hierarchical_decision_nodes(G, root_node, parameters, level=0):
            if level >= len(parameters):
                return G  # No more parameters to process

            parameter, options = parameters[level]
            if len(options) > 1:  # Only consider parameters with more than one option
                for option in options:
                    # Create a node for each option and connect it to the parent node
                    node_name = f"{parameter}: {option}"
                    G.add_node(node_name, level=level + 1, option=option, decision=parameter)  # Store both option and decision
                    G.add_edge(root_node, node_name)
                    # Recursively add nodes for the next level/parameter
                    G = add_hierarchical_decision_nodes(G, node_name, parameters, level + 1)
            else:
                # If the current parameter has only one option, skip to the next parameter
                G = add_hierarchical_decision_nodes(G, root_node, parameters, level + 1)

            return G

        # List of dicts. Each dict is a decision with its options
        parameters = [
            (parameter, multiverse_summary[parameter].unique())
            for parameter in multiverse_summary.columns[1:]
            if len(multiverse_summary[parameter].unique()) > 1
        ]

        # Initialize and build the graph
        G = nx.DiGraph()
        root_node = 'Start'
        G.add_node(root_node, level=0, label="Start")  # Ensure the root node has the 'level' attribute and label
        G = add_hierarchical_decision_nodes(G, root_node, parameters)

        # Get the decisions for the desired universe
        filtered_df = multiverse_summary[multiverse_summary['Universe'] == f"Universe_{universe}"]
        for column in filtered_df.columns:
            if column not in [param_name for param_name, _ in parameters]:
                filtered_df = filtered_df.drop(columns=column)

        row_dict = filtered_df.iloc[0].to_dict()
        values = ["Start"]  # Initialize the list with the "Start: Start" value
        values.extend([f"{column}: {value}" for column, value in row_dict.items()])


        # Red edge colors for the edges that are part of the universe
        red_edges = [(source_value, target_value) for source_value, target_value in G.edges if source_value in values and target_value in values]
        edge_colors = ["black"] * len(G.edges)
        edge_widths = [1.0] * len(G.edges)
        for i, edge in enumerate(G.edges):
            if edge in red_edges:
                edge_colors[i] = "black"
                edge_widths[i] = 2.5
            else:
                edge_colors[i] = "gray"
                edge_widths[i] = 1.0

        # Visualize the graph
        plt.figure(figsize=figsize)
        plt.title(f"Universe {universe}", size=14, fontweight='bold')
        pos = nx.multipartite_layout(G, subset_key="level")

        # Assigning colors based on levels using a colormap
        levels = set(nx.get_node_attributes(G, 'level').values())
        num_levels = len(levels)
        first_color = 'lightgray'
        colormap = plt.cm.get_cmap(cmap, num_levels)
        colors = [first_color] + [colormap(i / (num_levels - 1)) for i in range(0, num_levels)]
        level_colors = {level: colors[i] for i, level in enumerate(sorted(levels))}

        # Draw edges first because of the node size
        nx.draw(G, pos, with_labels=False, node_size=node_size-10, node_color="white", arrows=True, edge_color=edge_colors, width=edge_widths)

        # Draw nodes with colors based on their level
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get('level') == level]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_at_level, node_size=node_size, node_color=[level_colors[level] for _ in nodes_at_level])

        # Draw labels
        node_labels = {node: G.nodes[node]['option'] if node != root_node \
                       else G.nodes[node]['label'] for node in G.nodes} # Use only the option as a node label
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

        # Identify and annotate the bottom-most node at each level with the decision label
        levels = set(nx.get_node_attributes(G, 'level').values())

        # Create an appropriate label offset based on the maximum level node count
        node_nums = []
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get('level') == level]
            node_nums.append(len(nodes_at_level))
        label_offset = 0.04 * max(node_nums)

        # Draw the labels
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get('level') == level]

            if nodes_at_level:
                bottom_node = min(nodes_at_level, key=lambda node: pos[node][1])
                if bottom_node != root_node and 'decision' in G.nodes[bottom_node]:
                    decision = G.nodes[bottom_node]['decision']
                    x, y = pos[bottom_node]
                    plt.text(x, y-label_offset, decision, horizontalalignment='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()

        return

    def specification_curve(self, fname="multiverse_summary.csv", measure=None, cmap="Set2", ci=95, chance_level=None, \
                            linewidth=2, figsize=(16,9), height_ratio=(2,1), fontsize=10, dotsize=50, label_offset=-0.05):
        """
        Create and save a specification curve plot from multiverse results

        Parameters
        ----------
        fname : string
            Name of the .csv file containing the multiverse summary. Default is "multiverse_summary.csv"

        measure : string
            Name of the measure to plot. Default is None

        cmap : string
            Colormap to use for the nodes. Default is "Set2"

        ci : int
            Confidence interval to plot. Default is 95

        chance_level : float
            Chance level to plot. Default is None

        linewidth : int
            Width of the boxplots. Default is 2

        figsize : tuple
            Size of the figure. Default is (16,9)

        height_ratio : tuple
            Height ratio of the two subplots. Default is (2,1)

        fontsize : int
            Font size of the labels. Default is 10

        dotsize : int
            Size of the dots. Default is 50

        label_offset : float
            Offset of the labels. Needs to be adjusted manually, default is -0.05
        """

        calling_script_dir = os.getcwd() if 'in_notebook' in globals() and in_notebook else os.path.dirname(sys.argv[0])
        results_path = os.path.join(calling_script_dir, f"{self.name}/results")

        sorted_combined, forking_paths = self._load_and_prepare_data(fname, measure, results_path)

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': height_ratio}, sharex=True)
        fig.suptitle('Multiverse Analysis', fontweight="bold", fontsize=fontsize+2)

        single_params = []
        for decision, options in forking_paths.items():
            for i, opt in enumerate(options):
                if isinstance(opt, dict):
                    options[i] = opt.get('name')

            if len(options) == 1:
                single_params.append(decision)

        for param in single_params:
            del forking_paths[param]

        flat_list = []
        yticks = []
        yticklabels = []
        current_position = 0
        line_ends = []
        key_positions = {}
        space_between_groups = 1

        for key, options in forking_paths.items():
            key_position = current_position + len(options) / 2 - 0.5
            key_positions[key] = key_position

            for option in options:
                flat_list.append((key, option))
                yticks.append(current_position)
                yticklabels.append(option)
                current_position += 1

            line_ends.append(current_position)
            current_position += space_between_groups

        decision_info = (yticklabels, yticks, line_ends, forking_paths.keys())

        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(yticklabels)
        ax[1].tick_params(axis='y', labelsize=10)
        ax[1].set_ylim(-1, current_position)
        ax[1].yaxis.grid(False)

        trans1 = transforms.blended_transform_factory(ax[1].transAxes, ax[1].transData)

        for key, pos in key_positions.items():
            ax[1].text(label_offset - 0.01, pos, key, transform=trans1, ha='right', va='center', \
                       fontweight="bold", fontsize=fontsize, rotation=0)

        s = -0.5
        for i, line_end in enumerate(line_ends):
            e = line_end - 0.5
            line = mlines.Line2D([label_offset, label_offset], [s, e], color="black", lw=1, transform=trans1, clip_on=False)
            ax[1].add_line(line)
            s = line_end + 0.5

        for i, (result, decisions) in enumerate(sorted_combined):
            if hasattr(result, '__len__'):
                mean_val = np.mean(result)
            else:
                mean_val = result

            if hasattr(result, '__len__') and len(result) > 3:
                sem_val = np.std(result) / np.sqrt(len(result))
                ci_lower = mean_val - sem_val * stats.t.ppf((1 + ci / 100) / 2., len(result) - 1)
                ci_upper = mean_val + sem_val * stats.t.ppf((1 + ci / 100) / 2., len(result) - 1)

                ax[0].plot([i, i], [ci_lower, ci_upper], color="gray", linewidth=linewidth)

                if ci_lower > 0.5:
                    ax[0].scatter(i, mean_val, zorder=3, color="green", edgecolor="green", s=dotsize)
                elif ci_upper < 0.5:
                    ax[0].scatter(i, mean_val, zorder=3, color="red", edgecolor="red", s=dotsize)
                else:
                    ax[0].scatter(i, mean_val, zorder=3, color="black", edgecolor="black", s=dotsize)
            else:
                if chance_level is not None:
                    if mean_val > chance_level:
                        ax[0].scatter(i, mean_val, zorder=3, color="green", edgecolor="green", s=dotsize)
                    elif mean_val < chance_level:
                        ax[0].scatter(i, mean_val, zorder=3, color="red", edgecolor="red", s=dotsize)
                else:
                    ax[0].scatter(i, mean_val, zorder=3, color="black", edgecolor="black", s=dotsize)

            group_ends = decision_info[2]
            num_groups = len(group_ends) + 1
            colormap = plt.cm.get_cmap(cmap, num_groups)
            colors = [colormap(i) for i in range(num_groups)]

            for decision, option in decisions.items():
                if option in decision_info[0] and decision in decision_info[3]:
                    index = decision_info[0].index(option)
                    plot_pos = decision_info[1][index]

                    current_group = 0
                    for end in group_ends:
                        if plot_pos <= end:
                            break
                        current_group += 1

                    current_color = colors[current_group]

                    ax[1].scatter(i, plot_pos, color=current_color, marker='o', s=dotsize)

        trans0 = transforms.blended_transform_factory(ax[0].transAxes, ax[0].transData)

        ax[0].xaxis.grid(False)
        ax[0].set_xticks(np.arange(0, len(sorted_combined), 1))
        ax[0].set_xticklabels([])
        ax[0].set_xlim(-1, len(sorted_combined) + 1)

        ymin, ymax = ax[0].get_ylim()
        ycenter = (ymax + ymin) / 2
        ax[0].text(label_offset - 0.01, ycenter, measure, transform=trans0, ha='right', va='center', \
                   fontweight="bold", fontsize=fontsize, rotation=0)
        line = mlines.Line2D([label_offset, label_offset], [ymin, ymax], color="black", lw=1, transform=trans0, clip_on=False)
        ax[0].add_line(line)

        ax[0].hlines(chance_level, xmin=-2, xmax=len(sorted_combined) + 1, linestyles="--", lw=2, colors='black', zorder=1)

        legend_items = []

        if hasattr(result, '__len__') and len(result) > 1:
            legend_items.append(mlines.Line2D([], [], linestyle='None', marker='o', markersize=8, \
                                              markerfacecolor="black", markeredgecolor="black", label=f"Mean {measure}"))
        else:
            legend_items.append(mlines.Line2D([], [], linestyle='None', marker='o', markersize=8, \
                                              markerfacecolor="black", markeredgecolor="black", label=f"{measure}"))

        if hasattr(result, '__len__') and len(result) > 3:
            legend_items.append(mpatches.Patch(facecolor='gray', edgecolor='white', label=f"{ci}% CI"))

        if chance_level is not None:
            legend_items.append(mlines.Line2D([], [], color='black', linestyle='--', lw=linewidth, label='Chance level'))

        ax[0].legend(handles=legend_items, loc='upper left', fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(f"{results_path}/specification_curve.png")

        return

    # Internal methods
    def _check_paths(self, universe_path, invalid_paths):
        """
        Internal function: Check if a universe contains a specific path

        Parameters
        ----------
        universe_path : list
            The "decision path" of a universe

        invalid_paths : list
            List of invalid paths that should be excluded from the multiverse

        Returns
        -------
        result : bool
            Returns True if the universe contains an invalid path, else False
        """

        def get_decision(decision):
            if isinstance(decision, dict) and "name" in decision:
                return decision["name"]
            return decision

        # Standardize the universe_path
        standardized_universe_path = [get_decision(decision) for decision in universe_path]

        for invalid_path in invalid_paths:
            standardized_invalid_path = [get_decision(decision) for decision in invalid_path]

            # Check if standardized_invalid_path is a subsequence of standardized_universe_path
            iter_universe = iter(standardized_universe_path)
            if all(decision in iter_universe for decision in standardized_invalid_path):
                return True  # Found an invalid path, return

        return False

    def _read_csv(self):
        """
        Internal function: Reads the multiverse_summary.csv file

        Returns
        -------
        summary : pandas.DataFrame
            Pandas datframe containing the multiverse summary
        """

        calling_script_dir = os.getcwd() if in_notebook else os.path.dirname(sys.path[0])
        csv_path = os.path.join(calling_script_dir, f"{self.name}/results/multiverse_summary.csv")
        return pd.read_csv(csv_path)

    def _format_type(self, value):
        """
        Internal function: Converts the different data types of decision points
        to strings, which is necessary to create the template script

        Parameters
        ----------
        value : any
            The value to be formatted

        Returns
        -------
        formatted_value : string
            The formatted value
        """

        if isinstance(value, str):
            return f"'{value}'"  # Strings are wrapped in quotes
        elif isinstance(value, int):
            return str(value)  # Integers are converted directly
        elif isinstance(value, float):
            return str(value)  # Floats are converted directly
        elif isinstance(value, bool):
            return "True" if value else "False" # Booleans are converted to their literal representations
        elif isinstance(value, dict):
            return self._handle_dict(value) # Dictionaries are handeled in a separate function
        elif isinstance(value, type):
            return value.__name__  # If the forking path is a class, we return the name of the class
        elif callable(value):
            return value.__name__ # If the forking path is a function, we return the name of the function
        else:
            raise TypeError(f"Unsupported type for {value} which is of type {type(value)}")

    def _handle_dict(self, value):
        """
        Internal function: Handle the decision points that dicts.
        These are the decision points which require class/function imports.

        Parameters
        ----------
        value : dict
            The dictionary to be formatted

        Returns
        -------
        function_call : string
            The formatted function call
        """

        function_call = ""

        func = value["func"]
        args = value["args"].copy()

        if "Option" in args:
            del args["Option"]

        first_arg = next(iter(args))
        input_data = args[first_arg]
        del args[first_arg]

        if value["func"].startswith("comet.methods"):
            function_call = f"{func}({input_data}, **{args}).connectivity()"
        else:
            if "ci" in args:
                ci = args["ci"]
                del args["ci"]
                function_call = f"{func}({input_data}, ci={ci}[0], **{args})"
            else:
                function_call = f"{func}({input_data}, **{args})"

        return function_call

    def _create_csv(self, csv_path, all_universes, keys):
        """
        Internal function: Create a CSV file with the parameters of all universes

        Parameters
        ----------
        csv_path : string
            Path to save the CSV file

        all_universes : list
            List of all universes

        keys : list
            List of keys for the CSV file
        """

        # Generate CSV file with the parameters of all universes
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

    def _load_and_prepare_data(self, fname="multiverse_summary.csv", measure=None, results_path=None):
        """
        Internal function: Load and prepare the data for the specification curve plotting

        Parameters
        ----------
        fname : string
            Name of the .csv file containing the multiverse summary. Default is "multiverse_summary.csv"

        measure : string
            Name of the measure to plot. Default is None

        results_path : string
            Path to the results directory

        Returns
        -------
        sorted_combined : list
            List of sorted universes with their summary

        forking_paths : dict
            Dictionary containing the forking paths
        """

        csv_path = os.path.join(results_path, fname)
        multiverse_summary = pd.read_csv(csv_path)

        if measure in multiverse_summary.columns:
            print(f"Getting {measure} from .csv file")
            forking_paths = {}

            for column in multiverse_summary.columns:
                if column == measure:
                    continue
                unique_values = multiverse_summary[column].unique().tolist()
                forking_paths[column] = unique_values

            universe_data = multiverse_summary[measure].values
            parameters = multiverse_summary.drop(columns=[measure])
            universes_with_summary = [(data, parameters.iloc[i].to_dict()) for i, data in enumerate(universe_data)]
            sorted_combined = sorted(universes_with_summary, key=lambda x: np.mean(x[0]))

        else:
            print(f"Getting {measure} from .pkl files")
            with open(f"{results_path}/forking_paths.pkl", "rb") as file:
                forking_paths = pickle.load(file)

            pattern = os.path.join(results_path, "universe_*.pkl")
            results_files = glob.glob(pattern)

            universe_data = {}
            for filename in results_files:
                universe = os.path.basename(filename).split('.')[0]

                with open(filename, "rb") as file:
                    universe_data[universe] = pickle.load(file)

            universes_with_summary = []
            for universe, data in universe_data.items():
                summary_row = multiverse_summary[multiverse_summary['Universe'].str.lower() == universe]
                if not summary_row.empty:
                    universes_with_summary.append((data, summary_row.iloc[0]))

            sorted_combined = sorted(universes_with_summary, key=lambda x: np.mean(x[0]))

        return sorted_combined, forking_paths
