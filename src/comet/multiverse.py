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
from scipy.interpolate import make_interp_spline
from joblib import Parallel, delayed

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
        self.name = name.split('/')[-1].split('.')[0]
        self.calling_script_dir = os.getcwd() if in_notebook() else os.path.abspath(sys.modules['__main__'].__file__).rsplit('/', 1)[0]
        self.multiverse_dir = os.path.join(self.calling_script_dir, self.name)
        self.results_dir = os.path.join(self.multiverse_dir, "results")

    # Public methods
    def create(self, analysis_template, forking_paths, config=None):
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

        # If multiverse directory exists, remove all Python files but keep folders and template
        if os.path.exists(self.multiverse_dir):
            for item in os.listdir(self.multiverse_dir):
                item_path = os.path.join(self.multiverse_dir, item)
                if os.path.isfile(item_path) and item.endswith(".py") and item != "template.py":
                    os.remove(item_path)
        else:
            os.makedirs(self.multiverse_dir)

        # Ensure the results directory exists
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

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

        if config.get("order"):
            all_universes = []
            for order in config["order"]:
                
                reordered_values = [forking_paths[key] for key in order]
                unused_keys = [key for key in keys if key not in order]

                # Generate all unique combinations for the reordered values
                reordered_universes = list(itertools.product(*reordered_values))

                # Format each combination as (key, value) tuples and include None for remaining keys
                formatted_universes = [
                    tuple((key, value) for key, value in zip(order, combination)) +
                    tuple((key, "unused") for key in unused_keys)
                    for combination in reordered_universes
                ]
                all_universes.extend(formatted_universes)
        else:
            # Default behavior: Only provided order
            all_universes = [
                tuple((key, value) for key, value in zip(keys, combination))
                for combination in itertools.product(*values)
            ]
                
        # Remove universes that contain invalid paths
        if config.get("invalid_paths"):
            valid_universes = [combination for combination in all_universes if not self._check_paths(combination, config["invalid_paths"])]
        else:
            valid_universes = all_universes

        # Create Python scripts for each combination
        for i, combination in enumerate(valid_universes, start=1):
            context = {key: self._format_type(value) for key, value in combination}
            rendered_content = jinja_template.render(**context)
            
            # Convert combination to a dictionary
            forking_dict = {key: value for key, value in combination}
            
            if config.get("order"):
                forking_dict_str = f"# Ordering information was provided. The ordered decisions for this universe are:\nforking_paths = {forking_dict}\n\n"
                rendered_content = forking_dict_str + rendered_content

            # Write to Python script
            save_path = os.path.join(self.multiverse_dir, f"universe_{i}.py")
            with open(save_path, "w") as file:
                file.write(rendered_content)

        # Generate CSV file with the decisions of all universes
        self._create_csv(self.results_dir, valid_universes, keys)

        # Save forking paths
        with open(f"{self.results_dir}/forking_paths.pkl", "wb") as file:
            pickle.dump(forking_paths, file)

        return

    def run(self, universe_number=None, parallel=1):
        """
        Run either an individual universe or the entire multiverse

        Parameters
        ----------
        universe_number : int
            Number of the universe to run. Default is None, which runs all universes

        parallel : int
            Number of universes to run in parallel
        """

        os.makedirs(self.multiverse_dir, exist_ok=True)
        sorted_files = sorted(os.listdir(self.multiverse_dir))

        # Delete previous results (.pkl files)
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isfile(item_path) and item_path.endswith('.pkl') and not item_path.endswith('forking_paths.pkl'):
                os.remove(item_path)

        # Function for parallel processing, called by joblib.delayed
        def execute_script(file):
            print(f"Running {file}")
            subprocess.run([sys.executable, os.path.join(self.multiverse_dir, file)], check=True, env=os.environ.copy())

        if universe_number is None:
            print("Starting multiverse analysis for all universes...")
            Parallel(n_jobs=parallel)(delayed(execute_script)(file) for file in sorted_files if file.endswith(".py") and not file.startswith("template"))
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
            multiverse_selection = multiverse_summary.iloc[universe-1]
        elif isinstance(universe, range):
            multiverse_selection = multiverse_summary.iloc[universe.start-1:universe.stop]

        if in_notebook():
            from IPython.display import display
            display(multiverse_selection)
        else:
            print(multiverse_selection)

        return multiverse_summary

    def visualize(self, universe=None, cmap="Set2", node_size=1500, figsize=(8,5), label_offset=0.04, exclude_single=False):
        """
        Visualize the multiverse as a network.

        Parameters
        ----------
        universe : int or None
            The universe to highlight in the network. If None, no path will be highlighted. Default is None

        cmap : str
            Colormap to use for the nodes. Default is "Set2"

        node_size : int
            Size of the nodes. Default is 1500

        figsize : tuple
            Size of the figure. Default is (8,5)
        """

        multiverse_summary = self._read_csv()
        value_columns = [col for col in multiverse_summary.columns if (col.startswith("Value") or col == "Universe")]
        decision_columns = [col for col in multiverse_summary.columns if col.startswith("Decision")]

        multiverse_values = multiverse_summary[value_columns]
        multiverse_decision = multiverse_summary[decision_columns]

        # Function to recursively add nodes and edges
        def add_hierarchical_decision_nodes(G, root_node, parameters, level=0, exclude_single=False):
            if level >= len(parameters):
                return G  # No more parameters to process

            parameter, options = parameters[level]
            if not exclude_single or len(options) > 1:  # Include node if exclude_single is False or there are multiple options
                for option in options:
                    # Create a node for each option and connect it to the parent node
                    node_name = f"{parameter}: {option}"
                    G.add_node(node_name, level=level + 1, option=option, decision=parameter)  # Store both option and decision
                    G.add_edge(root_node, node_name)
                    # Recursively add nodes for the next level/parameter
                    G = add_hierarchical_decision_nodes(G, node_name, parameters, level + 1, exclude_single)
            else:
                # If excluding single-option parameters, skip to the next parameter
                G = add_hierarchical_decision_nodes(G, root_node, parameters, level + 1, exclude_single)

            return G

        # List of dicts. Each dict is a decision with its options
        value_to_decision_map = {value_col: decision_col for value_col, decision_col in zip(value_columns[1:], decision_columns)}

        parameters = [
            (value_to_decision_map.get(parameter, parameter), multiverse_values[parameter].unique())
            for parameter in multiverse_values.columns[1:]
            if len(multiverse_values[parameter].unique()) > 1
        ]
        parameters = [(decision, values[pd.notna(values)]) for decision, values in parameters] # remove nans

        # Initialize and build the graph
        G = nx.DiGraph()
        root_node = 'Start'
        G.add_node(root_node, level=0, label="Start")  # Ensure the root node has the 'level' attribute and label
        G = add_hierarchical_decision_nodes(G, root_node, parameters, exclude_single=exclude_single)

        values = ["Start"]  # Initialize the list with the "Start" value
        if universe is not None:
            # Get the decisions for the desired universe
            filtered_df = multiverse_values[multiverse_values['Universe'] == f"Universe_{universe}"]
            for column in filtered_df.columns:
                if column not in [param_name for param_name, _ in parameters]:
                    filtered_df = filtered_df.drop(columns=column)

            row_dict = filtered_df.iloc[0].to_dict()
            values.extend([f"{column}: {value}" for column, value in row_dict.items()])

        # Red edge colors for the edges that are part of the universe
        universe_edges = [(source_value, target_value) for source_value, target_value in G.edges if source_value in values and target_value in values]
        edge_colors = ["black"] * len(G.edges)
        edge_widths = [1.0] * len(G.edges)
        for i, edge in enumerate(G.edges):
            if edge in universe_edges:
                edge_colors[i] = "black"
                edge_widths[i] = 2.5
            else:
                edge_colors[i] = "gray"
                edge_widths[i] = 1.0

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        title_str = f"Universe {universe}" if universe is not None else "Multiverse"
        ax.set_title(title_str, size=14, fontweight='bold')

        pos = nx.multipartite_layout(G, subset_key="level")

        # Assigning colors based on levels using a colormap
        levels = set(nx.get_node_attributes(G, 'level').values())
        num_levels = len(levels)
        first_color = 'lightgray'
        colormap = plt.cm.get_cmap(cmap, num_levels)
        colors = [first_color] + [colormap(i / (num_levels - 1)) for i in range(0, num_levels)]
        level_colors = {level: colors[i] for i, level in enumerate(sorted(levels))}

        # Draw edges first because of the node size
        nx.draw(G, pos, with_labels=False, node_size=node_size-10, node_color="white", arrows=True, edge_color=edge_colors, width=edge_widths, ax=ax)

        # Draw nodes with colors based on their level
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get('level') == level]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_at_level, node_size=node_size, node_color=[level_colors[level] for _ in nodes_at_level], ax=ax)

        # Draw labels using corresponding entries from multiverse_decision DataFrame
        node_labels = {}
        for node in G.nodes:
            if node != root_node and 'decision' in G.nodes[node]:
                decision = G.nodes[node]['decision']
                option = G.nodes[node]['option']
                
                # Find the first matching row where the column equals the decision
                if decision in multiverse_decision.columns:
                    decision_label = multiverse_decision[decision].iloc[0]  # Extract the first value for simplicity
                    node_labels[node] = f"{decision_label}\n{option}"  # Combine the corresponding decision entry with the option
                else:
                    node_labels[node] = f"{decision}\n{option}"  # Fallback if no matching entry found
            else:
                node_labels[node] = G.nodes[node]['label']  # For the root node

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, ax=ax)

        # Identify and annotate the bottom-most node at each level with the decision label
        levels = set(nx.get_node_attributes(G, 'level').values())

        # Create an appropriate label offset based on the maximum level node count
        node_nums = []
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get('level') == level]
            node_nums.append(len(nodes_at_level))

        # Draw the labels
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get('level') == level]

            if nodes_at_level:
                bottom_node = min(nodes_at_level, key=lambda node: pos[node][1])
                if bottom_node != root_node and 'decision' in G.nodes[bottom_node]:
                    decision = G.nodes[bottom_node]['decision']
                    x, y = pos[bottom_node]
                    ax.text(x, y-label_offset * max(node_nums), decision, horizontalalignment='center', fontsize=12, fontweight='bold')

        plt.savefig(f"{self.results_dir}/multiverse.png", bbox_inches='tight')

        return fig

    def specification_curve(self, measure, baseline=None, p_value=None, ci=None, smooth_ci=True, \
                            title="Specification Curve", name_map=None, cmap="Set3", linewidth=2, figsize=(16,9), \
                            height_ratio=(2,1), fontsize=10, dotsize=50, ftype="png"):
        """
        Create and save a specification curve plot from multiverse results

        Parameters
        ----------
        measure : string
            Name of the measure to plot. Needs to be provided

        baseline : float
            Plot baseline/chance level as a dashed line. Default is None

        p_value : float
            Calculate and visualize statisticallz significant specification via p-value. Default is None

        ci : int
            Confidence interval to plot. Default is 95

        ci_smooth : bool
            Plot a smoothed confidence interval. Default is False

        title : string
            Title of the plot. Default is "Specification Curve"

        name_map : dict
            Dictionary to map the decision names to custom names. Default is None

        cmap : string
            Colormap to use for the nodes. Default is "Set3"

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

        ftype : string
            File type to save the plot. Default is "png"
        """

        # Check if the results directory exists
        universe_files = [f for f in os.listdir(self.results_dir) if f.startswith('universe_') and f.endswith('.pkl')]
        if not universe_files:
            raise ValueError("No results found. Please run the multiverse analysis first.")

        # Sort the universes based on the measure and get the forking paths
        sorted_universes, forking_paths = self._load_and_prepare_data(measure, self.results_dir)

        # Plotting
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': height_ratio}, sharex=True)

        # If the forking path is a single parameter only, we will not plot it and delete it from the forking paths
        single_params = []
        for decision, options in forking_paths.items():
            for i, opt in enumerate(options):
                if isinstance(opt, dict):
                    options[i] = opt.get('name')

            if len(options) == 1:
                single_params.append(decision)

        for param in single_params:
            del forking_paths[param]

        # Setup variables for the bottom plot
        flat_list = []
        yticks = []
        yticklabels = []
        line_ends = []
        p_vals = []
        key_positions = {}
        y_max = 0
        space_between_groups = 1
        color = "black"
        sig_color = "#018532"

        for key, options in forking_paths.items():
            # Get custom labels if a name map is provided
            if name_map is not None and key in name_map.keys():
                key_label = name_map[key]
            else:
                key_label = key

            key_position = y_max + len(options) / 2 - 0.5
            key_positions[key_label] = key_position

            for option in options:
                flat_list.append((key, option))
                yticks.append(y_max)
                yticklabels.append(option)
                y_max += 1

            line_ends.append(y_max)
            y_max += space_between_groups

        decision_info = (yticklabels, yticks, line_ends, forking_paths.keys())

        # Ticks and labels for bottom plot
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(yticklabels)
        ax[1].tick_params(axis='y', labelsize=10)
        ax[1].set_ylim(-1, y_max)
        ax[1].xaxis.grid(False)

        # Calculate the padding for the lines and labels
        fig.canvas.draw()
        trans1 = transforms.blended_transform_factory(ax[1].transAxes, ax[1].transData)

        # Get the bounding box of the y-tick labels
        renderer = fig.canvas.get_renderer()
        tick_label_extents = [label.get_window_extent(renderer=renderer) for label in ax[1].get_yticklabels()]
        max_extent = max(tick_label_extents, key=lambda bbox: bbox.width)

        # Convert the right edge of the bounding box to axis coordinates
        x_start_pixel = max_extent.x0  # leftmost edge of the bounding box
        x_start_axes = ax[1].transAxes.inverted().transform((x_start_pixel, 0))[0]

        padding = 0.01
        line_offset =  x_start_axes - padding

        # Plot the key labels
        for key, pos in key_positions.items():
            ax[1].text(line_offset - padding, pos, key, transform=trans1, ha='right', va='center',
                    fontweight="bold", fontsize=fontsize, rotation=0)

        # Lines for the bottom plot
        s = -0.5
        for i, line_end in enumerate(line_ends):
            e = line_end - 0.5
            line = mlines.Line2D([line_offset, line_offset], [s, e], color="black", lw=1, transform=trans1, clip_on=False)
            ax[1].add_line(line)
            s = line_end + 0.5

        # Setup variables for the CI
        ci_lower_values = []
        ci_upper_values = []
        x_values = np.arange(len(sorted_universes))

        # Check the length of the results
        result, decisions = sorted_universes[0]
        if hasattr(result, '__len__') and len(result) < 30:
            print(f"Warning: Only {len(result)} samples were available for the t-test and CI.")

        # Plotting of the dots in both plots
        for i, (result, decisions) in enumerate(sorted_universes):
            # If the measure for the top plot contains multiple values (e.g. CV results), we calculate the mean for the plot
            if hasattr(result, '__len__'):
                mean_val = np.mean(result)
            else:
                mean_val = result

            # Color coding for p-values (only for more than 30 samples)
            if p_value is not None:
                baseline = 0 if baseline is None else baseline # Compare against 0 if no baseline is provided

                if hasattr(result, '__len__'):
                    t_obs, p_obs = stats.ttest_1samp(result, baseline)
                    p_vals.append(p_obs)
                    color = sig_color if p_obs < p_value else "black"

            # Plot the confidence interval
            if ci is not None:
                # Only calculate for more than 3 samples
                if hasattr(result, '__len__') and len(result) > 3:
                    sem_val = np.std(result) / np.sqrt(len(result))
                    ci_lower = mean_val - sem_val * stats.t.ppf((1 + ci / 100) / 2., len(result) - 1)
                    ci_upper = mean_val + sem_val * stats.t.ppf((1 + ci / 100) / 2., len(result) - 1)

                    ci_lower_values.append(ci_lower)
                    ci_upper_values.append(ci_upper)

                    # Optional: Plot individual CIs if smooth_ci is False
                    if not smooth_ci:
                        ax[0].plot([i, i], [ci_lower, ci_upper], color="gray", linewidth=linewidth)

                else:
                    ci_lower_values.append(mean_val)  # In case of no CI, just use the mean value
                    ci_upper_values.append(mean_val)

            # Plot the universe measure
            ax[0].scatter(i, mean_val, zorder=3, color=color, edgecolor=color, s=dotsize)

            # Colors for the lower plot, each decision group has a different color
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

        # Plot the smooth CI band after the loop
        if smooth_ci and ci is not None:
            # Smoothing the CI bounds
            spline_lower = make_interp_spline(x_values, ci_lower_values, k=3)  # cubic spline
            spline_upper = make_interp_spline(x_values, ci_upper_values, k=3)

            x_smooth = np.linspace(x_values.min(), x_values.max(), 500)
            ci_lower_smooth = spline_lower(x_smooth)
            ci_upper_smooth = spline_upper(x_smooth)

            # Plot the smooth CI band
            ax[0].fill_between(x_smooth, ci_lower_smooth, ci_upper_smooth, color='gray', alpha=0.3)

        # Upper plot settings
        trans0 = transforms.blended_transform_factory(ax[0].transAxes, ax[0].transData)

        ax[0].set_title(title, fontweight="bold", fontsize=fontsize+2)
        ax[0].xaxis.grid(False)
        ax[0].set_xticks(np.arange(0, len(sorted_universes), 1))
        ax[0].set_xticklabels([])
        ax[0].set_xlim(-1, len(sorted_universes) + 1)

        # Upper plot label
        ymin, ymax = ax[0].get_ylim()
        ycenter = (ymax + ymin) / 2

        # Get custom label if provided
        if name_map is not None and measure in name_map.keys():
            measure_label = name_map[measure]
        else:
            measure_label = measure

        ax[0].text(line_offset - padding, ycenter, measure_label, transform=trans0, ha='right', va='center', \
                   fontweight="bold", fontsize=fontsize, rotation=0)
        line = mlines.Line2D([line_offset, line_offset], [ymin, ymax], color="black", lw=1, transform=trans0, clip_on=False)
        ax[0].add_line(line)

        # List for legend items
        legend_items = []

        # If the chance level is whithin the range of the measure, we plot it as a dashed line, otherwise we skip it for visualization purpose
        y_lim = ax[0].get_ylim()

        if baseline is not None and y_lim[0] <= baseline <= y_lim[1]:
            ax[0].hlines(baseline, xmin=-2, xmax=len(sorted_universes) + 1, linestyles="--", lw=2, colors='black', zorder=1)
            legend_items.append(mlines.Line2D([], [], linestyle='--', color='black', linewidth=2, label=f"Baseline"))

        # Legend items for confidence interval
        if hasattr(result, '__len__') and len(result) > 3 and ci is not None:
            legend_items.append(mpatches.Patch(facecolor='gray', edgecolor='white', label=f"{ci}% CI"))

        # Legend items for t-tests
        if p_value is not None:
            if len(p_vals) > 0:
                if min(p_vals) <= p_value:
                    legend_items.append(mlines.Line2D([], [], linestyle='None', marker='o', markersize=9, markerfacecolor=sig_color, markeredgecolor=sig_color, label=f"p < 0.05"))
                if max(p_vals) > p_value:
                    legend_items.append(mlines.Line2D([], [], linestyle='None', marker='o', markersize=9, markerfacecolor="black", markeredgecolor="black", label=f"p > 0.05"))
            else:
                print("Warning: No p-values were calculated (less than 30 samples)")

        # Add the legend to the plot if it contains at least one item
        if len(legend_items) > 0:
            ax[0].legend(handles=legend_items, loc='upper left', fontsize=fontsize)

        # Save the plot
        plt.savefig(f"{self.results_dir}/specification_curve.{ftype}", bbox_inches='tight')

        return fig

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
        csv_path = os.path.join(self.results_dir, "multiverse_summary.csv")

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
        elif value is None:
            return 'None'
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

        if value["func"].startswith("comet.connectivity"):
            function_call = f"{func}({input_data}, **{args}).estimate()"
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
            List of all universes (combinations of parameters)

        keys : list
            List of keys for the CSV file. Used when the order is consistent.

        config : dict
            Configuration dictionary. Used to check for the 'order' key.
        """
        fieldnames = ['Universe']
        for i in range(len(keys)):
            fieldnames.append(f"Decision {i+1}")
            fieldnames.append(f"Value {i+1}")

        # Generate CSV file with the parameters of all universes
        with open(f"{csv_path}/multiverse_summary.csv", "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, combination in enumerate(all_universes, start=1):
                context = {'Universe': f"Universe_{i}"}

                # Populate the decision and value columns
                for j, (key, value) in enumerate(combination):
                    context[f"Decision {j+1}"] = key
                    context[f"Value {j+1}"] = value

                writer.writerow(context)

    def _load_and_prepare_data(self, measure=None, results_path=None):
        """
        Internal function: Load and prepare the data for the specification curve plotting

        Parameters
        ----------
        measure : string
            Name of the measure to plot. Default is None

        results_path : string
            Path to the results directory

        Returns
        -------
        sorted_universes : list
            List of sorted universes with their summary

        forking_paths : dict
            Dictionary containing the forking paths
        """
        summary_path = os.path.join(results_path, "multiverse_summary.csv")
        multiverse_summary = pd.read_csv(summary_path)

        with open(f"{results_path}/forking_paths.pkl", "rb") as file:
            forking_paths = pickle.load(file)

        # Get the results files
        pattern = os.path.join(results_path, "universe_*.pkl")
        results_files = glob.glob(pattern)

        # Get the specified mesure from the .pkl files
        universe_data = {}
        for filename in results_files:
            universe = os.path.basename(filename).split('.')[0]

            with open(filename, "rb") as file:
                universe_data[universe] = pickle.load(file)[measure]

        # Create combined data structure and sort by the measure by mean
        universes_with_summary = []
        for universe, data in universe_data.items():
            summary_row = multiverse_summary[multiverse_summary['Universe'].str.lower() == universe]
            if not summary_row.empty:
                universes_with_summary.append((data, summary_row.iloc[0]))

        sorted_universes = sorted(universes_with_summary, key=lambda x: np.mean(x[0]))

        return sorted_universes, forking_paths


# Helper function
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

def notebookToScript(notebook):
        """
        Convert a Jupyter notebook JSON to a Python script.
        """
        scriptContent = ""
        try:
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    scriptContent += ''.join(cell['source']) + '\n\n'
        except KeyError as e:
            print("Error", f"Invalid notebook format: {str(e)}")

        return scriptContent
