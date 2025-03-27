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
from scipy import stats
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
    num_universes : str
        Number of universes in the multiverse.
    forking_paths : dict
        Dictionary containing the forking paths.
    multiverse_dir : str
        Path to the multiverse directory.
    results_dir : str
        Path to the results directory.
    """

    def __init__(self, name="multiverse", path=None):
        self.name = name.split('/')[-1].split('.')[0]
        self.num_universes = None
        self.forking_paths = None

        if path is not None:
            # Set the directories from the provided path (used by the GUI)
            self.multiverse_dir = path
            self.results_dir = os.path.join(self.multiverse_dir, "results")
            self.script_dir = os.path.join(self.multiverse_dir, "scripts")
        else:
            # Set the directories based on the calling script
            calling_script_dir = os.getcwd() if self._in_notebook() else os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
            self.multiverse_dir = os.path.join(calling_script_dir, self.name)
            self.results_dir = os.path.join(self.multiverse_dir, "results")
            self.script_dir = os.path.join(self.multiverse_dir, "scripts")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.script_dir, exist_ok=True)

    # Public methods
    def create(self, analysis_template, forking_paths, config={}):
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
        if os.path.exists(self.multiverse_dir) and os.path.exists(self.script_dir):
            for item in os.listdir(self.script_dir):
                item_path = os.path.join(self.script_dir, item)
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
            save_path = os.path.join(self.script_dir, f"universe_{i}.py")
            with open(save_path, "w") as file:
                file.write(rendered_content)

        # Generate CSV file with the decisions of all universes
        self._create_summary(valid_universes, keys)

        # Save forking paths
        with open(f"{self.results_dir}/forking_paths.pkl", "wb") as file:
            pickle.dump(forking_paths, file)

        # Set some attributes
        self.num_universes = len(valid_universes)
        self.forking_paths = forking_paths

        return

    def run(self, universe=None, parallel=1):
        """
        Run either an individual universe or the entire multiverse

        Parameters
        ----------
        universe : None, int, list, range
            Number of the universe to run. Default is None, which runs all universes

        parallel : int
            Number of universes to run in parallel
        """
        sorted_files = sorted(os.listdir(self.script_dir))

        # Delete previous results (.pkl files)
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isfile(item_path) and item_path.endswith('.pkl') and not item_path.endswith('forking_paths.pkl'):
                os.remove(item_path)

        # Function for parallel processing, called by joblib.delayed
        def execute_script(file):
            print(f"Running {file}")
            subprocess.run([sys.executable, os.path.join(self.script_dir, file)],
                        check=True, env=os.environ.copy())

        if universe is None:
            print("Starting multiverse analysis for all universes...")
            Parallel(n_jobs=parallel)(
                delayed(execute_script)(file)
                for file in sorted_files
                if file.endswith(".py") and not file.startswith("template")
            )
        else:
            # Subset of universes was chosen
            if isinstance(universe, int):
                universe_numbers = [universe]
            elif isinstance(universe, (list, tuple, range)):
                universe_numbers = list(universe)
            else:
                raise ValueError("universe_number should be None, an int, a list, tuple, or a range.")

            print(f"Starting analysis for universe(s): {universe_numbers}...")
            if parallel > 1:
                Parallel(n_jobs=parallel)(
                    delayed(execute_script)(f"universe_{num}.py")
                    for num in universe_numbers
                )
            else:
                for num in universe_numbers:
                    file = f"universe_{num}.py"
                    execute_script(file)

        # Save all results in a single dictionary
        self._combine_results()

        print("The multiverse analysis completed without any errors.")
        return

    def summary(self, universe=range(1,5), print_df=True):
        """
        Print the multiverse summary to the terminal/notebook

        Parameters
        ----------
        universe : int or range
            The universe number(s) to display. Default is range(1,5)
        """

        multiverse_summary = self._read_summary()

        if isinstance(universe, int):
            multiverse_selection = multiverse_summary.iloc[universe-1]
        elif isinstance(universe, range):
            multiverse_selection = multiverse_summary.iloc[universe.start-1:universe.stop]
        else:
            multiverse_selection = multiverse_summary

        if print_df:
            if self._in_notebook():
                from IPython.display import display
                display(multiverse_selection)
            else:
                print(multiverse_selection)

        return multiverse_summary

    def get_results(self, universe=None):
        """
        Get the results of the multiverse (or a specific universe) as a dictionary
        """
        path = f"{self.results_dir}/multiverse_results.pkl"

        with open(path, "rb") as file:
            results = pickle.load(file)

        if universe is not None:
            results = results[f"universe_{universe}"]            

        return results

    def visualize(self, universe=None, cmap="Set2", node_size=1500, figsize=(8,5), label_offset=0.04, exclude_single=False):
        """
        Visualize the multiverse as a network.

        Parameters
        ----------
        universe : int or None
            The universe to highlight in the network. If None or if the provided universe number
            is higher than available universes, the entire multiverse is shown without highlighting.
            Default is None.
        cmap : str
            Colormap to use for the nodes. Default is "Set2".
        node_size : int
            Size of the nodes. Default is 1500.
        figsize : tuple
            Size of the figure. Default is (8,5).
        label_offset : float
            Offset multiplier for decision labels.
        exclude_single : bool
            Whether to exclude parameters with only one unique option.
        """
        # Read the CSV summary into a DataFrame.
        multiverse_summary = self._read_summary()

        # Identify value and decision columns.
        value_columns = [col for col in multiverse_summary.columns if col.startswith("Value") or col == "Universe"]
        decision_columns = [col for col in multiverse_summary.columns if col.startswith("Decision")]

        # Create DataFrames for values and decisions.
        multiverse_values = multiverse_summary[value_columns]
        multiverse_decision = multiverse_summary[decision_columns]

        # Recursive function to add decision nodes and connect them hierarchically.
        def add_hierarchical_decision_nodes(G, root_node, parameters, level=0, exclude_single=exclude_single):
            if level >= len(parameters):
                return G  # No more parameters to process

            decision, options = parameters[level]
            # Only add the node if there are multiple options or if we are not excluding single-option parameters.
            if not exclude_single or len(options) > 1:
                for option in options:
                    node_name = f"{decision}: {option}"
                    G.add_node(node_name, level=level + 1, option=option, decision=decision)
                    G.add_edge(root_node, node_name)
                    # Recurse to add the next level.
                    G = add_hierarchical_decision_nodes(G, node_name, parameters, level + 1, exclude_single)
            else:
                # If excluding single-option parameters, skip to the next parameter.
                G = add_hierarchical_decision_nodes(G, root_node, parameters, level + 1, exclude_single)
            return G

        # Map each Value column (ignoring Universe) to its corresponding Decision column.
        # Assumes the order of Value columns (except Universe) matches the order of Decision columns.
        value_to_decision_map = {val: dec for val, dec in zip(value_columns[1:], decision_columns)}

        # Build parameters: For each Value column (except Universe) with more than one unique option,
        # use the decision name (from the map) and the unique values.
        parameters = [
            (value_to_decision_map.get(col, col), multiverse_values[col].unique())
            for col in multiverse_values.columns[1:]
        ]
        # Remove any NaN values from the unique options.
        parameters = [(dec, options[pd.notna(options)]) for dec, options in parameters]

        # Initialize the directed graph and add the root node.
        G = nx.DiGraph()
        root_node = "Start"
        G.add_node(root_node, level=0, label="Start")
        G = add_hierarchical_decision_nodes(G, root_node, parameters, exclude_single=exclude_single)

        # Build the list of node names that correspond to the specified universe.
        values = ["Start"]
        if universe is not None:
            filtered_df = multiverse_values[multiverse_values["Universe"] == f"Universe_{universe}"]
            # If the provided universe number is out-of-range, reset universe to None.
            if filtered_df.empty:
                universe = None
            else:
                row_dict = filtered_df.iloc[0].to_dict()
                for col, value in row_dict.items():
                    # Skip the Universe column.
                    if col == "Universe":
                        continue
                    # Convert the Value column name to its corresponding decision name.
                    decision_name = value_to_decision_map.get(col, col)
                    values.append(f"{decision_name}: {value}")

        # Determine which edges are part of the selected universe's path.
        universe_edges = [(src, tgt) for src, tgt in G.edges if src in values and tgt in values]
        edge_colors = []
        edge_widths = []
        for edge in G.edges:
            if edge in universe_edges:
                edge_colors.append("black")
                edge_widths.append(2.5)
            else:
                edge_colors.append("gray")
                edge_widths.append(1.0)

        # Create the figure and axis.
        fig, ax = plt.subplots(figsize=figsize)
        title_str = f"Universe {universe}" if universe is not None else "Multiverse"
        ax.set_title(title_str, size=14, fontweight="bold")

        # Use a multipartite layout based on the 'level' attribute.
        pos = nx.multipartite_layout(G, subset_key="level")

        # Create a colormap based on levels.
        levels = set(nx.get_node_attributes(G, "level").values())
        num_levels = len(levels)
        first_color = "lightgray"
        colormap = plt.cm.get_cmap(cmap, num_levels)
        # First level gets a fixed color; subsequent levels are derived from the colormap.
        colors = [first_color] + [colormap(i / (num_levels - 1)) for i in range(num_levels)]
        level_colors = {level: colors[i] for i, level in enumerate(sorted(levels))}

        # Draw edges.
        nx.draw(
            G,
            pos,
            with_labels=False,
            node_size=node_size - 10,
            node_color="white",
            arrows=True,
            edge_color=edge_colors,
            width=edge_widths,
            ax=ax,
        )

        # Draw nodes with colors based on their level.
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get("level") == level]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes_at_level,
                node_size=node_size,
                node_color=[level_colors[level] for _ in nodes_at_level],
                ax=ax,
            )

        # Prepare labels: For non-root nodes, use the 'option' text; for the root, use its label.
        node_labels = {}
        for node in G.nodes:
            if node != root_node and "option" in G.nodes[node]:
                node_labels[node] = str(G.nodes[node]["option"])
            else:
                node_labels[node] = G.nodes[node].get("label", node)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, ax=ax)

        # Calculate an offset based on the maximum number of nodes at any level.
        node_nums = [len([n for n in G.nodes if G.nodes[n].get("level") == level]) for level in levels]
        max_nodes = max(node_nums) if node_nums else 1

        # Annotate the bottom-most node at each level with the decision value.
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get("level") == level]
            if nodes_at_level:
                # Choose the node with the lowest y-position (bottom-most)
                bottom_node = min(nodes_at_level, key=lambda node: pos[node][1])
                if bottom_node != root_node and "decision" in G.nodes[bottom_node]:
                    decision = G.nodes[bottom_node]["decision"]
                    if decision in multiverse_decision.columns:
                        decision_value = multiverse_decision[decision].iloc[0]
                    else:
                        decision_value = decision
                    x, y = pos[bottom_node]
                    ax.text(
                        x,
                        y - label_offset * max_nodes,
                        decision_value,
                        horizontalalignment="center",
                        fontsize=12,
                        fontweight="bold",
                    )

        # Save the figure to the results directory.
        plt.savefig(f"{self.results_dir}/multiverse.png", bbox_inches="tight")
        return fig

    def specification_curve(self, measure, baseline=None, p_value=None, ci=None, smooth_ci=True, 
                          title="Specification Curve", name_map=None, cmap="Set3", linewidth=2, figsize=None, 
                          height_ratio=(2,1), fontsize=10, dotsize=50, ftype="png", dpi=72):
        """
        Create and save a specification curve plot from multiverse results

        Parameters
        ----------
        measure : string
            Name of the measure to plot. Needs to be provided

        baseline : float
            Plot baseline/chance level as a dashed line. Default is None

        p_value : float
            Calculate and visualize statistically significant specification via p-value. Default is None

        ci : int
            Confidence interval to plot. Default is 95

        smooth_ci : bool
            Plot a smoothed confidence interval. Default is True

        title : string
            Title of the plot. Default is "Specification Curve"

        name_map : dict
            Dictionary to map the decision names to custom names. Default is None

        cmap : string
            Colormap to use for the nodes. Default is "Set3"

        linewidth : int
            Width of the boxplots. Default is 2

        figsize : tuple
            Size of the figure. Default is None (will automatically try to determine the size).

        height_ratio : tuple
            Height ratio of the two subplots. Default is (2,1)

        fontsize : int
            Font size of the labels. Default is 10

        dotsize : int
            Size of the dots. Default is 50

        ftype : string
            File type to save the plot. Default is "png"

        dpi : int
            Dots per inch for the saved figure.
        """
        # Sort the universes based on an outcome measure and get the forking paths
        sorted_universes, forking_paths = self._load_and_prepare_data(measure)

        # Try to automatically determine the figure size
        if figsize is None:
            num_options = sum(len(values) for values in forking_paths.values())
            figsize = (max(8, len(sorted_universes)*0.07), max(6, num_options))
    
        # Plotting
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': height_ratio}, sharex=True)

        # Remove forking paths that only have a single option
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
        composite_labels = []
        display_labels = []
        flat_list = []
        yticks = []
        line_ends = []
        p_vals = []
        key_positions = {}
        y_max = 0
        space_between_groups = 1
        sig_color = "#018532"

        # Build the y-tick labels as composite strings for lookup but separate display labels.
        for decision, options in forking_paths.items():
            # Get custom label for the decision if provided
            group_label = name_map[decision] if (name_map is not None and decision in name_map) else decision
            key_position = y_max + len(options) / 2 - 0.5
            key_positions[group_label] = key_position

            for option in options:
                composite = f"{group_label}: {option}"  # composite used for lookup
                composite_labels.append(composite)
                flat_list.append((decision, option))
                yticks.append(y_max)
                display_labels.append(option)  # only the option will be displayed
                y_max += 1

            line_ends.append(y_max)
            y_max += space_between_groups

        # Save your decision_info using the composite mapping list.
        decision_info = (composite_labels, yticks, line_ends, list(forking_paths.keys()))

        # Setup bottom plot axes
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(display_labels, fontsize=fontsize)
        ax[1].tick_params(axis='y', labelsize=fontsize)
        ax[1].set_ylim(-1, y_max)
        ax[1].xaxis.grid(False)

        # Calculate padding for the lines and labels
        fig.canvas.draw()
        trans1 = transforms.blended_transform_factory(ax[1].transAxes, ax[1].transData)
        renderer = fig.canvas.get_renderer()
        
        # Find the minimum x position of a y label
        tick_label_extents = [label.get_window_extent(renderer=renderer) for label in ax[1].get_yticklabels()]
        max_extent = max(tick_label_extents, key=lambda bbox: bbox.width)
        x_start_pixel = max_extent.x0  # leftmost edge of the bounding box
        x_start_axes1 = ax[1].transAxes.inverted().transform((x_start_pixel, 0))[0]

        tick_label_extents = [label.get_window_extent(renderer=renderer) for label in ax[0].get_yticklabels()]
        max_extent = max(tick_label_extents, key=lambda bbox: bbox.width)
        x_start_pixel = max_extent.x0  # leftmost edge of the bounding box
        x_start_axes0 = ax[0].transAxes.inverted().transform((x_start_pixel, 0))[0]
        
        min_x_start_axes = min(x_start_axes1, x_start_axes0)
        padding = -0.3 * min_x_start_axes
        line_offset = min_x_start_axes - padding

        # Plot the key (decision group) labels on the left
        for key, pos in key_positions.items():
            ax[1].text(line_offset - padding, pos, key, transform=trans1, ha='right', va='center',
                    fontweight="bold", fontsize=fontsize, rotation=0)

        # Draw vertical lines separating decision groups on the bottom plot
        s = -0.5
        for i, line_end in enumerate(line_ends):
            e = line_end - 0.5
            line = mlines.Line2D([line_offset, line_offset], [s, e], color="black", lw=1, transform=trans1, clip_on=False)
            ax[1].add_line(line)
            s = line_end + 0.5

        # Setup variables for the CI and x-axis
        ci_lower_values = []
        ci_upper_values = []
        x_values = np.arange(len(sorted_universes))

        # Warn if few samples are available for t-test and CI
        result, decisions = sorted_universes[0]
        if hasattr(result, '__len__') and len(result) < 30:
            print(f"Warning: Only {len(result)} samples were available for the t-test and CI.")

        # Plot dots and confidence intervals for each universe in the top panel,
        # and the forking path markers in the bottom panel.
        for i, (result, decisions) in enumerate(sorted_universes):
            # Compute mean value
            mean_val = np.mean(result) if hasattr(result, '__len__') else result

            # Determine color based on p-value testing if provided
            color = "black"
            if p_value is not None:
                baseline = 0 if baseline is None else baseline
                if hasattr(result, '__len__'):
                    t_obs, p_obs = stats.ttest_1samp(result, baseline)
                    p_vals.append(p_obs)
                    color = sig_color if p_obs < p_value else "black"

            # Plot confidence interval for each universe if requested
            if ci is not None:
                if hasattr(result, '__len__') and len(result) > 3:
                    sem_val = np.std(result) / np.sqrt(len(result))
                    ci_lower = mean_val - sem_val * stats.t.ppf((1 + ci / 100) / 2., len(result) - 1)
                    ci_upper = mean_val + sem_val * stats.t.ppf((1 + ci / 100) / 2., len(result) - 1)
                    ci_lower_values.append(ci_lower)
                    ci_upper_values.append(ci_upper)
                    if not smooth_ci:
                        ax[0].plot([i, i], [ci_lower, ci_upper], color="gray", linewidth=linewidth)
                else:
                    ci_lower_values.append(mean_val)
                    ci_upper_values.append(mean_val)

            ax[0].scatter(i, mean_val, zorder=3, color=color, edgecolor=color, s=dotsize)

            # Determine colors for lower plot: each decision group gets its own color.
            group_ends = decision_info[2]
            num_groups = len(group_ends) + 1
            colormap_obj = plt.cm.get_cmap(cmap, num_groups)
            colors = [colormap_obj(j) for j in range(num_groups)]

            # Temporary solution to handle the new multiverse structure.
            # Build a dictionary of decisions and their corresponding options.
            formatted_decisions = {}
            formatted_decisions["Universe"] = decisions["Universe"]
            dc = 1  # Decision counter
            while f"Decision {dc}" in decisions and f"Value {dc}" in decisions:
                key = decisions[f"Decision {dc}"]
                value = decisions[f"Value {dc}"]
                formatted_decisions[key] = value
                dc += 1

            # For each decision in the current universe, plot its marker using the composite label.
            for decision, option in formatted_decisions.items():
                if decision in decision_info[3]:
                    # Build the composite label for lookup
                    group_label = name_map[decision] if (name_map is not None and decision in name_map) else decision
                    composite = f"{group_label}: {option}"
                    if composite in decision_info[0]:
                        index = decision_info[0].index(composite)
                        plot_pos = decision_info[1][index]

                        current_group = 0
                        for end in group_ends:
                            if plot_pos <= end:
                                break
                            current_group += 1
                        current_color = colors[current_group]
                        ax[1].scatter(i, plot_pos, color=current_color, marker='o', s=dotsize)

        # Plot the smooth CI band if required
        if smooth_ci and ci is not None:
            spline_lower = make_interp_spline(x_values, ci_lower_values, k=3)
            spline_upper = make_interp_spline(x_values, ci_upper_values, k=3)
            x_smooth = np.linspace(x_values.min(), x_values.max(), 500)
            ci_lower_smooth = spline_lower(x_smooth)
            ci_upper_smooth = spline_upper(x_smooth)
            ax[0].fill_between(x_smooth, ci_lower_smooth, ci_upper_smooth, color='gray', alpha=0.3)

        # Upper plot settings
        trans0 = transforms.blended_transform_factory(ax[0].transAxes, ax[0].transData)
        ax[0].set_title(title, fontweight="bold", fontsize=fontsize+2)
        ax[0].xaxis.grid(False)
        ax[0].set_xticks(np.arange(0, len(sorted_universes), 1))
        ax[0].set_xticklabels([])
        ax[0].set_xlim(-1, len(sorted_universes))

        ax[0].tick_params(axis='y', labelsize=fontsize)

        # Upper plot label for the measure
        ymin, ymax = ax[0].get_ylim()
        ycenter = (ymax + ymin) / 2
        measure_label = name_map[measure] if (name_map is not None and measure in name_map) else measure
        ax[0].text(line_offset - padding, ycenter, measure_label, transform=trans0, ha='right', va='center',
                fontweight="bold", fontsize=fontsize, rotation=0)
        line = mlines.Line2D([line_offset, line_offset], [ymin, ymax], color="black", lw=1, transform=trans0, clip_on=False)
        ax[0].add_line(line)

        # Build legend items
        legend_items = []
        y_lim = ax[0].get_ylim()
        if baseline is not None and y_lim[0] <= baseline <= y_lim[1]:
            ax[0].hlines(baseline, xmin=-2, xmax=len(sorted_universes) + 1, linestyles="--", lw=2, colors='black', zorder=1)
            legend_items.append(mlines.Line2D([], [], linestyle='--', color='black', linewidth=2, label="Baseline"))
        if hasattr(result, '__len__') and len(result) > 3 and ci is not None:
            legend_items.append(mpatches.Patch(facecolor='gray', edgecolor='white', label=f"{ci}% CI"))
        if p_value is not None:
            if len(p_vals) > 0:
                if min(p_vals) <= p_value:
                    legend_items.append(mlines.Line2D([], [], linestyle='None', marker='o', markersize=9,
                                                    markerfacecolor=sig_color, markeredgecolor=sig_color, label="p < 0.05"))
                if max(p_vals) > p_value:
                    legend_items.append(mlines.Line2D([], [], linestyle='None', marker='o', markersize=9,
                                                    markerfacecolor="black", markeredgecolor="black", label="p ≥ 0.05"))
            else:
                print("Warning: No p-values were calculated (less than 30 samples)")
        if legend_items:
            ax[0].legend(handles=legend_items, loc='upper left', fontsize=fontsize)

        # Save the plot with tight layout to avoid clipping
        plt.savefig(f"{self.results_dir}/specification_curve.{ftype}", bbox_inches='tight', dpi=dpi)
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

    def _create_summary(self, all_universes, keys):
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
        file_path = os.path.join(self.multiverse_dir, "multiverse_summary.csv")
        with open(file_path, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, combination in enumerate(all_universes, start=1):
                context = {'Universe': f"Universe_{i}"}

                # Populate the decision and value columns
                for j, (key, value) in enumerate(combination):
                    if isinstance(value, dict):
                        value = value.get('name', '')

                    context[f"Decision {j+1}"] = key
                    context[f"Value {j+1}"] = value

                writer.writerow(context)
        return

    def _read_summary(self):
        """
        Internal function: Reads the multiverse_summary.csv file

        Returns
        -------
        summary : pandas.DataFrame
            Pandas datframe containing the multiverse summary
        """
        summary_path = os.path.join(self.multiverse_dir, "multiverse_summary.csv")

        return pd.read_csv(summary_path)

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

    def _combine_results(self):
        """
        Internal function: Combines the results in a single dictionary and saves it as a pickle file
        """
        file_name = "multiverse_results.pkl"
        file_paths = glob.glob(os.path.join(self.script_dir, "temp", "universe_*.pkl"))
        
        if not file_paths:
            raise ValueError("No results files found. Please run the multiverse analysis first.")
        
        combined_results = {}
        
        for file_path in sorted(file_paths):
            universe_key = os.path.splitext(os.path.basename(file_path))[0]
            with open(file_path, 'rb') as f:
                result_dict = pickle.load(f)
            combined_results[universe_key] = result_dict
        
        # Save combined results file
        with open(os.path.join(self.results_dir, file_name), 'wb') as f:
            pickle.dump(combined_results, f)

        # Delete individual results files and folder
        for file_path in file_paths:
            os.remove(file_path)
        os.rmdir(os.path.join(self.script_dir, "temp"))

        return

    def _load_and_prepare_data(self, measure=None):
        """
        Internal function: Load and prepare the data for the specification curve plotting.

        Parameters
        ----------
        measure : str
            Name of the measure to plot.

        Returns
        -------
        sorted_universes : list
            List of tuples (data, summary_row) sorted by the mean value of the measure.
        forking_paths : dict
            Dictionary containing the forking paths.
        """
        # Load the summary CSV.
        summary_path = os.path.join(self.multiverse_dir, "multiverse_summary.csv")
        multiverse_summary = pd.read_csv(summary_path)

        # Load the forking paths.
        with open(os.path.join(self.results_dir, "forking_paths.pkl"), "rb") as file:
            forking_paths = pickle.load(file)

        # Load the combined results dictionary.
        combined_results_path = os.path.join(self.results_dir, "multiverse_results.pkl")
        if not os.path.exists(combined_results_path):
            raise ValueError("Combined results file not found. Please run the multiverse analysis first.")

        with open(combined_results_path, "rb") as file:
            combined_results = pickle.load(file)

        # Extract the specified measure for each universe.
        universe_data = {}
        for universe, result_dict in combined_results.items():
            if measure not in result_dict:
                raise ValueError(f"Measure '{measure}' not found in results for {universe}.")
            universe_data[universe] = result_dict[measure]

        # Build a list of tuples pairing each universe's measure data with its summary row.
        universes_with_summary = []
        for universe, data in universe_data.items():
            # Ensure matching is case-insensitive.
            summary_row = multiverse_summary[multiverse_summary['Universe'].str.lower() == universe.lower()]
            if not summary_row.empty:
                universes_with_summary.append((data, summary_row.iloc[0]))

        # Sort the universes by the mean of the measure values.
        sorted_universes = sorted(universes_with_summary, key=lambda x: np.mean(x[0]))

        return sorted_universes, forking_paths

    def _in_notebook(self):
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
