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

def in_notebook(self):
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:
                return False
        except Exception:
            return False
        return True

class Multiverse:
    def __init__(self, name="multiverse"):
        self.name = name

    # This function creates all the universe scripts
    def create(self, analysis_template, forking_paths):
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

        # Create Python scripts for each combination
        for i, combination in enumerate(all_universes, start=1):
            context = {key: self.format_type(value) for key, value in zip(keys, combination)}
            rendered_content = jinja_template.render(**context)

            # Write to Python script
            save_path = os.path.join(multiverse_dir, f"universe_{i}.py")
            with open(save_path, "w") as file:
                file.write(rendered_content)

        # Generate CSV file with the decisions of all universes
        self.create_csv(results_dir, all_universes, keys)

        # Save forking paths
        with open(f"{results_dir}/forking_paths.pkl", "wb") as file:
            pickle.dump(forking_paths, file)

    # This function can run all (or individual) universes
    def run(self, path=None, universe_number=None, parallel=1):
        if path is None:
            calling_script_dir = os.getcwd() if in_notebook else os.path.dirname(sys.path[0])
            multiverse_dir = os.path.join(calling_script_dir, self.name)
            os.makedirs(multiverse_dir, exist_ok=True)
            path = multiverse_dir

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

    def read_csv(self):
        calling_script_dir = os.getcwd() if in_notebook else os.path.dirname(sys.path[0])
        csv_path = os.path.join(calling_script_dir, f"{self.name}/results/multiverse_summary.csv")
        return pd.read_csv(csv_path)
    
    # Prints a summary of all universes
    def summary(self, universe=range(1,5)):
        multiverse_summary = self.read_csv()
        
        if isinstance(universe, int):
            multiverse_summary = multiverse_summary.iloc[universe-1]
        elif isinstance(universe, range):
            multiverse_summary = multiverse_summary.iloc[universe.start-1:universe.stop]

        if in_notebook:
            from IPython.display import display
            display(multiverse_summary)
        else:
            print(multiverse_summary + "\n")

    def visualize(self, universe=None, cmap="Set2"):
        multiverse_summary = self.read_csv()

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
        plt.figure(figsize=(8, 5))
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
        nx.draw(G, pos, with_labels=False, node_size=1490, node_color="white", arrows=True, edge_color=edge_colors, width=edge_widths)

        # Draw nodes with colors based on their level
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get('level') == level]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_at_level, node_size=1500, node_color=[level_colors[level] for _ in nodes_at_level])

        # Draw labels
        node_labels = {node: G.nodes[node]['option'] if node != root_node else G.nodes[node]['label'] for node in G.nodes} # Use only the option as a node label
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

        # Identify and annotate the bottom-most node at each level with the decision label
        levels = set(nx.get_node_attributes(G, 'level').values())
        for level in levels:
            nodes_at_level = [node for node in G.nodes if G.nodes[node].get('level') == level]
            if nodes_at_level:
                bottom_node = min(nodes_at_level, key=lambda node: pos[node][1])
                if bottom_node != root_node and 'decision' in G.nodes[bottom_node]:
                    decision = G.nodes[bottom_node]['decision']
                    x, y = pos[bottom_node]
                    plt.text(x, y - 0.25, decision, horizontalalignment='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()
    
    # This handles the types of the decision points to generate a working template script
    def format_type(self, value):
        if isinstance(value, str):
            return f"'{value}'"  # Strings are wrapped in quotes
        elif isinstance(value, int):
            return str(value)  # Integers are converted directly
        elif isinstance(value, float):
            return str(value)  # Floats are converted directly
        elif isinstance(value, bool):
            return "True" if value else "False" # Booleans are converted to their literal representations
        elif isinstance(value, dict):
            return self.handle_dict(value) # Dictionaries are handeled in a separate function
        elif isinstance(value, type):
            return value.__name__  # If the forking path is a class, we return the name of the class
        elif callable(value):     
            return value.__name__ # If the forking path is a function, we return the name of the function
        else:
            raise TypeError(f"Unsupported type for {value} which is of type {type(value)}")

    # This handles the decision points that require class/function imports
    def handle_dict(self, value):
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

    def create_csv(self, csv_path, all_universes, keys):
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
    
    def specification_curve(self, fname="multiverse_summary.csv", measure=None, cmap="Set2", ci=95, chance_level=None, linewidth=2, figsize=(16,9), height_ratio=[2,1], fontsize=10, dotsize=50, label_offset=-0.05):
        ###################################################################################################################
        # Load data and sort in ascending order for sequential plotting (specification curve is created from left to right)
        calling_script_dir = os.getcwd() if 'in_notebook' in globals() and in_notebook else os.path.dirname(sys.argv[0])
        results_path = os.path.join(calling_script_dir, f"{self.name}/results")
        csv_path = os.path.join(results_path, fname)
        multiverse_summary = pd.read_csv(csv_path)

        if measure in multiverse_summary.columns:
            print(f"Getting {measure} from .csv file")
            # Get forking paths/decisions from csv
            forking_paths = {}

            for column in multiverse_summary.columns:
                if column == measure:
                    continue

                unique_values = multiverse_summary[column].unique().tolist()
                forking_paths[column] = unique_values

            # Get results and corresponding decisions from csv
            universe_data = multiverse_summary[measure].values
            parameters = multiverse_summary.drop(columns=[measure])
            universes_with_summary = [(data, parameters.iloc[i].to_dict()) for i, data in enumerate(universe_data)]
            sorted_combined = sorted(universes_with_summary, key=lambda x: np.mean(x[0]))
            
        else:
            print(f"Getting {measure} from .pkl files")
            with open(f"{results_path}/forking_paths.pkl", "rb") as file:
                forking_paths = pickle.load(file)

            # Construct the search pattern to match files of the format 'universe_X.pkl'
            pattern = os.path.join(results_path, "universe_*.pkl")
            results_files = glob.glob(pattern)

            # Load and match universes to their summaries
            universe_data = {}
            for filename in results_files:
                # Extract the universe identifier from the filename
                universe = os.path.basename(filename).split('.')[0]

                # Load the universe data
                with open(filename, "rb") as file:
                    universe_data[universe] = pickle.load(file)

            # Create a list of tuples (universe_data, corresponding_row_in_summary)
            universes_with_summary = []
            for universe, data in universe_data.items():
                # Find the corresponding row in the summary DataFrame
                summary_row = multiverse_summary[multiverse_summary['Universe'].str.lower() == universe]
                if not summary_row.empty:
                    universes_with_summary.append((data, summary_row.iloc[0]))

            sorted_combined = sorted(universes_with_summary, key=lambda x: np.mean(x[0]))

        #############
        # Set up plot
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': height_ratio}, sharex=True)
        fig.suptitle('Multiverse Analysis', fontweight="bold", fontsize=fontsize+2)

        ######################################################################################################
        # Prepare decision labels: get only the name of the decision if its a dict and remove single parameters
        single_params = []
        for decision, options in forking_paths.items():
            for i, opt in enumerate(options):
                if isinstance(opt, dict):
                    options[i] = opt.get('name')
            
            if len(options) == 1:
                single_params.append(decision)

        for param in single_params:
            del forking_paths[param]

        ##########################################################
        # Plot decision labels, set up decision info for main plot
        flat_list = []
        yticks = []
        yticklabels = []
        current_position = 0
        line_ends = []
        key_positions = {}

        # Space to add after each group of decisions
        space_between_groups = 1  # Adjust this value as needed

        for key, options in forking_paths.items():
            # Calculate the position to annotate the decision key, centered over its options
            key_position = current_position + len(options) / 2 - 0.5
            key_positions[key] = key_position
            
            # Process each option in the current group
            for option in options:
                flat_list.append((key, option))
                yticks.append(current_position)
                yticklabels.append(option)
                current_position += 1  # Move to the next position for the next option

            line_ends.append(current_position)
            current_position += space_between_groups # Space after each group of decisions

        decision_info = (yticklabels, yticks, line_ends, forking_paths.keys())

        ###########################
        # Decision ticks and labels
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(yticklabels)
        ax[1].tick_params(axis='y', labelsize=10)
        ax[1].set_ylim(-1, current_position)
        ax[1].yaxis.grid(False)

        ####################
        # Plot decision keys
        trans1 = transforms.blended_transform_factory(ax[1].transAxes, ax[1].transData)

        for key, pos in key_positions.items():
            ax[1].text(label_offset - 0.01, pos, key, transform=trans1, ha='right', va='center', fontweight="bold", fontsize=fontsize, rotation=0)
        
        ##########################
        # Vertical parameter lines
        s=-0.5
        for i in range(len(line_ends)):  
            e = line_ends[i] - 0.5
            line = mlines.Line2D([label_offset, label_offset], [s, e], color="black", lw=1, transform=trans1, clip_on=False)
            ax[1].add_line(line)
            s = line_ends[i] + 0.5

        ####################
        # Plot each universe
        for i, (result, decisions) in enumerate(sorted_combined):
            ##########################
            # Specification curve + CI
            if hasattr(result, '__len__'):
                mean_val = np.mean(result)
            else:
                mean_val = result

            # If we have more than 3 values, we calculate the CI
            if hasattr(result, '__len__') and len(result) > 3:
                sem_val = np.std(result) / np.sqrt(len(result))
                ci_lower = mean_val - sem_val * stats.t.ppf((1 + ci / 100) / 2., len(result) - 1)
                ci_upper = mean_val + sem_val * stats.t.ppf((1 + ci / 100) / 2., len(result) - 1)

                # Plot CI
                ax[0].plot([i, i], [ci_lower, ci_upper], color="gray", linewidth=linewidth) # Draw CI
            
                # Plot mean measure value
                if ci_lower > 0.5:
                    ax[0].scatter(i, mean_val, zorder=3, color="green", edgecolor="green", s=dotsize)
                elif ci_upper < 0.5:
                    ax[0].scatter(i, mean_val, zorder=3, color="red", edgecolor="red", s=dotsize)
                else:
                    ax[0].scatter(i, mean_val, zorder=3, color="black", edgecolor="black", s=dotsize)
            
            # Less than 3 values, no CI
            else:
                if chance_level is not None:
                    if mean_val > chance_level:
                        ax[0].scatter(i, mean_val, zorder=3, color="green", edgecolor="green", s=dotsize)
                    elif mean_val < chance_level:
                        ax[0].scatter(i, mean_val, zorder=3, color="red", edgecolor="red", s=dotsize)
                else:
                    ax[0].scatter(i, mean_val, zorder=3, color="black", edgecolor="black", s=dotsize)

            #################
            # Decision points    
            group_ends = decision_info[2]
            num_groups = len(group_ends) + 1
            colormap = plt.cm.get_cmap(cmap, num_groups)
            colors = [colormap(i) for i in range(num_groups)]

            for decision, option in decisions.items():
                if option in decision_info[0] and decision in decision_info[3]:
                    index = decision_info[0].index(option)
                    plot_pos = decision_info[1][index]
                    
                    # Determine the current group based on plot_pos
                    current_group = 0
                    for end in group_ends:
                        if plot_pos <= end:
                            break
                        current_group += 1

                    current_color = colors[current_group]
                    
                    ax[1].scatter(i, plot_pos, color=current_color, marker='o', s=dotsize)

        #####################################################
        # Specification curve ticks and labels + chance level
        trans0 = transforms.blended_transform_factory(ax[0].transAxes, ax[0].transData)

        ax[0].xaxis.grid(False) 
        ax[0].set_xticks(np.arange(0,len(sorted_combined),1))
        ax[0].set_xticklabels([])
        ax[0].set_xlim(-1, len(sorted_combined)+1)

        ymin, ymax = ax[0].get_ylim()
        ycenter = (ymax + ymin) / 2
        ax[0].text(label_offset - 0.01, ycenter, measure, transform=trans0, ha='right', va='center', fontweight="bold", fontsize=fontsize, rotation=0)
        line = mlines.Line2D([label_offset, label_offset], [ymin, ymax], color="black", lw=1, transform=trans0, clip_on=False)
        ax[0].add_line(line)
        
        ax[0].hlines(chance_level, xmin=-2, xmax=len(sorted_combined)+1, linestyles="--", lw=2, colors='black', zorder=1)

        ########
        # Legend
        legend_items = []

        if hasattr(result, '__len__') and len(result) > 1:
            legend_items.append(mlines.Line2D([], [], linestyle='None', marker='o', markersize=8, markerfacecolor="black", markeredgecolor="black", label=f"Mean {measure}"))
        else:
            legend_items.append(mlines.Line2D([], [], linestyle='None', marker='o', markersize=8, markerfacecolor="black", markeredgecolor="black", label=f"{measure}"))
    
        if hasattr(result, '__len__') and len(result) > 3:
            legend_items.append(mpatches.Patch(facecolor='gray', edgecolor='white', label=f"{ci}% CI"))
        
        if chance_level is not None:
            legend_items.append(mlines.Line2D([], [], color='black', linestyle='--', lw=linewidth, label='Chance level'))
        
        ax[0].legend(handles=legend_items, loc='upper left', fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(f"{results_path}/specification_curve.png")