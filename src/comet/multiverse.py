import os
import re
import sys
import csv
import glob
import shutil
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
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from matplotlib import gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches as mpatches
from scipy.interpolate import make_interp_spline
from IPython.display import display as ipy_display
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

class Multiverse:
    """
    Multiverse class for creating, running, and visualizing a multiverse analysis.

    Parameters
    ----------
    name : str
        Name of the multiverse analysis. Default is "multiverse".
    path : str
        Path to a multiverse directory (only used by the GUI).
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

        config : dict
            Configuration dictionary with optional combination rules
            - order : list of lists specifying the order of decisions
            - exclude : list of list[dict or str] (set listed keys to NaN if conditions match)
            - remove : list of list[dict or str] (drop universes if conditions match)
            - deduplicate : bool (collapse duplicates after exclude/remove; default True)
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
        template_code = inspect.getsource(analysis_template)  # Extract the source code
        template_body = "\n".join(template_code.splitlines()[1:])  # Remove the function definition

        # Determine the indentation level of the first line of the function body and remove it from all lines
        first_line = template_body.splitlines()[0]
        indentation_level = len(first_line) - len(first_line.lstrip())
        adjusted_lines = [line[indentation_level:] if len(line) > indentation_level else line
                        for line in template_body.splitlines()]
        adjusted_template_body = "\n".join(adjusted_lines)

        # Create jinja template
        jinja_template = Template(adjusted_template_body)

        # Generate all unique combinations of forking paths
        if not forking_paths:
            print("No forking paths provided; nothing to create.")
            return
        keys, values = zip(*forking_paths.items())

        # Rules
        all_universes = []
        exclude_rules = config.get("exclude", [])
        removed_rules = config.get("remove", [])
        deduplicate = config.get("deduplicate", True)

        def _normalise_value(v):
            return v["name"] if isinstance(v, dict) and "name" in v else v

        def _apply_exclude_to_universe(universe):
            if not exclude_rules:
                return universe
            path_dict = {k: _normalise_value(v) for k, v in universe}
            to_nan = set()
            for rule in exclude_rules:
                matches = True
                keys_to_nan = []
                for condition in rule:
                    if isinstance(condition, dict):
                        for key, val in condition.items():
                            if path_dict.get(key) != val:
                                matches = False
                                break
                    elif isinstance(condition, str):
                        keys_to_nan.append(condition)
                    else:
                        matches = False
                        break
                if matches:
                    for k in keys_to_nan:
                        if k in path_dict:
                            to_nan.add(k)
            if not to_nan:
                return universe
            updated = []
            for k, v in universe:
                if k in to_nan:
                    updated.append((k, float("nan")))
                else:
                    updated.append((k, v))
            return updated

        def _rule_matches(universe, ruleset):
            path_dict = {k: _normalise_value(v) for k, v in universe}
            for rule in ruleset:
                if not isinstance(rule, list):
                    continue
                matches = True
                for condition in rule:
                    if isinstance(condition, dict):
                        for key, val in condition.items():
                            if path_dict.get(key) != val:
                                matches = False
                                break
                    elif isinstance(condition, str):
                        if condition not in path_dict:
                            matches = False
                            break
                    else:
                        matches = False
                        break
                if matches:
                    return True, rule
            return False, None

        # Build universes (apply exclude during creation)
        pre_dedup_count = 0
        dedup_removed = 0

        if config.get("order"):
            for order in config["order"]:
                reordered_values = [forking_paths[key] for key in order]
                unused_keys = [key for key in keys if key not in order]

                order_universes = []
                for combination in itertools.product(*reordered_values):
                    universe = list(zip(order, combination)) + [(k, "unused") for k in unused_keys]
                    universe = _apply_exclude_to_universe(universe)
                    order_universes.append(tuple(universe))

                pre_dedup_count += len(order_universes)

                # Deduplicate **within this order only**
                if deduplicate and order_universes:
                    unique, seen = [], set()

                    def _canon_sig(d):
                        def canon(x):
                            x = _normalise_value(x)
                            if isinstance(x, float) and (x != x):  # NaN
                                return ("NaN",)
                            return x
                        return tuple(sorted((k, canon(v)) for k, v in d.items()))

                    for u in order_universes:
                        sig = _canon_sig(dict(u))
                        if sig not in seen:
                            seen.add(sig)
                            unique.append(u)
                    order_universes = unique
                    dedup_removed += (len(order_universes) - len(unique))

                # Now append; note we do **not** run a global dedup later
                all_universes.extend(order_universes)
        
        else:
            for combination in itertools.product(*values):
                universe = list(zip(keys, combination))
                universe = _apply_exclude_to_universe(universe)
                all_universes.append(tuple(universe))

            pre_dedup_count = len(all_universes)
            dedup_removed = len(all_universes)

        # Exclusion summary with rule context (post-dedup for accurate counts)
        rule_context_counts = defaultdict(int)

        def _freeze_conditions(rule):
            # Extract dict conditions only; sort for stable printing
            conds = []
            for c in rule:
                if isinstance(c, dict):
                    for k, v in c.items():
                        conds.append((k, v))
            return tuple(sorted(conds))

        for u in all_universes:
            u_dict = dict(u)
            path_dict = {k: _normalise_value(v) for k, v in u}
            for rule in exclude_rules:
                conditions, keys_to_nan = [], []
                for c in rule:
                    if isinstance(c, dict):
                        conditions.append(c)
                    elif isinstance(c, str):
                        keys_to_nan.append(c)
                # match conditions?
                matches = True
                for cond in conditions:
                    for ck, cv in cond.items():
                        if path_dict.get(ck) != cv:
                            matches = False
                            break
                    if not matches:
                        break
                if not matches:
                    continue
                # count keys that are actually NaN in this universe
                cond_sig = _freeze_conditions(rule)
                for k in keys_to_nan:
                    if k in u_dict:
                        val = u_dict[k]
                        if isinstance(val, float) and (val != val):  # NaN
                            rule_context_counts[(cond_sig, k)] += 1

        if rule_context_counts:
            print("Exclusion summary")
            print("-----------------")
            
            if config:
                if config.get("order"):
                    print(f"Total number of universes: {pre_dedup_count} (includes ordering permutations)")
                else:
                    print(f"Total number of universes: {pre_dedup_count}")
            
            if dedup_removed:
                print(f"  - Removed {dedup_removed} duplicate universes (set 'deduplicate' to False if you want to keep them)")

            for (cond_sig, key), count in rule_context_counts.items():
                human_cond = "{" + ", ".join([f"'{k}': {repr(v)}" for k, v in cond_sig]) + "}"
                print(f"  - Set '{key}' to NaN for universes matching {human_cond} ({count} total).")

        # Remove universes that match 'remove' rules (printed last)
        if removed_rules:
            valid_universes = []
            removed_universes = []

            for combination in all_universes:
                is_invalid, matched_rule = _rule_matches(combination, removed_rules)
                if is_invalid:
                    removed_universes.append((combination, matched_rule))
                else:
                    valid_universes.append(combination)

            # Group removed universes by rule
            rule_to_universes = defaultdict(list)
            for universe, rule in removed_universes:
                rule_to_universes[str(rule)].append(dict(universe))

            print(f"  - Removed {len(removed_universes)} out of {len(all_universes)} remaining universes:")
            for i, (rule, universes) in enumerate(rule_to_universes.items(), 1):
                print(f"      Rule {rule} excluded {len(universes)} universes:")
                for u in universes:
                    print(f"      {u}")
        else:
            valid_universes = all_universes
        
        # Final number of universes
        if rule_context_counts:
            print(f"\n{len(valid_universes)} universes remain for analysis.")

        # Create Python scripts for each combination
        for i, combination in enumerate(valid_universes, start=1):
            combination_dict = dict(combination)

            # Smart formatting: only format dicts, pass raw types for strings/numbers
            context = {}
            for key in forking_paths.keys():
                val = combination_dict.get(key, None)
                # Map NaN to None for Jinja
                if isinstance(val, float) and (val != val):
                    context[key] = None
                else:
                    context[key] = self._format_type(val)

            rendered_content = jinja_template.render(**context)

            if config.get("order"):
                forking_dict_str = (
                    "# Ordering information was provided. The ordered decisions for this universe are:\n"
                    f"forking_paths = {combination_dict}\n\n"
                )
                rendered_content = forking_dict_str + rendered_content

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

    def run(self, universe=None, parallel=1, combine_results=True):
        """
        Run either an individual universe or the entire multiverse

        Parameters
        ----------
        universe : None, int, list, range
            Number of the universe to run. Default is None, which runs all universes

        parallel : int
            Number of universes to run in parallel
        """
        # Get all universe scripts
        scripts = [f for f in sorted(os.listdir(self.script_dir))
                     if f.endswith(".py") and not f.startswith("template")]

        # Delete previous results (.pkl files)
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isfile(item_path) and item_path.endswith('.pkl') and not item_path.endswith('forking_paths.pkl'):
                os.remove(item_path)

        # Function for parallel processing, called by joblib.delayed
        def execute_script(file):
            subprocess.run([sys.executable, os.path.join(self.script_dir, file)],
                        check=True, env=os.environ.copy())
        
        if universe is None:
            print("Starting multiverse analysis for all universes...")
            
            if parallel == 1:
                for file in tqdm(scripts):
                    execute_script(file)
            else:
                with tqdm_joblib(total=len(scripts), desc="Performing multiverse analysis:") as progress:
                    Parallel(n_jobs=parallel)(delayed(execute_script)(file) for file in scripts)
        else:
            # Subset of universes was chosen
            if isinstance(universe, int):
                universe_numbers = [universe]
            elif isinstance(universe, (list, tuple, range)):
                universe_numbers = list(universe)
            else:
                raise ValueError("universe_number should be None, an int, a list, tuple, or a range.")
            
            selected_universes = [f for f in scripts if any(f.endswith(f"universe_{u}.py") for u in universe_numbers)]
            print(f"Starting analysis for universe(s): {universe_numbers}...")
            
            if parallel == 1:
                for file in tqdm(selected_universes):
                    execute_script(file)
            else:
                with tqdm_joblib(total=len(selected_universes), desc="Performing multiverse analysis:") as progress:
                    Parallel(n_jobs=parallel)(delayed(execute_script)(file) for file in selected_universes)

        # Save all results in a single dictionary
        if combine_results:
            self._combine_results()
            self.combined_results = True
        else:
            self.combined_results = False

        print("The multiverse analysis completed without any errors.")
        return

    def summary(self, universe=None, print_df=True, return_df=False):
        """
        Print the multiverse summary to the terminal/notebook

        Parameters
        ----------
        universe : int, range, or None
            The universe number(s) to display. Default is None (prints the head)
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

        return multiverse_selection if return_df else None

    def get_results(self, universe=None, as_df=False):
        """
        Get the results of the multiverse (or a specific universe).

        Parameters
        ----------
        universe : int | None
            If given, return results for that specific universe.
        as_df : bool
            False returns the raw dict (default).
            True returns a pandas DataFrame (only valid when universe is None).
        """
        if not isinstance(as_df, bool):
            raise ValueError("as_df must be a boolean")
        
        if os.path.exists(f"{self.results_dir}/multiverse_results.pkl"):
            path = f"{self.results_dir}/multiverse_results.pkl"

            with open(path, "rb") as file:
                results = pickle.load(file)

            if universe is not None:
                results = results[f"universe_{universe}"]
                return results

            if as_df:
                df = pd.DataFrame.from_dict(results, orient="index")

                # keep string universe label as a column
                df.insert(0, "universe", df.index)

                # integer index
                df.index = (
                    df["universe"]
                    .str.replace("universe_", "", regex=False)
                    .astype(int)
                )
                df.index.name = None

                return df.sort_index()

            return results

        else:
            if universe is None:
                raise ValueError(
                    "Multiverse results are not combined. Please specify a universe number."
                )

            path = f"{self.results_dir}/tmp/universe_{universe}.pkl"

            with open(path, "rb") as file:
                results = pickle.load(file)

            if as_df:
                df = pd.DataFrame([results])
                df.insert(0, "universe", f"universe_{universe}")
                df.index = pd.Index([int(universe)], name="universe_id")
                return df

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

        return self._handle_figure_returns(fig)

    def specification_curve(
        self,
        measure: str,
        baseline: float | None = None,
        p_value: float | str | bool | None = None,
        ci: int | str | bool | None = None,
        smooth_ci: bool = True,
        title: str | None = None,
        name_map: dict | None = None,
        cmap: str = "Set3",
        linewidth: float = 2,
        figsize: tuple | None = None,
        height_ratio: tuple = (2, 1),
        fontsize: int = 10,
        dotsize: int = 50,
        line_pad: float = 0.3,
        ftype: str = "pdf",
        dpi: int = 300,
        p_threshold: float = 0.05,
        ci_level_default: int = 95,
    ):
        """
        Create and save a specification curve plot from multiverse results (df-based).

        Supports significance and CI either computed from per-universe samples in `measure`
        (when `p_value`/`ci` are bool/float/int) or read from columns (when `p_value`/`ci`
        are strings naming columns).

        Notes
        -----
        - If `p_value` is float or True, `measure` must contain list/array samples per universe.
        - If `ci` is int or True, `measure` must contain list/array samples per universe.
        - If `p_value` is a string, it is interpreted as a p-value column (numeric) or a
        significance flag (bool).
        - If `ci` is a string, it must contain per-universe (lower, upper) bounds.

        Returns
        -------
        Any
            Whatever ``self._handle_figure_returns(fig)`` returns.
        """
        def _map_name(key: str) -> str:
            return name_map.get(key, key) if isinstance(name_map, dict) else key

        def _extract_decision_order(decisions_obj) -> list[str]:
            if not isinstance(decisions_obj, dict):
                return []
            dec_block = decisions_obj.get("decisions")
            d = dec_block if isinstance(dec_block, dict) else decisions_obj
            order = []
            i = 1
            while f"Decision {i}" in d:
                order.append(str(d[f"Decision {i}"]))
                i += 1
            return order

        # ------------------------------------------------------------
        # Load df and basic checks
        df = self.get_results(as_df=True).copy()

        if measure not in df.columns:
            raise ValueError(f"'{measure}' not found in multiverse results.")
        if "decisions" not in df.columns:
            raise ValueError("Expected a 'decisions' column containing decision dictionaries.")

        # Keep raw before scalarising (for CI / p-value computation)
        raw_measure = df[measure].copy()

        # Scalarise outcome for plotting (mean over list/array; scalar -> float)
        df[measure] = df[measure].apply(
            lambda x: float(np.mean(x)) if isinstance(x, (list, tuple, np.ndarray)) else float(x)
        )
        if df[measure].isna().any():
            raise ValueError(f"NaNs detected in '{measure}' after reduction to mean.")

        # ------------------------------------------------------------
        # Flatten decisions -> columns
        flat = df["decisions"].map(self._flatten_decisions)
        dec_df = pd.DataFrame(list(flat), index=df.index)
        if dec_df.shape[1] == 0:
            raise ValueError("Could not extract any decisions from the 'decisions' dicts.")

        # decision-group order from stored Decision 1..N
        decision_order = _extract_decision_order(df.iloc[0]["decisions"])
        decision_order = [d for d in decision_order if d in dec_df.columns]
        leftovers = [c for c in dec_df.columns if c not in decision_order]
        decision_cols_all = decision_order + leftovers

        df = pd.concat([df.drop(columns=["decisions"]), dec_df], axis=1)

        # Keep only decisions with >1 unique option
        decision_cols = [c for c in decision_cols_all if df[c].nunique(dropna=True) > 1]
        if len(decision_cols) == 0:
            raise ValueError("No decisions with more than one option found; nothing to plot in the bottom panel.")

        # ------------------------------------------------------------
        # Sort by outcome
        sort_idx = df[measure].sort_values().index
        df = df.loc[sort_idx].reset_index(drop=True)
        raw_measure_sorted = raw_measure.loc[sort_idx].reset_index(drop=True)  # <-- aligned raw samples
        n = len(df)
        x_values = np.arange(n)
        y_values = df[measure].to_numpy(dtype=float)

        # ------------------------------------------------------------
        # Significance handling
        baseline_for_tests = 0.0 if baseline is None else float(baseline)
        sig_color = "#018532"

        significant = np.zeros(n, dtype=bool)
        pvals = None

        if p_value is None:
            pass

        elif isinstance(p_value, str):
            if p_value not in df.columns:
                raise ValueError(f"p_value column '{p_value}' not found in results df.")
            col = df[p_value]
            if pd.api.types.is_bool_dtype(col):
                significant = col.fillna(False).to_numpy(dtype=bool)
            else:
                pvals = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
                if np.isnan(pvals).any():
                    raise ValueError(f"NaNs detected in p-value column '{p_value}'.")
                significant = pvals < float(p_threshold)

        else:
            thr = float(p_threshold) if p_value is True else float(p_value)

            pvals_list = []
            for rv in raw_measure_sorted:
                if isinstance(rv, (list, tuple, np.ndarray)):
                    arr = np.asarray(rv, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if len(arr) >= 2:
                        _, p = stats.ttest_1samp(arr, baseline_for_tests)
                        pvals_list.append(float(p))
                    else:
                        pvals_list.append(np.nan)
                else:
                    pvals_list.append(np.nan)

            pvals = np.asarray(pvals_list, dtype=float)
            if np.isnan(pvals).any():
                raise ValueError(
                    "Cannot compute p-values for some universes: "
                    f"'{measure}' must contain list/array samples per universe when p_value is float/True."
                )
            significant = pvals < thr

        top_colors = np.where(significant, sig_color, "black")

        # ------------------------------------------------------------
        # CI handling
        ci_lower = None
        ci_upper = None

        if ci is None:
            pass

        elif isinstance(ci, str):
            if ci not in df.columns:
                raise ValueError(f"CI column '{ci}' not found in results df.")
            bounds = df[ci].to_list()
            lows, highs = [], []
            for b in bounds:
                if isinstance(b, (list, tuple, np.ndarray)) and len(b) == 2:
                    lo, hi = float(b[0]), float(b[1])
                    if not (np.isfinite(lo) and np.isfinite(hi)):
                        raise ValueError(f"Non-finite CI bounds in column '{ci}'.")
                    lows.append(lo)
                    highs.append(hi)
                else:
                    raise ValueError(f"CI column '{ci}' must contain (lower, upper) tuples/lists per universe.")
            ci_lower = np.asarray(lows, dtype=float)
            ci_upper = np.asarray(highs, dtype=float)

        else:
            level = int(ci_level_default) if ci is True else int(ci)

            lows, highs = [], []
            for rv, mean_val in zip(raw_measure_sorted, y_values):
                if not isinstance(rv, (list, tuple, np.ndarray)):
                    raise ValueError(
                        f"Cannot compute CI because '{measure}' does not contain per-universe samples. "
                        "Provide a CI column instead (ci='colname')."
                    )
                arr = np.asarray(rv, dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) < 4:
                    raise ValueError(
                        f"Cannot compute CI for a universe with <4 finite samples in '{measure}'. "
                        "Provide a CI column instead (ci='colname') or store more samples."
                    )

                sem = np.std(arr, ddof=1) / np.sqrt(len(arr))
                half = sem * stats.t.ppf((1.0 + level / 100.0) / 2.0, len(arr) - 1)
                lows.append(float(mean_val - half))
                highs.append(float(mean_val + half))

            ci_lower = np.asarray(lows, dtype=float)
            ci_upper = np.asarray(highs, dtype=float)

        # ------------------------------------------------------------
        # Figure size (auto if None)
        if title is None:
            title = "Specification Curve"

        if figsize is None:
            num_options = sum(df[c].nunique(dropna=True) for c in decision_cols)
            figsize = (max(8, n * 0.07), max(6, num_options * 0.35))

        # ------------------------------------------------------------
        # Bottom panel layout (decision order enforced; options order-of-appearance)
        decision_positions = {}
        display_labels = []
        yticks = []
        line_ends = []
        key_positions = {}
        y_max = 0
        space_between_groups = 1

        num_groups = len(decision_cols)
        cmap_obj = plt.cm.get_cmap(cmap, num_groups if num_groups > 0 else 1)
        group_colors = [cmap_obj(i) for i in range(num_groups)]

        for group_idx, decision in enumerate(decision_cols):
            options = pd.unique(df[decision].astype("object"))
            options = [o for o in options if pd.notna(o)]

            group_label = _map_name(decision)
            key_positions[group_label] = y_max + len(options) / 2.0 - 0.5

            for opt in options:
                yticks.append(y_max)
                display_labels.append(str(opt))
                decision_positions[(decision, opt)] = (y_max, group_idx)
                y_max += 1

            line_ends.append(y_max)
            y_max += space_between_groups

        # ------------------------------------------------------------
        # Plot
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": height_ratio}, sharex=True
        )

        # Bottom axes setup
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(display_labels, fontsize=fontsize)
        ax[1].tick_params(axis="y", labelsize=fontsize)
        ax[1].set_ylim(-1, y_max)
        ax[1].xaxis.grid(False)
        ax[1].invert_yaxis()

        # Left padding for group labels/lines
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        trans1 = transforms.blended_transform_factory(ax[1].transAxes, ax[1].transData)
        tick_extents = [lbl.get_window_extent(renderer=renderer) for lbl in ax[1].get_yticklabels()]
        max_extent = max(tick_extents, key=lambda bb: bb.width)
        x_start_pixel = max_extent.x0
        x_start_axes1 = ax[1].transAxes.inverted().transform((x_start_pixel, 0))[0]

        trans0 = transforms.blended_transform_factory(ax[0].transAxes, ax[0].transData)
        tick_extents0 = [lbl.get_window_extent(renderer=renderer) for lbl in ax[0].get_yticklabels()]
        if tick_extents0:
            max_extent0 = max(tick_extents0, key=lambda bb: bb.width)
            x_start_pixel0 = max_extent0.x0
            x_start_axes0 = ax[0].transAxes.inverted().transform((x_start_pixel0, 0))[0]
        else:
            x_start_axes0 = x_start_axes1

        min_x_start_axes = min(x_start_axes1, x_start_axes0)
        padding = -line_pad * min_x_start_axes
        line_offset = min_x_start_axes - padding

        # Group labels + separators
        for key, pos in key_positions.items():
            ax[1].text(
                line_offset - padding, pos, key, transform=trans1,
                ha="right", va="center", fontweight="bold", fontsize=fontsize
            )

        s = -0.5
        for line_end in line_ends:
            e = line_end - 0.5
            ax[1].add_line(
                mlines.Line2D([line_offset, line_offset], [s, e], color="black", lw=1, transform=trans1, clip_on=False)
            )
            s = line_end + 0.5

        # Top scatter
        ax[0].scatter(x_values, y_values, c=top_colors, s=dotsize, edgecolors=top_colors, zorder=3)

        # CI drawing
        if ci_lower is not None and ci_upper is not None:
            if smooth_ci and len(ci_lower) >= 4:
                spline_lo = make_interp_spline(x_values, ci_lower.astype(float), k=3)
                spline_hi = make_interp_spline(x_values, ci_upper.astype(float), k=3)
                x_smooth = np.linspace(x_values.min(), x_values.max(), 500)
                ax[0].fill_between(x_smooth, spline_lo(x_smooth), spline_hi(x_smooth), color="gray", alpha=0.3)
            else:
                for i in range(n):
                    ax[0].plot([i, i], [ci_lower[i], ci_upper[i]], color="gray", linewidth=linewidth, zorder=2)

        # Baseline line
        legend_items = []
        if baseline is not None:
            ax[0].hlines(float(baseline), xmin=-2, xmax=n + 1, linestyles="--", lw=2, colors="black", zorder=1)
            legend_items.append(mlines.Line2D([], [], linestyle="--", color="black", linewidth=2, label="Baseline"))

        # Measure label + left line
        ymin, ymax = ax[0].get_ylim()
        ycenter = (ymin + ymax) / 2.0
        ax[0].text(
            line_offset - padding, ycenter, _map_name(measure), transform=trans0,
            ha="right", va="center", fontweight="bold", fontsize=fontsize
        )
        ax[0].add_line(
            mlines.Line2D([line_offset, line_offset], [ymin, ymax], color="black", lw=1, transform=trans0, clip_on=False)
        )

        # Bottom markers (vectorised melt)
        long = df[decision_cols].reset_index().melt(
            id_vars="index", value_vars=decision_cols, var_name="decision", value_name="option"
        ).rename(columns={"index": "x"})

        bottom_x, bottom_y, bottom_c = [], [], []
        for row in long.itertuples(index=False):
            opt = row.option
            if pd.isna(opt):
                continue
            key = (row.decision, opt)
            if key in decision_positions:
                y_pos, grp_idx = decision_positions[key]
                bottom_x.append(row.x)
                bottom_y.append(y_pos)
                bottom_c.append(group_colors[grp_idx] if num_groups > 0 else "black")

        if bottom_x:
            ax[1].scatter(np.asarray(bottom_x), np.asarray(bottom_y), c=np.asarray(bottom_c), s=dotsize, marker="o")

        # Top panel styling
        ax[0].set_title(title, fontweight="bold", fontsize=fontsize + 2)
        ax[0].xaxis.grid(False)
        ax[0].set_xticks([])
        ax[0].set_xlim(-1, n)
        ax[0].tick_params(axis="y", labelsize=fontsize)

        # Legend for CI
        if ci_lower is not None and ci_upper is not None:
            if isinstance(ci, int):
                ci_lab = f"{ci}% CI"
            elif ci is True:
                ci_lab = f"{ci_level_default}% CI"
            else:
                ci_lab = "CI"
            legend_items.append(mpatches.Patch(facecolor="gray", edgecolor="white", label=ci_lab))

        # Legend for significance (only if p_value requested)
        if p_value is not None:
            if significant.any():
                legend_items.append(
                    mlines.Line2D([], [], linestyle="None", marker="o", markersize=9,
                                markerfacecolor=sig_color, markeredgecolor=sig_color,
                                label="significant")
                )
            if (~significant).any():
                legend_items.append(
                    mlines.Line2D([], [], linestyle="None", marker="o", markersize=9,
                                markerfacecolor="black", markeredgecolor="black",
                                label="not significant")
                )

        if legend_items:
            ax[0].legend(handles=legend_items, loc="upper left", fontsize=fontsize, frameon=False)

        plt.savefig(f"{self.results_dir}/specification_curve2.{ftype}", bbox_inches="tight", dpi=dpi)
        sns.reset_orig()
        return self._handle_figure_returns(fig)

    def multiverse_plot(
        self,
        measure: str,
        n_bins: int = 20,
        sig_col: str | None = None,
        sig_threshold: float = 0.05,
        baseline: float | None = None,
        name_map: dict | None = None,
        figsize: tuple = (7, 9),
        ftype: str = "pdf",
        dpi: int = 300,
    ):
        """
        Multiverse plot as introduced by Krähmer & Young (2026).

        Visualises the distribution of multiverse outcomes together with decision-wise
        heatmap strips showing how different analytic choices relate to the outcome.
        For each decision level, the average change in the outcome relative to the
        reference level is shown on the right.

        Decision-group order
        --------------------
        The *vertical order of decision groups* (strips) follows the original COMET
        forking-path order: "Decision 1", "Decision 2", ... as stored in the decisions
        dicts. This ensures that plots match the order in which decisions were defined
        (e.g., software -> resampling -> stimulation -> electrode).

        Parameters
        ----------
        measure : str
            Name of the outcome/measure column in the multiverse results.
            Entries may be scalars or lists/arrays (in which case the mean is used).
        n_bins : int, optional
            Number of bins used to discretise the outcome axis for the heatmap strips.
        sig_col : str | None, optional
            Column indicating statistical significance. If provided:
            - boolean values are interpreted directly (True = significant),
            - numeric values are compared against ``sig_threshold``.
            If None, no significance overlay is drawn.
        sig_threshold : float, optional
            Threshold used when ``sig_col`` is numeric (default is 0.05).
        baseline : float | None, optional
            Optional baseline value for the outcome. If provided, a vertical dashed
            reference line is drawn at this value in the density plot.
        name_map : dict | None, optional
            Optional mapping for display names. Keys may include the measure name
            and decision names. Values are the desired display labels.
        figsize : tuple, optional
            Figure size passed to Matplotlib (width, height) in inches.
        ftype : str, optional
            File type used when saving the figure (e.g., ``"pdf"``, ``"png"``).
        dpi : int, optional
            Resolution (dots per inch) used when saving the figure.

        Returns
        -------
        Any
            Whatever ``self._handle_figure_returns(fig)`` returns.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap
        from scipy import stats

        # Helpers
        def _map_name(key: str) -> str:
            return name_map.get(key, key) if isinstance(name_map, dict) else key

        def _kde_density(values, grid, x_min, x_max):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if len(values) < 2:
                hist, edges = np.histogram(values, bins=20, range=(x_min, x_max), density=True)
                mids = (edges[:-1] + edges[1:]) / 2
                return np.interp(grid, mids, hist, left=0, right=0)
            return stats.gaussian_kde(values)(grid)

        def _build_heatmap_data(df_in, varname, outcome_var, breaks):
            tmp = df_in[[varname, outcome_var]].copy()
            tmp["outcome_bin"] = pd.cut(tmp[outcome_var], bins=breaks, include_lowest=True)

            all_bins = tmp["outcome_bin"].cat.categories
            levels = tmp[varname].cat.categories

            idx = pd.MultiIndex.from_product([all_bins, levels], names=["outcome_bin", varname])
            counts = (
                tmp.groupby(["outcome_bin", varname], observed=False)
                .size()
                .reindex(idx, fill_value=0)
                .reset_index(name="n")
            )

            total_by_bin = counts.groupby("outcome_bin")["n"].transform(lambda x: x.sum() if x.sum() > 0 else 1)
            counts["prop"] = counts["n"] / total_by_bin

            bin_to_idx = {iv: i for i, iv in enumerate(all_bins)}
            counts["bin_idx"] = counts["outcome_bin"].map(bin_to_idx)
            return counts

        def _extract_decision_order(decisions_obj) -> list[str]:
            """
            Return ["<Decision 1 name>", "<Decision 2 name>", ...] from COMET decisions storage.
            Supports two schemas:
            1) row["decisions"] is the decisions block itself
            2) row["decisions"] is a result dict containing {"decisions": {...}}
            """
            if not isinstance(decisions_obj, dict):
                return []

            # schema 2: {"decisions": {...}}
            dec_block = decisions_obj.get("decisions")
            if isinstance(dec_block, dict):
                d = dec_block
            else:
                # schema 1: already the block
                d = decisions_obj

            order = []
            i = 1
            while f"Decision {i}" in d:
                order.append(str(d[f"Decision {i}"]))
                i += 1
            return order

        # Load results
        df = self.get_results(as_df=True)

        if measure not in df.columns:
            raise ValueError(
                f"'{measure}' not found in multiverse results. Make sure to save it in the multiverse template."
            )
        if "decisions" not in df.columns:
            raise ValueError("Expected a 'decisions' column containing decision dictionaries.")
        if n_bins > len(df):
            raise ValueError(f"n_bins ({n_bins}) cannot be higher than the number of universes ({len(df)}).")

        # Significance handling
        if sig_col is not None:
            if sig_col not in df.columns:
                raise ValueError(f"sig_col='{sig_col}' not found in results df.")

            s = df[sig_col]
            if pd.api.types.is_bool_dtype(s):
                significant = s.fillna(False).astype(bool)
            else:
                s_num = pd.to_numeric(s, errors="coerce")
                significant = (s_num < sig_threshold).fillna(False)

            df = df.copy()
            df["significant"] = significant
        else:
            df = df.copy()
            df["significant"] = False

        # Scalarise outcome: mean over list/array; scalar -> float
        df[measure] = df[measure].apply(
            lambda x: float(np.mean(x)) if isinstance(x, (list, tuple, np.ndarray)) else float(x)
        )
        if df[measure].isna().any():
            raise ValueError(f"NaNs detected in '{measure}' after reduction to mean.")

        # Flatten decisions into columns
        flat = df["decisions"].map(self._flatten_decisions)
        dec_only_df = pd.DataFrame(list(flat), index=df.index)
        if dec_only_df.shape[1] == 0:
            raise ValueError("Could not extract any decisions from the 'decisions' dicts.")

        # Enforce decision-group order from "Decision 1..N"
        decision_order = _extract_decision_order(df.iloc[0]["decisions"])
        decision_order = [d for d in decision_order if d in dec_only_df.columns]
        leftovers = [c for c in dec_only_df.columns if c not in decision_order]
        decisions_list = decision_order + leftovers

        # Reorder decision columns and merge
        dec_only_df = dec_only_df.reindex(columns=decisions_list)
        df = pd.concat([df.drop(columns=["decisions"]), dec_only_df], axis=1)

        # Options within each decision: keep order-of-appearance (do NOT reverse)
        for d in decisions_list:
            s = df[d].astype("object")
            cats = list(pd.unique(s))
            df[d] = pd.Categorical(s, categories=cats, ordered=True)

        # Average increase labels (mean difference vs reference = first option)
        avg_diff_lookup = {}
        for varname in decisions_list:
            levels_all = list(df[varname].cat.categories)
            if len(levels_all) == 0:
                continue
            ref = levels_all[0]
            ref_mean = float(df.loc[df[varname] == ref, measure].mean())

            lvl_map = {}
            for lvl in levels_all:
                if lvl == ref:
                    lvl_map[lvl] = "Ref."
                else:
                    lvl_mean = float(df.loc[df[varname] == lvl, measure].mean())
                    lvl_map[lvl] = f"{(lvl_mean - ref_mean):+.2f}"
            avg_diff_lookup[varname] = lvl_map

        # Binning & density
        multiverse_outcome = df[measure].to_numpy(dtype=float)
        x_min = float(np.min(multiverse_outcome))
        x_max = float(np.max(multiverse_outcome))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            raise ValueError(f"'{measure}' must have finite, non-constant values.")

        x_pad = 0.02 * np.ptp(multiverse_outcome)
        common_xlim = (x_min - x_pad, x_max + x_pad)

        breaks = np.linspace(x_min, x_max, n_bins + 1)
        grid_x = np.linspace(x_min, x_max, 1000)

        y_all = _kde_density(multiverse_outcome, grid_x, x_min, x_max)

        if df["significant"].any():
            sig_vals = df.loc[df["significant"], measure].to_numpy(dtype=float)
            y_sig = _kde_density(sig_vals, grid_x, x_min, x_max)
            sig_share = float(df["significant"].mean())
            y_sig_scaled = np.minimum(y_sig * sig_share, y_all)
        else:
            y_sig_scaled = None

        # Plot layout
        DENSITY_RATIO = 0.5
        strip_levels_counts = [len(df[v].cat.categories) for v in decisions_list]
        total_strip_levels = sum(strip_levels_counts)
        density_height = max(1, int(DENSITY_RATIO * total_strip_levels))
        n_rows = density_height + sum([2 + c for c in strip_levels_counts]) + 1

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.0)
        current_row = 0

        tab10 = plt.get_cmap("tab10").colors
        base_colors = {v: tab10[i % len(tab10)] for i, v in enumerate(decisions_list)}

        # Density panel
        ax_density = fig.add_subplot(gs[current_row : current_row + density_height, 0])
        current_row += density_height

        if baseline is not None:
            ax_density.plot([baseline, baseline], [0, float(np.max(y_all))], linestyle="--", linewidth=1, color="black")

        ax_density.plot(grid_x, y_all, label="All", linewidth=1.5)

        if y_sig_scaled is not None:
            ax_density.fill_between(grid_x, y_sig_scaled, alpha=0.4, label="Significant", color="tomato")

        ax_density.set_xlim(*common_xlim)
        ax_density.set_yticks([])
        ax_density.set_xticks([])
        ax_density.legend(frameon=False)
        ax_density.grid(False)
        ax_density.set_frame_on(False)
        ax_density.plot([common_xlim[0], common_xlim[1]], [0, 0], linewidth=1, color="black")

        # Strips (decision-group order enforced; options unchanged)
        x_label_pos = common_xlim[1] + 0.01 * (x_max - x_min)

        for varname in decisions_list:
            levels = list(df[varname].cat.categories)  # keep option order
            n_levels = len(levels)
            n_rows_strip = 2 + n_levels  # blank + header + levels

            ax = fig.add_subplot(gs[current_row : current_row + n_rows_strip, 0])
            current_row += n_rows_strip

            counts = _build_heatmap_data(df, varname, measure, breaks)

            H = np.zeros((n_levels + 2, len(breaks) - 1))
            for i_level, lvl in enumerate(levels):
                sub = counts[counts[varname] == lvl]
                prop_by_bin = dict(zip(sub["bin_idx"], sub["prop"]))
                for b in range(len(breaks) - 1):
                    H[i_level + 2, b] = prop_by_bin.get(b, 0.0)

            cmap = LinearSegmentedColormap.from_list("white_to_color", [(1, 1, 1), base_colors[varname]])
            X = breaks
            Y = np.arange(n_levels + 3)  # edges
            ax.pcolormesh(X, Y, H, shading="flat", cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_ylim(n_levels + 2, 0)

            # Right-side mean diffs
            for i_level, lvl in enumerate(levels):
                lab = avg_diff_lookup.get(varname, {}).get(lvl, "")
                ax.text(x_label_pos, i_level + 2.5, lab, va="center", ha="left", fontsize=8, clip_on=False)

            # y ticks
            ytick_pos = [0.5, 1.5] + [i + 2.5 for i in range(n_levels)]
            ytick_labels = ["", _map_name(str(varname))] + [str(lvl) for lvl in levels]
            ax.set_yticks(ytick_pos)
            ax.set_yticklabels(ytick_labels)

            ticklbls = ax.get_yticklabels()
            if len(ticklbls) > 1:
                ticklbls[1].set_fontweight("bold")

            ax.tick_params(axis="y", length=0)
            ax.set_xlim(*common_xlim)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)

        # Bottom x-axis only
        ax_x = fig.add_subplot(gs[current_row : current_row + 1, 0])
        current_row += 1

        ax_x.set_xlim(*common_xlim)
        ax_x.set_yticks([])
        ax_x.tick_params(axis="y", left=False, labelleft=False)

        ax_x.set_xlabel(_map_name(measure))
        ax_x.set_xticks([x_min, (x_min + x_max) / 2.0, x_max])

        for spine in ["top", "left", "right"]:
            ax_x.spines[spine].set_visible(False)
        ax_x.spines["bottom"].set_visible(True)
        ax_x.tick_params(axis="x", bottom=True, labelbottom=True)

        # Save and return
        plt.savefig(f"{self.results_dir}/multiverse_plot.{ftype}", bbox_inches="tight", dpi=dpi)
        return self._handle_figure_returns(fig)

    def integrate(self, measure=None, method="uniform", type="mean"):
        """
        Integrate the multiverse results.

        Parameters
        ----------
        measure : string
            Name of the measure to integrate.

        method : string
            Method to use for integration. Options are:
                 "uniform" (default): Simple mean/median across all universes
                 "bma": Bayesian model averaging (requires BIC values in the results)
                 
        type : string
            Type of (weighted) integration. Options are "mean" (default) or "median".
        """
        # Get results dataframe and convert columns to lowercase
        results = self.get_results(as_df=True)
        results.columns = results.columns.str.lower()

        # Initial checks
        if measure is None:
            raise ValueError("Please provide a measure to integrate.")
        if measure not in results.columns:
            raise ValueError(f"The measure '{measure}' was not found in the results.")
        if method == "bma" and "bic" not in results.columns:
            raise ValueError("BMA weights require a 'bic' column in the results.")
        
        # Get measure and compute weights
        x = results[measure].to_numpy()

        if method == "uniform":
            weights = self._uniform_weights(x)
        elif method == "bma":
            weights = self._bma_weights(results)
        else:
            raise ValueError("method must be 'uniform' or 'bma'")
        
        # Compute integrated estimate
        if type == "mean":
            integrated_estimate = self._weighted_mean(x, weights)
        elif type == "median":
            integrated_estimate = self._weighted_median(x, weights)
        else:
            raise ValueError("type must be 'mean' or 'median'")
        
        return integrated_estimate, weights

    # Internal methods
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

    def _render_val(self, v):
        """Render decision value. `$...` → literal inject, else double-quoted string."""
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, (int, float)) or v is None:
            return repr(v)
        if isinstance(v, str):
            if v.startswith("$"):
                return v[1:]  # literal inject (variable, code, collection, call...)
            return f"\"{self._escape_double_quotes(v)}\""  # always double-quoted string
        if isinstance(v, list):
            return "[" + ", ".join(self._render_val(x) for x in v) + "]"
        if isinstance(v, tuple):
            return "(" + ", ".join(self._render_val(x) for x in v) + ")"
        if isinstance(v, dict):
            # If it's a function spec, _handle_dict will be used by _format_type.
            # If it's a plain dict value, render a Python dict literal.
            items = ", ".join(f"{self._render_val(k)}: {self._render_val(val)}" for k, val in v.items())
            return "{" + items + "}"
        return repr(v)

    def _handle_dict(self, value):
        """
        Turn a dict spec into a Python call string.
        Supports:
        {"func": "comet.connectivity.Static_Pearson(ts).estimate()"}
        {"func": "comet.connectivity.Static_Partial", "args": {...}}
        {"func": "bct.clustering_coef_bu", "args": {"G": "$G"}}
        Optional: {"positional": ["$ts", 123]} for explicit positional args.
        """
        func = value.get("func", "")
        args = value.get("args", {}) or {}
        positional = value.get("positional", []) or []

        # If func already looks like a full call string, pass it through untouched.
        # (Convention: if you need variables here, prefer using "args" with "$var")
        if "(" in func:
            return func

        # Build argument list
        parts = []
        parts += [self._render_val(p) for p in positional]
        parts += [f"{k}={self._render_val(v)}" for k, v in args.items()]
        arg_str = ", ".join(parts)

        call_expr = f"{func}({arg_str})" if parts else f"{func}()"

        # comet.connectivity classes require .estimate()
        if func.startswith("comet.connectivity."):
            call_expr += ".estimate()"

        return call_expr

    def _format_type(self, value):
        """
        Convert decision values into Python code strings for template injection.
        """
        # Function spec dicts go through _handle_dict; everything else through _render_val.
        if isinstance(value, dict) and "func" in value:
            return self._handle_dict(value)
        return self._render_val(value)

    def _escape_double_quotes(self, s: str) -> str:
        # Keep things robust if the literal happens to contain quotes/backslashes
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _combine_results(self):
        """
        Internal function: Combines the results in a single dictionary and saves it as a pickle file
        """
        file_name = "multiverse_results.pkl"
        file_paths = glob.glob(os.path.join(self.script_dir, "temp", "universe_*.pkl"))
        
        if not file_paths:
            return
        
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

    def _flatten_decisions(self, dec_block):
        """ 
        Convert {'Decision 1': 'X', 'Value 1': Y, ...} into {'X': 'Y', ...}.
        Values are stringified for categorical plotting. 
        """
        if not isinstance(dec_block, dict):
            return {}

        out = {}
        for k, v in dec_block.items():
            m = re.fullmatch(r"Decision\s+(\d+)", str(k))
            if not m:
                continue

            idx = m.group(1)
            value_key = f"Value {idx}"
            out[str(v)] = str(dec_block.get(value_key, "NA"))

        return out

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
            raise ValueError("Results file not found. Please run the multiverse analysis first.")

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

    def _handle_figure_returns(self, fig):
        """
        Helper function to handle plotting
        """
        if self._in_notebook():
            ipy_display(fig)
            plt.close(fig)
            return
        else:
            return fig

    # Multiverse integration
    def _weighted_mean(self,x: np.ndarray, w: np.ndarray) -> float:
        """
        Compute the weighted mean of x with weights w.
        Assumes w >= 0 and w.sum() == 1.
        """
        return float(np.sum(w * x))

    def _weighted_median(self, x: np.ndarray, w: np.ndarray) -> float:
        """
        Compute the weighted median of x with weights w.
        Assumes w >= 0 and w.sum() == 1.
        """
        order = np.argsort(x)
        x_sorted = x[order]
        w_sorted = w[order]

        cw = np.cumsum(w_sorted)
        return float(x_sorted[np.searchsorted(cw, 0.5, side="left")])

    def _uniform_weights(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute uniform weights for all universes.
        """
        n = len(data)
        return np.full(n, 1.0 / n, dtype=float)

    def _bma_weights(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute normalised BMA weights from BIC values.
        """
        bic = data["bic"].to_numpy(float)
        delta = bic - np.min(bic)
        w = np.exp(-0.5 * delta)
        return w / w.sum()

# Load an existing multiverse
def load_multiverse(path=None):
    """
    Load a previously created multiverse from disk.

    Parameters
    ----------
    path : str
        A full/relative path to an existing multiverse folder
        
    """
    if path is not None:
        name = path.rstrip("/") # Remove trailing slash if present
        mverse = Multiverse(name=name, path=path) # Create the Multiverse object for the given path
    else:
        raise ValueError("Please provide a name/path to a multiverse directory.")
    
    if not os.path.exists(mverse.multiverse_dir + "/multiverse_summary.csv"):
        shutil.rmtree(mverse.multiverse_dir) # clean up created directory
        raise ValueError("The specified path does not seem to contain a valid multiverse.")

    return mverse
