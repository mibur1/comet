import os
import sys
import types
import pickle
import warnings
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from comet import multiverse

"""
Templates. They are only rendered into universe scripts, never executed here, so the
{{placeholders}} just have to be valid Python syntax (a set inside a set).
"""
def template_basic():
    from comet.utils import save_universe_results
    save_universe_results({"value": {{a}} * 10, "label": {{b}}})

def template_samples():
    import numpy as np
    from comet.utils import save_universe_results
    rng = np.random.default_rng({{seed}})
    save_universe_results({"samples": rng.normal({{mu}}, 1.0, 20)})

def template_exclude():
    from comet.utils import save_universe_results
    save_universe_results({"value": 1.0 if {{ws}} is None else float({{ws}})})

def template_value():
    from comet.utils import save_universe_results
    save_universe_results({"value": {{value}}})

def template_braces():
    from comet.utils import save_universe_results
    x = 5
    s = f"{{x}}"            # not a forking path: must stay a literal f-string brace escape
    t = "{% not jinja %}"
    save_universe_results({"s": s, "t": t, "a": {{a}}})

def template_blank_first_line():

    # the body starts with a blank line and a comment
    from comet.utils import save_universe_results
    save_universe_results({"value": {{a}}})

def template_wrapped():
    import sys
    sys.path.insert(0, {{helper_dir}})
    import mv_helper
    mv_helper.save({"value": 1})

def template_stamped():
    import time
    from comet.utils import save_universe_results
    save_universe_results({"value": {{a}} * 10, "stamp": time.time_ns()})

def template_order():
    from comet.utils import save_universe_results
    save_universe_results({"value": 1 if {{value}} is None else {{value}}, "f1": {{function1}}, "f2": {{function2}}})


# Helpers
def _mv(tmp_path, name="mv"):
    return multiverse.Multiverse(name=name, path=str(tmp_path / name))

def _ran(tmp_path, template=template_basic, forking_paths=None, config=None, name="mv", **run_kwargs):
    if forking_paths is None:
        forking_paths = {"a": [1, 2, 3], "b": ["x", "y"]}
    mv = _mv(tmp_path, name)
    mv.create(template, forking_paths, config)
    mv.run(**run_kwargs)
    return mv

def _decisions(mv):
    return mv.summary(print_df=False, return_df=True)


# Loading
def test_load_multiverse_invalid_path_keeps_files(tmp_path):
    # A directory without multiverse_summary.csv must be left untouched
    user_file = tmp_path / "notes.txt"
    user_file.write_text("keep me")

    with pytest.raises(ValueError):
        multiverse.load_multiverse(str(tmp_path))

    assert user_file.read_text() == "keep me"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"] # no results/ or scripts/ created

def test_load_multiverse_valid_path(tmp_path):
    (tmp_path / "multiverse_summary.csv").write_text("Universe\n")

    mverse = multiverse.load_multiverse(str(tmp_path) + "/")
    assert mverse.multiverse_dir == str(tmp_path) + "/"
    assert mverse.name == tmp_path.name

def test_load_multiverse_no_path():
    with pytest.raises(ValueError):
        multiverse.load_multiverse()

def test_load_multiverse_restores_attributes(tmp_path):
    _ran(tmp_path)
    mv = multiverse.load_multiverse(str(tmp_path / "mv"))
    assert mv.num_universes == 6
    assert mv.forking_paths == {"a": [1, 2, 3], "b": ["x", "y"]}
    assert len(mv.get_results()) == 6


# Constructor
def test_constructor_without_main_file(tmp_path, monkeypatch):
    # REPLs and embedded interpreters have no __main__.__file__: fall back to the working directory
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    monkeypatch.chdir(tmp_path)
    mv = multiverse.Multiverse(name="mv")
    assert os.path.isdir(tmp_path / "mv" / "scripts")
    assert os.path.samefile(mv.multiverse_dir, tmp_path / "mv")

def test_constructor_name_parsing(tmp_path):
    assert multiverse.Multiverse(name="some/dir/my_mv", path=str(tmp_path / "a")).name == "my_mv"
    assert multiverse.Multiverse(name="C:\\data\\my_mv", path=str(tmp_path / "b")).name == "my_mv"
    assert multiverse.Multiverse(name="analysis.py", path=str(tmp_path / "c")).name == "analysis"


# Create -> run -> get_results round trip
def test_roundtrip(tmp_path):
    mv = _ran(tmp_path)
    df = mv.get_results(as_df=True, expand_dec=True)
    assert list(df.index) == [1, 2, 3, 4, 5, 6]
    assert list(df["__a"]) == ["1", "1", "2", "2", "3", "3"]
    assert list(df["__b"]) == ["x", "y"] * 3
    assert list(df["value"]) == [10, 10, 20, 20, 30, 30]
    assert list(df["label"]) == ["x", "y"] * 3
    assert mv.get_results(universe=4)["value"] == 20
    assert mv.num_universes == 6

def test_get_results_uncombined(tmp_path):
    mv = _ran(tmp_path, combine_results=False)
    assert mv.get_results(universe=2)["value"] == 10
    assert mv.get_results(universe=3, as_df=True)["value"].iloc[0] == 20
    with pytest.raises(ValueError):
        mv.get_results()

def test_uncombined_runs_accumulate(tmp_path):
    # e.g. on a cluster: run parts of the multiverse separately, combine at the end
    mv = _mv(tmp_path)
    mv.create(template_basic, {"a": [1, 2, 3], "b": ["x", "y"]})
    mv.run(universe=[1, 2, 3], combine_results=False)
    mv.run(universe=[4, 5, 6])
    assert sorted(mv.get_results()) == [f"universe_{i}" for i in range(1, 7)]

def test_create_again_keeps_results(tmp_path):
    # Re-creating without running (e.g. to analyse existing results) must not lose them
    mv = _ran(tmp_path)
    mv.create(template_basic, {"a": [1, 2, 3], "b": ["x", "y"]})
    assert len(mv.get_results()) == 6

def test_rerunning_some_universes_merges_into_the_results(tmp_path):
    mv = _ran(tmp_path, template=template_stamped, forking_paths={"a": [1, 2, 3]})
    before = mv.get_results()

    mv.run(universe=[2])
    after = mv.get_results()
    assert sorted(after) == ["universe_1", "universe_2", "universe_3"]
    assert after["universe_2"]["stamp"] != before["universe_2"]["stamp"] # rerun
    assert after["universe_1"]["stamp"] == before["universe_1"]["stamp"] # kept
    assert "__script_digest" not in after["universe_1"]

    mv.run() # a full run recomputes everything
    assert all(mv.get_results()[k]["stamp"] != after[k]["stamp"] for k in after)

def test_rerunning_some_universes_drops_results_of_an_earlier_design(tmp_path):
    mv = _ran(tmp_path)                                             # 6 universes with a = 1, 2, 3
    mv.create(template_basic, {"a": [7, 8, 9], "b": ["x", "y"]})   # same universe numbers, other decisions
    mv.run(universe=1)
    results = mv.get_results()
    assert sorted(results) == ["universe_1"]
    assert results["universe_1"]["value"] == 70

def test_recreating_the_same_design_keeps_results_for_merging(tmp_path):
    # Re-running the analysis script re-creates identical universe scripts, so their results stay valid
    mv = _ran(tmp_path)
    mv.create(template_basic, {"a": [1, 2, 3], "b": ["x", "y"]})
    mv.run(universe=6)
    assert len(mv.get_results()) == 6

def test_stale_results_from_larger_design_are_ignored(tmp_path):
    mv = _ran(tmp_path, combine_results=False) # 6 universes left in scripts/temp
    mv.create(template_basic, {"a": [1], "b": ["x", "y"]})
    mv.run()
    assert sorted(mv.get_results()) == ["universe_1", "universe_2"]

def test_stale_results_from_previous_design_are_ignored(tmp_path):
    mv = _ran(tmp_path, combine_results=False)
    mv.create(template_basic, {"a": [7, 8, 9], "b": ["x", "y"]}) # same size, different universes
    mv.run(universe=1)
    results = mv.get_results()
    assert sorted(results) == ["universe_1"]
    assert results["universe_1"]["value"] == 70

def test_run_combines_with_stray_file_in_temp(tmp_path):
    mv = _mv(tmp_path)
    mv.create(template_basic, {"a": [1], "b": ["x"]})
    os.makedirs(os.path.join(mv.script_dir, "temp"), exist_ok=True)
    with open(os.path.join(mv.script_dir, "temp", "notes.txt"), "w") as f:
        f.write("not a result")
    mv.run()
    assert sorted(mv.get_results()) == ["universe_1"]


# Universe selection
def test_summary_and_run_select_the_same_universes(tmp_path):
    mv = _ran(tmp_path, universe=range(2, 4))
    shown = list(mv.summary(range(2, 4), print_df=False, return_df=True)["Universe"])
    assert shown == ["Universe_2", "Universe_3"]
    assert sorted(mv.get_results()) == ["universe_2", "universe_3"]
    assert mv.summary(4, print_df=False, return_df=True)["Universe"] == "Universe_4" # a single row
    with pytest.raises(ValueError):
        mv.summary(7, print_df=False, return_df=True)
    assert list(mv.summary([1, 6], print_df=False, return_df=True)["Universe"]) == ["Universe_1", "Universe_6"]

def test_run_unknown_universe_raises(tmp_path):
    mv = _mv(tmp_path)
    mv.create(template_basic, {"a": [1, 2], "b": ["x"]})
    with pytest.raises(ValueError, match="99"):
        mv.run(universe=99)

def test_run_ignores_other_python_files(tmp_path):
    mv = _mv(tmp_path)
    mv.create(template_basic, {"a": [1], "b": ["x"]})
    with open(os.path.join(mv.script_dir, "helper.py"), "w") as f:
        f.write("raise RuntimeError('must not be executed')\n")
    mv.run()
    assert sorted(mv.get_results()) == ["universe_1"]


# Rules
def test_deduplicate_without_order(tmp_path):
    fp = {"method": ["SW", "FLS"], "ws": [10, 20]}
    config = {"exclude": [[{"method": "FLS"}, "ws"]]}

    mv = _mv(tmp_path)
    mv.create(template_exclude, fp, config)
    assert len(_decisions(mv)) == 3 # SW/10, SW/20, FLS/NaN

    mv.create(template_exclude, fp, {**config, "deduplicate": False})
    assert len(_decisions(mv)) == 4

def test_deduplicate_keeps_distinct_function_specs(tmp_path):
    # Two specs that share a display name but differ in their arguments are different universes
    fp = {"f": [{"name": "pearson", "func": "abs", "args": {"x": 1}},
                {"name": "pearson", "func": "abs", "args": {"x": -2}}],
          "g": [[1, 2], [3]]}
    mv = _mv(tmp_path)
    mv.create(template_value, {**fp, "value": [1]}, {"exclude": [[{"f": "pearson"}, "g"]]})
    assert len(_decisions(mv)) == 2

def test_dedup_report(tmp_path, capsys):
    mv = _mv(tmp_path)
    mv.create(template_exclude, {"method": ["SW", "FLS"], "ws": [10, 20]},
              {"exclude": [[{"method": "FLS"}, "ws"]]}, verbose=True)
    out = capsys.readouterr().out
    assert "Removed 1 duplicate universes" in out
    assert "3 universes remain" in out

def test_exclude_summary_counts_list_conditions(tmp_path, capsys):
    mv = _mv(tmp_path)
    mv.create(template_exclude, {"method": ["SW", "FLS", "MTD"], "ws": [10, 20]},
              {"exclude": [[{"method": ["FLS", "MTD"]}, "ws"]]}, verbose=True)
    out = capsys.readouterr().out
    assert "Set 'ws' to NaN" in out and "(2 total)" in out

def test_remove_rule_with_bare_string_raises(tmp_path):
    mv = _mv(tmp_path)
    with pytest.raises(ValueError, match="remove"):
        mv.create(template_exclude, {"method": ["SW", "FLS"], "ws": [10, 20]}, {"remove": [["ws"]]})

def test_remove_rule(tmp_path):
    mv = _mv(tmp_path)
    mv.create(template_exclude, {"method": ["SW", "FLS"], "ws": [10, 20]},
              {"remove": [[{"method": "FLS"}, {"ws": [10]}]]})
    summary = _decisions(mv)
    assert len(summary) == 3
    assert not ((summary["Value 1"] == "FLS") & (summary["Value 2"] == 10)).any()

def test_order_with_exclude_runs(tmp_path):
    fp = {"function1": ["a", "b"], "function2": ["c", "d"], "value": [1, 2]}
    config = {"order": [["function1", "function2", "value"], ["function2", "function1", "value"]],
              "exclude": [[{"function1": "a"}, "value"]]}
    mv = _ran(tmp_path, template=template_order, forking_paths=fp, config=config)
    df = mv.get_results(as_df=True)
    assert len(df) == 12 # 2 orders x (a: 2 + b: 4)
    assert set(df["value"]) == {1, 2}


# Template rendering
def test_template_braces_are_left_alone(tmp_path):
    mv = _ran(tmp_path, template=template_braces, forking_paths={"a": [1]})
    res = mv.get_results(universe=1)
    assert res["s"] == "{x}"
    assert res["t"] == "{% not jinja %}"
    assert res["a"] == 1

def test_template_with_blank_first_line(tmp_path):
    mv = _ran(tmp_path, template=template_blank_first_line, forking_paths={"a": [1, 2]})
    assert mv.get_results(universe=2)["value"] == 2

def test_string_values_roundtrip(tmp_path):
    values = ["line1\nline2", 'say "hi"', "back\\slash", "tab\there", "ümlaut ✓", "it's"]
    mv = _ran(tmp_path, template=template_value, forking_paths={"value": values})
    df = mv.get_results(as_df=True)
    assert list(df["value"]) == values

def test_invalid_universe_script_raises_at_create(tmp_path):
    mv = _mv(tmp_path)
    with pytest.raises(ValueError, match="[Uu]niverse 2"):
        mv.create(template_value, {"value": ["$1", "$1 +"]})


# Results handling
def test_excluded_decisions_are_missing_not_nan_strings(tmp_path):
    mv = _ran(tmp_path, template=template_exclude, forking_paths={"method": ["SW", "FLS"], "ws": [10, 20]},
              config={"exclude": [[{"method": "FLS"}, "ws"]]})
    df = mv.get_results(as_df=True, expand_dec=True)
    assert df["__ws"].isna().sum() == 1
    assert "nan" not in set(df["__ws"].dropna())

def test_none_value_is_a_level(tmp_path):
    mv = _ran(tmp_path, template=template_value, forking_paths={"value": [None, 1]})
    df = mv.get_results(as_df=True, expand_dec=True)
    assert list(df["__value"]) == ["None", "1"]

def test_save_universe_results_from_a_helper_module(tmp_path):
    helper_dir = tmp_path / "helpers"
    helper_dir.mkdir()
    (helper_dir / "mv_helper.py").write_text(
        "from comet.utils import save_universe_results\n"
        "def save(data):\n"
        "    save_universe_results(data)\n")
    mv = _ran(tmp_path, template=template_wrapped, forking_paths={"helper_dir": [str(helper_dir)]})
    assert mv.get_results(universe=1)["value"] == 1


# Integration
def test_integrate_and_compare(tmp_path):
    mv = _ran(tmp_path)
    est, w = mv.integrate("value")
    assert np.isclose(est, 20) and np.allclose(w, 1 / 6)
    table, weights = mv.compare_integration("value")
    assert list(table["Scheme"]) == ["Uniform", "MLI"]
    assert np.isclose(weights["MLI"].sum(), 1)

def test_mli_ignores_excluded_decisions(tmp_path):
    mv = _ran(tmp_path, template=template_exclude, forking_paths={"method": ["SW", "FLS"], "ws": [10, 20]},
              config={"exclude": [[{"method": "FLS"}, "ws"]]})
    df = mv.get_results(as_df=True, expand_dec=True)
    Q, col_types, col_names = mv._get_decision_matrix(df)
    assert not any("nan" in c for c in col_names) # NaN is not a level
    dist = mv._gower_distance(Q, col_types)
    assert np.all(np.isfinite(dist)) and np.allclose(dist, dist.T)
    est, w = mv.integrate("value", method="mli")
    assert np.isfinite(est) and np.isclose(w.sum(), 1)

def test_gower_distance_with_missing_values():
    mv = multiverse.Multiverse.__new__(multiverse.Multiverse)
    Q = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, np.nan]])
    dist = mv._gower_distance(Q, ["binary", "binary"])
    assert np.isclose(dist[0, 1], 0.5) # one of two decisions differs
    assert np.isclose(dist[0, 2], 1.0) # only the first decision is comparable, and it differs
    assert np.isclose(dist[1, 2], 0.0)


# Plotting
@pytest.fixture
def no_deprecations():
    with warnings.catch_warnings():
        warnings.simplefilter("error", matplotlib.MatplotlibDeprecationWarning)
        yield
    plt.close("all")

def _ytick_labels(fig):
    return {t.get_text() for ax in fig.axes for t in ax.get_yticklabels()}

def test_visualize(tmp_path, no_deprecations):
    mv = _mv(tmp_path)
    mv.create(template_basic, {"a": [1, 2, 3], "b": ["x", "y"]})
    assert mv.visualize(universe=2) is not None
    assert os.path.isfile(os.path.join(mv.results_dir, "multiverse.png"))

def test_visualize_single_options(tmp_path, no_deprecations):
    mv = _mv(tmp_path)
    mv.create(template_basic, {"a": [1], "b": ["x"]})
    assert mv.visualize(exclude_single=True) is not None

def test_specification_curve(tmp_path, no_deprecations):
    mv = _ran(tmp_path, template=template_samples, forking_paths={"seed": [1, 2], "mu": [0.0, 1.0, 2.0]})
    fig = mv.specification_curve("samples", baseline=0, p_value=True, ci=True)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "95% CI" in legend
    fig = mv.specification_curve("samples", ci=90, smooth_ci=False)
    assert "90% CI" in [t.get_text() for t in fig.axes[0].get_legend().get_texts()]

def test_specification_curve_ci_band_stays_within_bounds(tmp_path, no_deprecations):
    mv = _ran(tmp_path, template=template_samples, forking_paths={"seed": [1, 2, 3], "mu": [0.0, 0.1, 5.0]})
    fig = mv.specification_curve("samples", ci=True, smooth_ci=True)
    band = fig.axes[0].collections[-1].get_paths()[0].vertices[:, 1]

    # The smoothed band must not overshoot the actual CI bounds at the step between universes
    from scipy import stats
    half = [stats.t.ppf(0.975, len(s) - 1) * np.std(s, ddof=1) / np.sqrt(len(s))
            for s in mv.get_results(as_df=True)["samples"]]
    means = [np.mean(s) for s in mv.get_results(as_df=True)["samples"]]
    lo = min(m - h for m, h in zip(means, half))
    hi = max(m + h for m, h in zip(means, half))
    assert band.min() >= lo - 1e-9 and band.max() <= hi + 1e-9

def test_plots_skip_excluded_decisions(tmp_path, no_deprecations):
    mv = _ran(tmp_path, template=template_exclude, forking_paths={"method": ["SW", "FLS"], "ws": [10, 20]},
              config={"exclude": [[{"method": "FLS"}, "ws"]]})
    mv.add_results("noisy", [1.0, 2.0, 3.5])
    assert "nan" not in _ytick_labels(mv.specification_curve("noisy"))
    assert "nan" not in _ytick_labels(mv.multiverse_plot("noisy", n_bins=3))
