import pytest
from comet import multiverse

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
