This folder contains scripts for creating the test data to validate the dFC methods.

Workflow (make sure to adjust the file paths in every script):

1. Run all MATLAB scripts (required toolboxes are listed at the beginning of each script)
2. Run `combine.py` to create a single data file
3. The data is now ready for the tests: `pytest tests/connectivity.py`
