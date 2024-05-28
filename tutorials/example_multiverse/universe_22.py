import numpy as np
import bct
import comet

print(f"Decision 1: 'world'")
print(f"Decision 2: 2")
print(f"Decision 3:True")

# Load example data and calculate dFC + local efficiency
ts = comet.data.load_example()
dfc = comet.methods.Jackknife(ts, **{'windowsize': 11}).connectivity()
dfc = dfc[0] if isinstance(dfc, tuple) else dfc #required as LeiDA returns multiple outputs

efficiency = np.zeros((ts.shape[0], dfc.shape[1]))
for i in range(dfc.shape[2]):
    W = dfc[:, :, i]
    W = np.abs(W)
    efficiency[i] = comet.graph.efficiency_wei(W, **{'local': True})

print("Universe finished.")