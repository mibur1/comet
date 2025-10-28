from scipy.io import savemat, loadmat
import teneto
import numpy as np

wd = "/home/mibur/comet/tests/MATLAB"
ts = np.loadtxt(f"{wd}/time_series.txt")

sd = teneto.timeseries.derive_temporalnetwork(ts[:,:5].T, params={"method": "distance", "distance": "euclidean"})
dcc = loadmat(f"{wd}/dcc.mat")["dcc"]
fls = loadmat(f"{wd}/fls.mat")["fls"]
leida = loadmat(f"{wd}/leida.mat")["leida"]
ets = loadmat(f"{wd}/ets.mat")["ets"]

savemat("/home/mibur/comet/src/comet/data/tests/connectivity.mat", {"sd": sd, "dcc": dcc, "fls": fls, "leida": leida, "ets": ets})
