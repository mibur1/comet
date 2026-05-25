import math
import numpy as np
import nibabel as nib
import pyvista as pv
import urllib.request
import importlib_resources
from pathlib import Path
from typing import Any, cast
from scipy.io import loadmat
from matplotlib import cm as mpl_cm
from matplotlib.colors import ListedColormap

nib.imageglobals.logger.setLevel(40)
_DISCRETE_CMAP_REF_MAX: dict[str, int] = {}

# Parcellation
def parcellate(dtseries :str|np.ndarray|nib.cifti2.cifti2.Cifti2Image|None=None, 
               atlas         : str="schaefer", 
               resolution    : int=100, 
               subcortical   : None|str=None,
               networks      : int=7, 
               kong          : bool=False,
               standardize   : bool=True,
               method        = np.mean,
               return_labels : bool=False,
               debug         : bool=False
    ) -> np.ndarray | tuple[np.ndarray|None, list[str], np.ndarray]:
    """ 
    Parcellate cifti data (.dtseries.nii) using a given atlas.  
    Atlases for many different parameter combinations are available and will be downloaded on demand.  
    If the atlas for the parameter combination is not available, a ValueError is raised.

    References
    ----------
    - Schaefer, Glasser, Gordon (+ Tian subcortical): https://github.com/yetianmed/subcortex
    - Schaefer + Yan (cortical only): https://github.com/ThomasYeoLab/CBIG

    Note: Any cortical atlas can be used on its own or combined with any Tian scale. Combinations
          without an available file are assembled at runtime from the cortical atlas plus the Tian
          subcortical block reused from the Gordon+Tian atlas.

    Parameters
    ----------
    dtseries : str, np.ndarray nibabel.cifti2.cifti2.Cifti2Image
        string containing a path, array containing vertex data, or nibabel cifti image object
    
    atlas : string
        Name of the atlas to use for parcellation. Available options are:
        - "schaefer": Schaefer et al. (2018) atlas
        - "yan": Yan et al. (2023) homotopic atlas
        - "glasser": Glasser et al. (2016) atlas
        - "gordon": Gordon et al. (2016) atlas
    
    resolution : int
        Number of parcels in the atlas. Only used with the Schaefer and Yan atlases.
        Available options are: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000.

    subcortical : None or string
        If a string containing the scale is provided, the Tian subcortical parcels are included.
        Available options are: None, 'S1' (16 ROIs), 'S2' (32 ROIs), 'S3' (50 ROIs), 'S4' (54 ROIs).
        Works with every atlas at any scale: None gives a cortical-only parcellation, a scale
        appends the Tian structures (keyed first). Cortical parcel counts: schaefer/yan = resolution,
        glasser = 360, gordon = 333.

    networks : int
        Number of networks in the atlas. Only used with Schaefer atlas.
        Available options are: 7, 17
    
    kong : bool
        Use the Kong 2022 version of the Schaefer atlas (only for Schaefer cortical atlas with 17 networks).
        Reference: https://doi.org/10.1093/cercor/bhab101

    standardize : bool
        Standardize the time series to zero (temporal) mean and unit variance before parcellation.

    method : function
        Aggregation function to use for parcellation. Default (and the only tested function) is np.mean.

    debug : bool
        Flag to provide additional debugging information. Default is False.

    Returns
    -------
    ts_parc : np.ndarray or tuple  
        If ``return_labels`` is False (default):  
            - ts_parc : (T, P) np.ndarray  
                Parcellated time series data.

        If ``return_labels`` is True:  
            -  ts_parc : (T, P) np.ndarray  
                Parcellated time series data. None if no input data was provided.
            -  node_labels : list of str  
                Label name for each parcel.  
            -  vertex_labels : np.ndarray  
                ROI index for each vertex in the CIFTI file.   
    """
    if isinstance(dtseries, nib.cifti2.cifti2.Cifti2Image):
        ts = dtseries.get_fdata()
    elif isinstance(dtseries, np.ndarray) or isinstance(dtseries, np.memmap):
        ts = dtseries
    elif isinstance(dtseries, str):
        data = nib.load(dtseries)
        ts = data.get_fdata()
    elif dtseries is None:
        pass
    else:
        print("Error: Input must be either a string to a CIFTI file, a nibabel CIFTI object, " \
        "a numpy array containing vertex data, or None (to return only atlas labels).")
        return
    
    # Check provided parameters
    if atlas not in ["schaefer", "glasser", "gordon", "yan"]:
        raise ValueError(f"Atlas '{atlas}' not available. Please choose from ['schaefer', 'glasser', 'gordon', 'yan'].")
    if resolution not in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        raise ValueError(f"Resolution '{resolution}' not available. Please choose from [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000].")
    if networks not in [7, 17]:
        raise ValueError(f"Networks '{networks}' not available. Please choose from [7, 17].")
    if subcortical not in [None, 'S1', 'S2', 'S3', 'S4']:
        raise ValueError(f"Subcortical scale '{subcortical}' not available. Please choose from [None, 'S1', 'S2', 'S3', 'S4'].")
    if kong not in [True, False]:
        raise ValueError(f"Kong flag must be a boolean value (True or False).")

    # Combinations which automatically adjust parameters with a warning instead of raising an error.
    if atlas == "schaefer" and networks == 7 and kong is True:
        print(f"[WARN] Schaefer Kong version is only available with 17 networks. Networks were set to 17.")
        networks = 17

    # Get the atlas    
    vertex_labels, keys, node_labels, _ = _get_atlas(atlas=atlas, resolution=resolution, networks=networks, subcortical=subcortical, kong=kong, debug=debug)
    
    # If we have no input data, return the labels now
    if dtseries is None:
        return (None, node_labels, vertex_labels)
    
    # Cortical-only atlases are 64984-vertex surface maps that include the medial wall,
    # so the medial-wall columns have to be inserted into the (59412-vertex) cortical data.
    if subcortical is None:
        with importlib_resources.path("comet.data.atlas", "fs_LR_32k_medial_mask.mat") as maskdir:
            medial_mask = loadmat(maskdir)['medial_mask'].squeeze().astype(bool)
        idx = np.where(medial_mask == 0)[0]

        # prepare idices and insert them into the HCP data
        for i, value in enumerate(idx):
            idx[i] = value - i

        cortical_vertices = 59412 # HCP data has 59412 cortical vertices
        ts = ts[:,:cortical_vertices]
        ts = np.insert(ts, idx, np.nan, axis=1)

    # Standardize before parcellation
    if standardize:
        ts = _stdize(ts)

    # Parcellation
    ts_parc = np.zeros((len(ts), len(node_labels)), dtype=ts.dtype)

    i = 0
    for k, lab in zip(keys, node_labels):
        if k == 0:
            continue

        mask = (vertex_labels == k)
        if not np.any(mask):
            ts_parc[:, i] = 0
            i += 1
            print(f"[WARN] ROI {lab} is empty and was set to zero.")
            continue

        ts_parc[:, i] = method(ts[:, mask], axis=1)
        i += 1

    return (ts_parc, node_labels, vertex_labels) if return_labels else ts_parc

def get_networks(labels: list[str]) -> tuple[list[str], np.ndarray, list[str], dict[str, int]]:
    """
    Extract network information for Schaefer-Yeo parcellations.

    Parameters
    ----------
    labels : list of str
        Atlas parcel labels obtained from ``cifti.parcellate()``.

    Returns
    -------
    networks : list of str
        Network label per parcel (length N).
    ids : np.ndarray
        Integer network ids per parcel (length N).
    hemisphere : list of str    
        Hemisphere label per parcel ('LH' or 'RH'; length N).
    network_map : dict[str, int]
        Mapping from network name to integer id.

    Raises
    ------
    ValueError
        If network labels cannot be inferred from the atlas labels.
    """
    if len(labels) == 0:
        raise ValueError("Empty label list.")

    networks: list[str] = []
    hemisphere: list[str] = []

    for lab in labels:
        # Cortical Schaefer Yeo-style labels
        if ("networks_" in lab) or ("Networks_" in lab):
            parts = lab.split("_")
            if len(parts) < 3:
                raise ValueError(f"Unexpected Schaefer label format: {lab}")
            hemisphere.append(parts[1])
            networks.append(parts[2])

        # Simple subcortical extension
        elif lab.endswith("-lh") or lab.endswith("-rh"):
            hemisphere.append("LH" if lab.endswith("-lh") else "RH")
            networks.append("Subcortical")

        # Other atlases without canonical network labels will raise errors
        elif lab.startswith("Parcel_"):
            raise ValueError("Error: Gordon atlas detected. No canonical network partition is available.")
        elif lab.endswith("_ROI"):
            raise ValueError("Error: Glasser/HCP-MMP atlas detected. No canonical network partition is available.")
        else:
            raise ValueError(f"Unknown atlas label format; cannot infer network assignments: {lab}")

    # Map network names to integers (stable alphabetical order)
    uniq = sorted(set(networks))
    
    if "Subcortical" in uniq:
        uniq.remove("Subcortical")
        network_map = {"Subcortical": 0}
        for i, name in enumerate(uniq, start=1):
            network_map[name] = i
    else:
        network_map = {name: i + 1 for i, name in enumerate(uniq)}

    ids = np.array([network_map[n] for n in networks], dtype=int)

    return networks, ids, hemisphere, network_map

def _get_atlas(atlas, resolution, networks, subcortical, kong, debug) -> tuple:
    """
    Helper function: Get and prepare a CIFTI-2 atlas for parcellation.

    Parameters
    ----------
    **See parcellate() for details.**

    Returns
    -------
    tuple
        A tuple containing:
        - rois : np.ndarray
            ROI indices for each vertex.
        - keys : np.ndarray
            Keys of the atlas.
        - labels : list
            Labels of the atlas.
        - rgba : list
            RGBA values of each label.
    """

    # Cortical sources. Schaefer/Yan ship standalone cortical dlabels; Glasser/Gordon are only
    # published as Tian-combined files, so their cortex is extracted from those. The Tian
    # subcortical block is always taken from the Gordon+Tian file and appended for any scale,
    # so every atlas works both cortical-only and combined with any Tian scale.
    base_urls = {
        "schaefer_c": "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_{parcels}Parcels_{kong}{networks}Networks_order.dlabel.nii",
        "yan":        "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Yan2023_homotopic/parcellations/HCP/fsLR32k/yeo{networks}/{parcels}Parcels_Yeo2011_{networks}Networks.dlabel.nii",
        "gordon":     "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Cortex-Subcortex/Gordon333.32k_fs_LR_Tian_Subcortex_{subcortical}.dlabel.nii",
        "glasser":    "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Cortex-Subcortex/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_{subcortical}.dlabel.nii",
    }

    def _download_and_load(filename, url):
        """Download (if needed) and load a CIFTI-2 dlabel atlas."""
        with importlib_resources.path("comet.data.atlas", filename) as atlas_path:
            if not atlas_path.exists():
                urllib.request.urlretrieve(url, atlas_path)
                print(f"Atlas not available. Downloading to: {atlas_path}")
            return nib.load(str(atlas_path))

    def _extract_labels(img):
        """Return (keys, labels, rgba) from the dlabel label table, skipping background (key 0)."""
        # Usually for dlabel.nii files we have the following header structure
        #       axis 0: LabelAxis
        #       axis 1: BrainModelAxis
        named_map = list(img.header.get_index_map(0).named_maps)[0]
        keys, labels, rgba = [], [], []
        for key, label in named_map.label_table.items():
            if key == 0:
                continue # skip background
            keys.append(key)
            labels.append(label.label)
            rgba.append(label.rgba)
        return np.asarray(keys), labels, rgba

    def _print_brainmodels(img):
        for idx, (name, _slice, _bm) in enumerate(img.header.get_axis(1).iter_structures()):
            print(idx, str(name), _slice, _bm)

    def _load_medial_mask():
        with importlib_resources.path("comet.data.atlas", "fs_LR_32k_medial_mask.mat") as maskdir:
            return loadmat(maskdir)["medial_mask"].squeeze().astype(bool)

    def _cortical():
        """Cortical parcellation as a 64984-vertex surface map with contiguous keys 1..N.
        
        Glasser and Gordon are downloaded as Tian-combined files, so we fetch the S1 file, drop the 
        leading Tian structures, renumber the cortical parcels to 1..N, and re-insert the medial wall.
        """
        if atlas == "schaefer":
            url = base_urls["schaefer_c"].format(parcels=resolution, networks=networks, kong="Kong2022_" if kong else "")
        elif atlas == "yan":
            url = base_urls["yan"].format(parcels=resolution, networks=networks)
        else:  # glasser, gordon
            url = base_urls[atlas].format(subcortical="S1")
        img = _download_and_load(url.split("/")[-1], url)
        rois_full = img.dataobj[0].astype(int).squeeze()
        keys, labels, rgba = _extract_labels(img)

        if atlas in ["schaefer", "yan"]:
            return rois_full, keys, labels, rgba, img  # already a 64984-vertex cortical map

        # Combined file: subcortex is keyed first (1..M), cortex after. Keep the cortex only.
        n_sub = len(set(np.unique(rois_full[59412:]).tolist()) - {0})
        cort_keys_sorted = sorted(int(k) for k in keys if int(k) > n_sub)
        remap = {old: i + 1 for i, old in enumerate(cort_keys_sorted)}  # -> 1..N
        lut = np.zeros(int(rois_full.max()) + 1, dtype=int)
        for old, new in remap.items():
            lut[old] = new
        medial_mask = _load_medial_mask()
        rois = np.zeros(medial_mask.size, dtype=int)  # 64984
        rois[medial_mask] = lut[rois_full[:59412]]  # re-insert the medial wall
        key_to_idx = {int(k): i for i, k in enumerate(keys)}
        return (rois,
                np.asarray([remap[k] for k in cort_keys_sorted]),
                [labels[key_to_idx[k]] for k in cort_keys_sorted],
                [rgba[key_to_idx[k]] for k in cort_keys_sorted],
                img)

    cort_rois, cort_keys, cort_labels, cort_rgba, cort_img = _cortical()

    # Cortical-only: return the 64984-vertex surface map.
    if subcortical is None:
        if debug:
            _print_brainmodels(cort_img)
        return (cort_rois, cort_keys, cort_labels, cort_rgba)

    # Combined: append the Tian subcortical block taken from the Gordon+Tian file
    sub_url = base_urls["gordon"].format(subcortical=subcortical)
    sub_img = _download_and_load(sub_url.split("/")[-1], sub_url)
    sub_rois = sub_img.dataobj[0].astype(int).squeeze()[59412:]  # 31870 subcortical grayordinates
    sub_keys, sub_labels, sub_rgba = _extract_labels(sub_img)
    sub_present = sorted(int(k) for k in sub_keys if np.any(sub_rois == k))
    n_sub = len(sub_present)

    sub_remap = {old: i + 1 for i, old in enumerate(sub_present)}
    sub_rois_new = np.zeros_like(sub_rois)
    for old, new in sub_remap.items():
        sub_rois_new[sub_rois == old] = new

    cort_rois59412 = cort_rois[_load_medial_mask()]  # drop the medial wall -> 59412 grayordinates
    cort_rois_new = np.where(cort_rois59412 > 0, cort_rois59412 + n_sub, 0)

    rois = np.concatenate([cort_rois_new, sub_rois_new])  # 91282 grayordinates (cortex-first)
    key_to_idx = {int(k): i for i, k in enumerate(sub_keys)}
    keys = np.concatenate([np.arange(1, n_sub + 1, dtype=cort_keys.dtype), cort_keys + n_sub])
    labels = [sub_labels[key_to_idx[old]] for old in sub_present] + list(cort_labels)
    rgba   = [sub_rgba[key_to_idx[old]] for old in sub_present] + list(cort_rgba)

    if debug:
        print(f"[{atlas} cortical source]"); _print_brainmodels(cort_img)
        print("[Gordon+Tian subcortical source]"); _print_brainmodels(sub_img)

    return (rois, keys, labels, rgba)

def _stdize(ts) -> np.ndarray:
    """
    Helper function: Standardize time series to zero (temporal) mean and unit standard deviation.

    Parameters
    ----------
    ts : np.ndarray
        Time series data

    Returns
    -------
    ts : np.ndarray
        Standardized time series data
    """
    mean = np.mean(ts, axis=0, keepdims=True)
    std  = np.std(ts, axis=0, keepdims=True)
    std[std == 0] = 1.0
    
    return (ts - mean) / std

# Plotting
def surface_plot(node_values : np.ndarray|None=None, 
                 vertex_labels: np.ndarray|None=None, 
                 hemi: str="both", 
                 surface: str="inflated", 
                 view_names: tuple[str, str]=("medial", "lateral"), 
                 ncols: int|None=None, 
                 colwise: bool=True, 
                 cmap: str="viridis", 
                 border_color: None|str=None, 
                 border_width: int=5,
                 distance: float=400.0, 
                 size : list[int]|None=None,
                 labelsize : int=18,
                 colorbar: None|str="bottom",
                 colorbar_label : str|None=None,
                 interactive : bool=True,
                 fname : str|None=None):
    """
    Plot cortical hemispheres with optional parcel border overlays.

    Parameters
    ----------
    node_values : ndarray or None  
        Parcel-level values (1D). If None, only surfaces are shown.
    vertex_labels : ndarray or None  
        Vertex-to-parcel labels for both hemispheres (length 64984).
    hemi : {"left", "right", "both"}
        Hemisphere(s) to render.
    surface : str
        Surface type. Valid options are:  
        - "midthickness_orig"
        - "midthickness_mni"
        - "inflated"
        - "very_inflated"
        - "super_inflated"
        - "sphere"
    view_names : tuple containing one or multiple strings
        Views to render per hemisphere. Options are:  
        - "lateral"
        - "medial"
        - "anterior"
        - "posterior"
        - "superior"
        - "inferior"
    ncols : int or None
        Number of subplot columns.
    colwise : bool
        If True (default), fill subplots column-wise, else fill row-wise.
    cmap : str
        Colormap for node values.
    border_color : str or None
        Border color. If None, no border overlay is added.
    border_mode : {"lines", "mask"}
        Border rendering mode. "lines" draws smooth edge lines (default).
        "mask" uses the legacy vertex-mask overlay method.
    border_line_smoothing : int
        Number of Chaikin smoothing iterations for line borders.
        Only used when ``border_mode="lines"``.
    border_line_decimate : int
        Keep every Nth point along each border polyline before smoothing.
        Use 1 to keep all points. Only used when ``border_mode="lines"``.
    distance : float
        Camera distance.
    size : tuple[int, int] or None
        Plotter window size.
    colorbar : {"bottom", "right", None}
        Shared colorbar placement outside data panels. If None, no colorbar is shown.
    colorbar_label : str or None
        Label for the colorbar.
    interactive : bool
        Show the plot in an interactive window (default is True).
    fname : string or None
        Save the plot (will consider manipulations done in the interactive window).  
        The name should contain the desired file type with the options being:    
        - Raster: ".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"  
        - Vectorised: ".svg", ".eps", ".ps", ".pdf", ".tex"  
    """
    # Input validation / normalization
    if node_values is None:
        print("Warning: node_values are required for data plotting. Proceeding with blank surfaces.")
    else:
        node_values = np.asarray(node_values, dtype=float)
        vertex_labels = np.asarray(vertex_labels, dtype=np.int64)
        if vertex_labels.ndim != 1:
            raise ValueError("vertex_labels must be a 1D array of parcel IDs per vertex.")
        if node_values.ndim != 1:
            raise ValueError("node_values must be a 1D array of node/parcel values.")
        if vertex_labels.size != 64984:
            raise ValueError(f"vertex_labels must have length 64984. Got {vertex_labels.size}.")

    # Get surface meshes and desired views
    meshes = _get_surface(surface=surface)  # get the surface mesh(es)
    hemi_order = [hemi] if hemi in ("left", "right") else ["left", "right"] # which hemispheres to plot
    panels = [(h, v) for h in hemi_order for v in view_names]  # list of (hemisphere, view) pairs
    
    # Define default camera positions
    base_cams = {
        "lateral":   ("x", +1, (0, 0, 1)),
        "medial":    ("x", -1, (0, 0, 1)),
        "anterior":  ("y", -1, (0, 0, 1)),
        "posterior": ("y", +1, (0, 0, 1)),
        "superior":  ("z", -1, (0, 1, 0)),
        "inferior":  ("z", +1, (0, 1, 0))
    }

    # Check validity of views
    unknown_views = sorted({v for _, v in panels if v not in base_cams})
    if unknown_views:
        raise ValueError(f"Unknown view(s): {unknown_views}. Available: {list(base_cams.keys())}")

    # Set up plot layout
    n = len(panels)
    if n == 0:
        raise ValueError("No panels to plot. Check 'hemi' and available surfaces.")
    if ncols is None:
        ncols = min(3, n) if n <= 6 else int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    
    if colwise:
        ncols = int(math.ceil(n / nrows)) # guarantee enough columns
    
    axis_idx = {"x": 0, "y": 1, "z": 2}
    
    # Build per-vertex arrays for each hemisphere
    scalars = {}

    if node_values is not None and vertex_labels is not None:
        def _map_to_vertices(parc):
            out = np.full(32492, np.nan, dtype=float)
            mask = (parc > 0) & (parc <= node_values.size)
            out[mask] = node_values[parc[mask] - 1]  # labels are 1-based
            return out

        lh_parc = vertex_labels[:32492]
        rh_parc = vertex_labels[32492:]
        scalars["left"] = _map_to_vertices(lh_parc)
        scalars["right"] = _map_to_vertices(rh_parc)
 
    # Use one shared colour scale across all plotted hemispheres
    vals_list = [scalars[h][~np.isnan(scalars[h])] for h in scalars if h in meshes]
    vals_list = [v for v in vals_list if v.size > 0]
    clim = None
    discrete_values = None
    discrete_nlabels = 4
    discrete_ref_max = None
    if vals_list:
        vals = np.concatenate(vals_list)
        clim = (float(np.nanmin(vals)), float(np.nanmax(vals)))
        if clim[0] == clim[1]:
            eps = 1e-12 if clim[0] == 0.0 else abs(clim[0]) * 1e-12
            clim = (clim[0] - eps, clim[1] + eps)
        # Auto-detect categorical integer-valued maps (e.g., network IDs)
        uniq_vals = np.unique(vals[np.isfinite(vals)])
        # Background values are often encoded as 0 and rendered as NaN later.
        # If nonzero categories exist, exclude 0 from discrete labeling.
        if np.any(uniq_vals != 0):
            uniq_vals = uniq_vals[uniq_vals != 0]
        if uniq_vals.size > 0 and np.allclose(uniq_vals, np.round(uniq_vals)) and uniq_vals.size <= 32:
            discrete_values = uniq_vals.astype(float)
            discrete_nlabels = int(discrete_values.size)
            # Use exact integer limits so scalar-bar ticks can be labeled 1..K.
            clim = (float(np.min(discrete_values)), float(np.max(discrete_values)))
            # Update/lookup reference max ID for this cmap across calls.
            local_max = int(np.max(discrete_values))
            prev_max = _DISCRETE_CMAP_REF_MAX.get(cmap, 0)
            discrete_ref_max = max(prev_max, local_max)
            _DISCRETE_CMAP_REF_MAX[cmap] = discrete_ref_max

    # Colorbar needs an extra grid slot
    show_shared_colorbar = (node_values is not None) and (clim is not None) and (colorbar is not None)
    panel_nrows, panel_ncols = nrows, ncols
    row_weights = None
    col_weights = None
    groups = None
    if show_shared_colorbar and colorbar == "bottom":
        plot_shape = (panel_nrows + 1, panel_ncols)
        # Merge the full bottom row into one renderer and keep it narrow.
        groups = [([panel_nrows], list(range(panel_ncols)))]
        row_weights = [1.0] * panel_nrows + [0.2]
    elif show_shared_colorbar and colorbar == "right":
        plot_shape = (panel_nrows, panel_ncols + 1)
        # Merge the full right column into one renderer and keep it narrow.
        groups = [(list(range(panel_nrows)), [panel_ncols])]
        col_weights = [1.0] * panel_ncols + [0.2]
    else:
        plot_shape = (panel_nrows, panel_ncols)
    
    # Plotting
    pv.global_theme.font.family = "times"
    pl = pv.Plotter(shape=plot_shape, window_size=size, title="Comet Toolbox Surface Viewer", border=False, line_smoothing=True,
                    notebook=_in_notebook() and not interactive, off_screen=not interactive, row_weights=row_weights, col_weights=col_weights, groups=groups)
    pl.enable_anti_aliasing("msaa")

    # Loop through panels and plot each view
    for i, (h, v) in enumerate(panels):
        mesh = meshes[h]
        center = mesh.center
        axis, sign, up = base_cams[v]
        row, col = (i % panel_nrows, i // panel_nrows) if colwise else (i // panel_ncols, i % panel_ncols)
        
        # Swap lateral/medial for right hemisphere
        if h == "right" and v in ("lateral", "medial"):
            sign *= -1

        # Plot the mesh
        pl.subplot(row, col)
        pl.add_text(f"{h} {v}", font_size=int(labelsize*0.7))

        if node_values is None or vertex_labels is None:
            pl.add_mesh(mesh, color="lightgray", smooth_shading=True)
            colorbar_mapper = None
        else:
            values = scalars[h].copy()

            # Make 0 values white by masking them
            zero_mask = values == 0
            values = values.astype(float)
            values[zero_mask] = np.nan  # treat zeros as NaN

            # Plot data to the surface
            mesh_kwargs = dict(scalars=values, clim=clim, nan_color="white", nan_opacity=1.0,
                               show_scalar_bar=False, interpolate_before_map=False, smooth_shading=True)
            
            if discrete_values is not None:
                # Keep ID->color mapping stable for single-ID subsets.
                if int(discrete_values.size) == 1:
                    v = int(round(float(discrete_values[0])))
                    n_ref = max(int(discrete_ref_max) if discrete_ref_max is not None else v, v, 2)
                    cmap_ref = mpl_cm.get_cmap(cmap, n_ref)
                    mesh_kwargs["cmap"] = ListedColormap([cmap_ref(v - 1)])
                    mesh_kwargs["clim"] = (float(v) - 0.5, float(v) + 0.5)
                else:
                    # Exact number of used categories/colors in the scalar bar.
                    mesh_kwargs["cmap"] = mpl_cm.get_cmap(cmap, int(discrete_values.size))
            else:
                mesh_kwargs["cmap"] = cmap

            actor = pl.add_mesh(mesh, **mesh_kwargs)
            colorbar_mapper = actor.mapper

            # Draw parcel outlines
            if border_color is not None:
                outline_scalars = vertex_labels[:32492] if h == "left" else vertex_labels[32492:]
                border_lines = _parcel_border_lines(mesh, outline_scalars)
                pl.add_mesh(border_lines, color=border_color, line_width=border_width, 
                            render_lines_as_tubes=True, lighting=True, show_scalar_bar=False)

        # Camera position
        cam_pos = list(center)
        cam_pos[axis_idx[axis]] += sign * distance
        pl.camera_position = [tuple(cam_pos), center, up]

    # Plot the colorbar
    if show_shared_colorbar and colorbar_mapper is not None:
        add_scalar_bar = cast(Any, pl.add_scalar_bar)
        if colorbar == "bottom":
            cb_row, cb_col = panel_nrows, 0
            pl.subplot(cb_row, cb_col)
            n_labels = discrete_nlabels if discrete_values is not None else 4
            fmt = "%.0f" if discrete_values is not None else "%.3g"
            add_scalar_bar(title=colorbar_label, mapper=colorbar_mapper, vertical=False, width=0.33, height=0.7,
               position_x=0.33, position_y=0.1, n_labels=n_labels, fmt=fmt, label_font_size=int(labelsize), title_font_size=int(labelsize*1.2))
        else:
            cb_row, cb_col = 0, panel_ncols
            pl.subplot(cb_row, cb_col)
            n_labels = discrete_nlabels if discrete_values is not None else 4
            fmt = "%.0f" if discrete_values is not None else "%.3g"
            add_scalar_bar(title=colorbar_label, mapper=colorbar_mapper, vertical=True, width=0.7, height=0.25,
                           position_x=0.1, position_y=0.33, n_labels=n_labels, fmt=fmt, label_font_size=int(labelsize), title_font_size=int(labelsize*1.2))

    # Show static/interactive figure
    if interactive:
        pl.show(auto_close=False)
    else:
        if _in_notebook():
            pl.show(jupyter_backend="static")
        else:
            pl.show()

    # Save the figure
    if fname is not None:
        if fname.endswith(("svg", "pdf", "eps", "ps")):
            pl.save_graphic(fname, raster=False)
        elif fname.endswith(("png", "jpeg", "jpg", "bmp", "tif", "tiff")):
            pl.screenshot(fname)
    else:
        pass
   
    return pl.close()

def subcortical_plot(node_values: np.ndarray | list[float] | None = None,
                     scale: str = "S1",
                     surface: str | None = "inflated",
                     view_names: tuple[str, ...] = ("lateral", "medial"),
                     ncols: int | None = None,
                     colwise: bool = True,
                     cmap: str = "viridis",
                     nan_color: str = "lightgray",
                     nan_alpha: float = 0.0,
                     surface_color: str = "lightgray",
                     surface_alpha: float = 0.10,
                     smooth_iter: int = 20,
                     smooth_relaxation: float = 0.2,
                     distance: float = 600.0,
                     size: list[int] | None = None,
                     labelsize: int = 18,
                     colorbar: None | str = "bottom",
                     colorbar_label: str | None = None,
                     interactive: bool = True,
                     fname: str | None = None):
    """
    Plot Tian subcortical structures for scale ``S1`` to ``S4``.

    If ``node_values`` contains cortex + subcortex values from ``parcellate(..., subcortical=scale)``,
    the first N values are used automatically, where N is the number of Tian meshes for that scale.
    The combined atlases place the subcortical parcels in the leading columns, and both the parcels
    and the meshes are ordered by Tian region id, so the i-th value maps to the i-th mesh.
    """
    meshes = _get_subcortical(scale=scale, smooth_iter=smooth_iter, smooth_relaxation=smooth_relaxation)
    mesh_names = sorted(meshes)
    n_sub = len(mesh_names)

    values = None
    if node_values is not None:
        values = np.asarray(node_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("node_values must be a 1D array-like or None.")
        if values.size < n_sub:
            raise ValueError(f"node_values must contain at least {n_sub} values for Tian {scale}.")
        values = values[:n_sub]

    panels = list(view_names)
    base_cams = {
        "lateral":   ("x", +1, (0, 0, 1)),
        "medial":    ("x", -1, (0, 0, 1)),
        "anterior":  ("y", -1, (0, 0, 1)),
        "posterior": ("y", +1, (0, 0, 1)),
        "superior":  ("z", -1, (0, 1, 0)),
        "inferior":  ("z", +1, (0, 1, 0))
    }
    unknown_views = sorted({v for v in panels if v not in base_cams})
    if unknown_views:
        raise ValueError(f"Unknown view(s): {unknown_views}. Available: {list(base_cams.keys())}")

    n = len(panels)
    if ncols is None:
        ncols = min(3, n) if n <= 6 else int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    if colwise:
        ncols = int(math.ceil(n / nrows))

    clim = None
    if values is not None:
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            clim = (float(np.nanmin(finite)), float(np.nanmax(finite)))
            if clim[0] == clim[1]:
                eps = 1e-12 if clim[0] == 0.0 else abs(clim[0]) * 1e-12
                clim = (clim[0] - eps, clim[1] + eps)

    show_shared_colorbar = (values is not None) and (clim is not None) and (colorbar is not None)
    panel_nrows, panel_ncols = nrows, ncols
    row_weights = None
    col_weights = None
    groups = None
    if show_shared_colorbar and colorbar == "bottom":
        plot_shape = (panel_nrows + 1, panel_ncols)
        groups = [([panel_nrows], list(range(panel_ncols)))]
        row_weights = [1.0] * panel_nrows + [0.2]
    elif show_shared_colorbar and colorbar == "right":
        plot_shape = (panel_nrows, panel_ncols + 1)
        groups = [(list(range(panel_nrows)), [panel_ncols])]
        col_weights = [1.0] * panel_ncols + [0.2]
    else:
        plot_shape = (panel_nrows, panel_ncols)

    context_meshes = _get_surface(surface=surface) if surface is not None else {}

    pv.global_theme.font.family = "times"
    pl = pv.Plotter(shape=plot_shape, window_size=size, title=f"Comet Toolbox Tian {scale} Viewer", border=False,
                    notebook=_in_notebook() and not interactive, off_screen=not interactive,
                    row_weights=row_weights, col_weights=col_weights, groups=groups)
    pl.enable_anti_aliasing("msaa")

    colorbar_mapper = None
    axis_idx = {"x": 0, "y": 1, "z": 2}
    all_centers = [np.asarray(mesh.center) for mesh in meshes.values()]
    for mesh in context_meshes.values():
        all_centers.append(np.asarray(mesh.center))
    center = np.mean(np.vstack(all_centers), axis=0) if all_centers else np.zeros(3, dtype=float)

    for i, view_name in enumerate(panels):
        row, col = (i % panel_nrows, i // panel_nrows) if colwise else (i // panel_ncols, i % panel_ncols)
        axis, sign, up = base_cams[view_name]

        pl.subplot(row, col)
        pl.add_text(view_name, font_size=int(labelsize * 0.7))

        for mesh in context_meshes.values():
            pl.add_mesh(mesh, color=surface_color, opacity=surface_alpha, show_scalar_bar=False,
                        smooth_shading=True)

        for j, name in enumerate(mesh_names):
            mesh = meshes[name]
            if values is None:
                actor = pl.add_mesh(mesh, color=surface_color, opacity=1.0, show_scalar_bar=False,
                                    smooth_shading=True)
            else:
                val = float(values[j])
                mesh["Data"] = np.full(mesh.n_points, val, dtype=float)
                actor = pl.add_mesh(mesh, scalars="Data", cmap=cmap, clim=clim, nan_color=nan_color,
                                    opacity=1.0 if np.isfinite(val) else nan_alpha, show_scalar_bar=False,
                                    smooth_shading=True)
            colorbar_mapper = actor.mapper

        cam_pos = list(center)
        cam_pos[axis_idx[axis]] += sign * distance
        pl.camera_position = [tuple(cam_pos), tuple(center), up]
        pl.hide_axes()

    if show_shared_colorbar and colorbar_mapper is not None:
        add_scalar_bar = cast(Any, pl.add_scalar_bar)
        if colorbar == "bottom":
            pl.subplot(panel_nrows, 0)
            add_scalar_bar(title=colorbar_label, mapper=colorbar_mapper, vertical=False, width=0.33, height=0.7,
                           position_x=0.33, position_y=0.1, n_labels=4, fmt="%.3g",
                           label_font_size=int(labelsize), title_font_size=int(labelsize * 1.2))
        else:
            pl.subplot(0, panel_ncols)
            add_scalar_bar(title=colorbar_label, mapper=colorbar_mapper, vertical=True, width=0.7, height=0.25,
                           position_x=0.1, position_y=0.33, n_labels=4, fmt="%.3g",
                           label_font_size=int(labelsize), title_font_size=int(labelsize * 1.2))

    if interactive:
        pl.show(auto_close=False)
    else:
        if _in_notebook():
            pl.show(jupyter_backend="static")
        else:
            pl.show()

    if fname is not None:
        if fname.endswith(("svg", "pdf", "eps", "ps")):
            pl.save_graphic(fname, raster=False)
        elif fname.endswith(("png", "jpeg", "jpg", "bmp", "tif", "tiff")):
            pl.screenshot(fname)

    return pl.close()

def _get_surface(surface: str = "very_inflated") -> dict[str, pv.PolyData]:
    """
    Download (if needed) and load fs_LR 32k cortical surfaces
    from the CBIG template repository and return them as PyVista meshes.

    Parameters
    ----------
    surface : str, default="very_inflated"
        Surface type. Options:
        - "midthickness_orig"
        - "midthickness_mni"
        - "inflated"
        - "very_inflated"
        - "super_inflated"
        - "sphere"

    Returns
    -------
    dict
        Dictionary containing the loaded hemispheres:
        {"left":  pv.PolyData,
         "right": pv.PolyData}

    Notes
    -----
    Surface files are downloaded from:
    https://github.com/ThomasYeoLab/CBIG

    Files are cached in the `comet.data.surf` resource directory.
    """
    base_url = ("https://github.com/ThomasYeoLab/CBIG/raw/master/data/templates/surface/fs_LR_32k")

    # CBIG filenames in that folder
    lh_name = f"fsaverage.L.{surface}.32k_fs_LR.surf.gii"
    rh_name = f"fsaverage.R.{surface}.32k_fs_LR.surf.gii"

    lh_url = f"{base_url}/{lh_name}"
    rh_url = f"{base_url}/{rh_name}"

    # Store in resources
    with importlib_resources.path("comet.data.surf", lh_name) as lh_path:
        if not lh_path.exists():
            urllib.request.urlretrieve(lh_url, lh_path)
        lh_path = str(lh_path)

    with importlib_resources.path("comet.data.surf", rh_name) as rh_path:
        if not rh_path.exists():
            urllib.request.urlretrieve(rh_url, rh_path)
        rh_path = str(rh_path)

    meshes = {}
    # VTK requires faces stored in a single array structured as:
    # [n_points, v0, v1, v2,  n_points, v0, v1, v2, ...]
    vertices, triangles = nib.load(lh_path).agg_data()
    meshes["left"] = pv.make_tri_mesh(vertices, triangles)

    vertices, triangles = nib.load(rh_path).agg_data()
    meshes["right"] = pv.make_tri_mesh(vertices, triangles)

    return meshes

def _get_subcortical(scale: str, smooth_iter: int = 20, smooth_relaxation: float = 0.2) -> dict[str, pv.PolyData]:
    """Download/build cached Tian subcortical meshes for one scale."""
    scale = scale.upper()
    if scale not in {"S1", "S2", "S3", "S4"}:
        raise ValueError("scale must be one of 'S1', 'S2', 'S3', 'S4'.")

    with importlib_resources.path("comet.data.surf", ".") as surf_root:
        cache_root = Path(surf_root) / "subcortex"
    cache_root.mkdir(parents=True, exist_ok=True)
    nii_path = cache_root / f"Tian_Subcortex_{scale}_3T_2009cAsym.nii.gz"
    txt_path = cache_root / f"Tian_Subcortex_{scale}_3T_label.txt"
    mesh_dir = cache_root / "meshes" / scale.lower()
    mesh_dir.mkdir(parents=True, exist_ok=True)

    if not nii_path.exists():
        url = f"https://raw.githubusercontent.com/yetianmed/subcortex/master/Group-Parcellation/3T/Subcortex-Only/{nii_path.name}"
        nii_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, nii_path)
    if not txt_path.exists():
        url = f"https://raw.githubusercontent.com/yetianmed/subcortex/master/Group-Parcellation/3T/Subcortex-Only/{txt_path.name}"
        try:
            urllib.request.urlretrieve(url, txt_path)
        except Exception:
            txt_path = None

    mesh_files = sorted(mesh_dir.glob("*.vtk"))
    if not mesh_files:
        labels: dict[int, str] = {}
        if txt_path is not None and txt_path.exists():
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.replace("\t", " ").split()
                    if len(parts) < 2:
                        continue
                    try:
                        labels[int(parts[0])] = parts[1]
                    except ValueError:
                        continue

        img = nib.load(str(nii_path))
        data = np.asarray(img.get_fdata(), dtype=np.int32)
        ids = np.unique(data)
        ids = ids[ids > 0]
        for label_id in ids:
            mask = data == label_id
            if not np.any(mask):
                continue
            label = labels.get(int(label_id), f"Region_{int(label_id):03d}")
            fname = f"{int(label_id):03d}_{label}.vtk"

            grid = pv.ImageData(dimensions=np.array(mask.shape))
            grid.point_data["values"] = np.ascontiguousarray(mask.astype(np.uint8)).ravel(order="F")
            mesh = grid.contour(isosurfaces=[0.5], scalars="values")
            mesh = mesh.triangulate().clean()
            mesh.points = nib.affines.apply_affine(img.affine, mesh.points)
            if smooth_iter > 0:
                mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=smooth_relaxation)
            mesh.compute_normals(inplace=True)
            mesh.save(mesh_dir / fname)
        mesh_files = sorted(mesh_dir.glob("*.vtk"))

    return {f.stem: pv.read(f) for f in mesh_files}

def _parcel_border_lines(mesh, parc_labels, offset: float = 0.10, 
                         smooth_iters: int = 2, decimate_step: int = 2):
    """Build a line mesh representing parcel boundaries."""
    parc_labels = np.asarray(parc_labels)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    edges = np.vstack([e0, e1, e2])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    edge_labels = parc_labels[edges]
    border_edges = edges[edge_labels[:, 0] != edge_labels[:, 1]]
    if border_edges.size == 0:
        return pv.PolyData()

    # Prune dangling branches by repeatedly removing degree-1 endpoints.
    while border_edges.size > 0:
        vertex_degree = np.bincount(border_edges.ravel(), minlength=mesh.n_points)
        keep = (vertex_degree[border_edges[:, 0]] > 1) & (vertex_degree[border_edges[:, 1]] > 1)
        new_border_edges = border_edges[keep]
        if new_border_edges.shape[0] == border_edges.shape[0]:
            break
        border_edges = new_border_edges
    if border_edges.size == 0:
        return pv.PolyData()

    points = np.asarray(mesh.points).copy()
    if offset != 0.0:
        normals_mesh = mesh.compute_normals(point_normals=True, cell_normals=False, inplace=False)
        normals = np.asarray(normals_mesh.point_data["Normals"])
        points += offset * normals

    polylines = _edges_to_polylines(border_edges)

    out_points: list[np.ndarray] = []
    out_lines: list[np.ndarray] = []
    next_id = 0
    for chain in polylines:
        chain_idx = np.asarray(chain, dtype=np.int64)
        coords = points[chain_idx]
        is_closed = bool(chain[0] == chain[-1])
        
        coords = _decimate_polyline(coords, step=decimate_step, closed=is_closed)
        coords = _chaikin_smooth(coords, iterations=smooth_iters, closed=is_closed)

        out_points.append(coords)
        ids = np.arange(next_id, next_id + coords.shape[0], dtype=np.int64)
        out_lines.append(np.concatenate(([coords.shape[0]], ids)))
        next_id += coords.shape[0]

    line_mesh = pv.PolyData()
    line_mesh.points = np.vstack(out_points)
    line_mesh.lines = np.concatenate(out_lines)
    return line_mesh

def _edges_to_polylines(edges: np.ndarray) -> list[list[int]]:
    """Convert undirected edges into connected polylines/cycles."""
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        ui, vi = int(u), int(v)
        adj.setdefault(ui, []).append(vi)
        adj.setdefault(vi, []).append(ui)

    visited: set[tuple[int, int]] = set()
    polylines: list[list[int]] = []

    def _edge(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    starts = [node for node, neigh in adj.items() if len(neigh) != 2]
    for start in starts:
        for nxt in adj[start]:
            e = _edge(start, nxt)
            if e in visited:
                continue
            visited.add(e)
            chain = [start, nxt]
            prev, cur = start, nxt
            while True:
                neigh = adj[cur]
                if len(neigh) != 2:
                    break
                cand = neigh[0] if neigh[1] == prev else neigh[1]
                e2 = _edge(cur, cand)
                if e2 in visited:
                    break
                visited.add(e2)
                chain.append(cand)
                prev, cur = cur, cand
            polylines.append(chain)

    for u, v in edges:
        start, nxt = int(u), int(v)
        e = _edge(start, nxt)
        if e in visited:
            continue
        visited.add(e)
        chain = [start, nxt]
        prev, cur = start, nxt
        while True:
            neigh = adj[cur]
            cand = neigh[0] if neigh[1] == prev else neigh[1]
            e2 = _edge(cur, cand)
            if e2 in visited:
                if cand == start:
                    chain.append(cand)
                break
            visited.add(e2)
            chain.append(cand)
            prev, cur = cur, cand
        polylines.append(chain)

    return polylines

def _decimate_polyline(points: np.ndarray, step: int, closed: bool) -> np.ndarray:
    """Keep every Nth point in a polyline."""
    n = points.shape[0]
    if step <= 1 or n <= 2:
        return points

    if closed:
        core = points[:-1] if np.allclose(points[0], points[-1]) else points
        if core.shape[0] <= 3:
            out = core
        else:
            idx = np.arange(0, core.shape[0], step, dtype=int)
            if idx[-1] != core.shape[0] - 1:
                idx = np.append(idx, core.shape[0] - 1)
            out = core[idx]
        return np.vstack([out, out[0]])

    idx = np.arange(0, n, step, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return points[idx]

def _chaikin_smooth(points: np.ndarray, iterations: int, closed: bool) -> np.ndarray:
    """Smooth the polylines."""
    pts = points
    for _ in range(iterations):
        if pts.shape[0] < 3:
            break
        if closed:
            core = pts[:-1] if np.allclose(pts[0], pts[-1]) else pts
            nxt = np.roll(core, -1, axis=0)
            q = 0.75 * core + 0.25 * nxt
            r = 0.25 * core + 0.75 * nxt
            out = np.empty((2 * core.shape[0], 3), dtype=core.dtype)
            out[0::2] = q
            out[1::2] = r
            pts = np.vstack([out, out[0]])
        else:
            out = np.empty((2 * (pts.shape[0] - 1) + 2, 3), dtype=pts.dtype)
            out[0] = pts[0]
            j = 1
            for i in range(pts.shape[0] - 1):
                p0, p1 = pts[i], pts[i + 1]
                out[j] = 0.75 * p0 + 0.25 * p1
                out[j + 1] = 0.25 * p0 + 0.75 * p1
                j += 2
            out[-1] = pts[-1]
            pts = out
    return pts

def _in_notebook():
    """Check if the code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except Exception:
        return False
    return True
