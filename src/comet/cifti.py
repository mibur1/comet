import math
import urllib
import numpy as np
import nibabel as nib
import pyvista as pv
import importlib_resources
from typing import Any, cast
from scipy.io import loadmat
from matplotlib import cm as mpl_cm
from matplotlib.colors import ListedColormap

nib.imageglobals.logger.setLevel(40)
_DISCRETE_CMAP_REF_MAX: dict[str, int] = {}

def parcellate(dtseries:str|nib.cifti2.cifti2.Cifti2Image, 
               atlas         : str="schaefer", 
               resolution    : int=100, 
               subcortical   : None|str=None,
               networks      : int=7, 
               kong          : bool=False,
               standardize   : bool=True,
               method        = np.mean,
               return_labels : bool=False,
               debug         : bool=False
    ) -> np.ndarray | tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """ 
    Parcellate cifti data (.dtseries.nii) using a given atlas.  
    Atlases for many different parameter combinations are available and will be downloaded on demand (see References).  
    If the atlas for the parameter combination is not available, a ValueError is raised.

    References
    ----------
    - Schaefer, Glasser, Gordon (+ Tian subcortical): https://github.com/yetianmed/subcortex
    - Schaefer (cortical only): https://github.com/ThomasYeoLab/CBIG

    Parameters
    ----------
    dtseries : str or nibabel.cifti2.cifti2.Cifti2Image
        string containing a path or nibabel cifti image object
    
    atlas : string
        Name of the atlas to use for parcellation. Available options are:
        - "schaefer": Schaefer et al. (2018) atlas
        - "glasser": Glasser et al. (2016) atlas
        - "gordon": Gordon et al. (2016) atlas
    
    resolution : int
        Number of parcels in the atlas. Only used with Schaefer atlas.
        Available options are: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000. 
    
    subcortical : None or string
        If a string containing the scale is provided, the Tian subcortical parcels are included.
        Available options are: None, 'S1' (16 ROIs), 'S2' (32 ROIs), 'S3' (50 ROIs), 'S4' (54 ROIs).

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
                Parcellated time series data.  
            -  node_labels : list of str  
                Label name for each parcel.  
            -  vertex_labels : np.ndarray  
                ROI index for each vertex in the CIFTI file.  
            -  rgba : list of tuple  
                RGBA colour for each parcel.  

    """
    if isinstance(dtseries, nib.cifti2.cifti2.Cifti2Image):
        ts = dtseries.get_fdata()
    elif isinstance(dtseries, np.ndarray) or isinstance(dtseries, np.memmap):
        ts = dtseries
    elif isinstance(dtseries, str):
        data = nib.load(dtseries)
        ts = data.get_fdata()
    else:
        print("Error: Input must be a nibabel cifti image object or a numpy memmap object")
        return
    
    # Check provided parameters
    if atlas not in ["schaefer", "glasser", "gordon"]:
        raise ValueError(f"Atlas '{atlas}' not available. Please choose from ['schaefer', 'glasser', 'gordon'].")
    if resolution not in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        raise ValueError(f"Resolution '{resolution}' not available. Please choose from [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000].")
    if networks not in [7, 17]:
        raise ValueError(f"Networks '{networks}' not available. Please choose from [7, 17].")
    if subcortical not in [None, 'S1', 'S2', 'S3', 'S4']:
        raise ValueError(f"Subcortical scale '{subcortical}' not available. Please choose from [None, 'S1', 'S2', 'S3', 'S4'].")
    if kong not in [True, False]:
        raise ValueError(f"Kong flag must be a boolean value (True or False).")

    # Check invalid combinations
    if atlas == "schaefer" and resolution not in [100, 200, 400] and subcortical is not None:
        raise ValueError(f"Schaefer + Tian subcortical parcellation is only available for resolutions 100, 200, and 400.")
    if atlas in ["glasser", "gordon"] and subcortical is None:
        raise ValueError(f"Atlas '{atlas}' includes subcortical parcels. Please provide a subcortical scale.")

    # Combinations which automatically adjust parameters with a warning instead of raising an error.
    if atlas == "schaefer" and networks == 7 and kong is True:
        print(f"[WARN] Schaefer Kong version is only available with 17 networks. Networks were set to 17.")
        networks = 17
    if atlas == "schaefer" and resolution == 100 and networks == 17 and subcortical is not None and kong is False:
        print(f"[WARN] Schaefer 100 + Tian subcortical parcellation is only available with 7 networks. Networks were set to 7.")
        networks = 7
    if atlas == "schaefer" and kong is True and subcortical is not None :
        print(f"[WARN] Schaefer Kong atlases are not available with subcortical parcels. 'subcortical' was set to None.")
        subcortical = None

    # Get the atlas    
    vertex_labels, keys, node_labels, rgba = _get_atlas(atlas=atlas, resolution=resolution, networks=networks, subcortical=subcortical, kong=kong, debug=debug)
    
    # Schaefer cortical includes the medial wall which we have to insert into the data
    if atlas == "schaefer" and subcortical is None:
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

    return (ts_parc, node_labels, vertex_labels, rgba) if return_labels else ts_parc

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
            pl.add_mesh(mesh, color="lightgray")
            colorbar_mapper = None
        else:
            values = scalars[h].copy()

            # Make 0 values white by masking them
            zero_mask = values == 0
            values = values.astype(float)
            values[zero_mask] = np.nan  # treat zeros as NaN

            # Plot data to the surface
            mesh_kwargs = dict(scalars=values, clim=clim, nan_color="white", nan_opacity=1.0,
                               show_scalar_bar=False, interpolate_before_map=False)
            
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
        pl.show(jupyter_backend="static")

    # Save the figure
    if fname is not None:
        if fname.endswith(("svg", "pdf", "eps", "ps")):
            pl.save_graphic(fname, raster=False)
        elif fname.endswith(("png", "jpeg", "jpg", "bmp", "tif", "tiff")):
            pl.screenshot(fname)
    else:
        pass
   
    return pl.close()

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

        # Simple subcortical extension (your labels: *-lh / *-rh)
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

# Parcellation helpers
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

    base_urls = {
        "schaefer_c": "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_{parcels}Parcels_{kong}{networks}Networks_order.dlabel.nii",
        "schaefer":   "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Cortex-Subcortex/Schaefer2018_{parcels}Parcels_{networks}Networks_order_Tian_Subcortex_{subcortical}.dlabel.nii",
        "gordon":     "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Cortex-Subcortex/Gordon333.32k_fs_LR_Tian_Subcortex_{subcortical}.dlabel.nii",
        "glasser":    "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Cortex-Subcortex/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_{subcortical}.dlabel.nii",
        #"glasser":   "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors_with_Atlas_ROIs2.32k_fs_LR.dlabel.nii"
    }

    # Prepare the atlas url
    if atlas == "schaefer" and subcortical is None:
        url = base_urls["schaefer_c"].format(parcels=resolution, networks=networks, kong="Kong2022_" if kong else "")
    else:
        url = base_urls[atlas].format(parcels=resolution, networks=networks, subcortical=subcortical)
    filename = url.split("/")[-1]

    # Download the atlas
    with importlib_resources.path("comet.data.atlas", filename) as atlas_path:
        if not atlas_path.exists():
            urllib.request.urlretrieve(url, atlas_path)
            print(f"Atlas not available. Downloading to: {atlas_path}")

        atlas = nib.load(str(atlas_path))

    # Prepare the atlas
    # Usually for dlabel.nii files we have the following header stucture
    #       axis 0: LabelAxis
    #       axix 1: BrainModelAxis
    # print(atlas.header.get_axis(0), atlas.header.get_axis(1))

    # Roi numbers for each vertex
    rois = atlas.dataobj[0].astype(int).squeeze()

    if debug:
        brainmodelAxis = atlas.header.get_axis(1)
        for idx, (name, _slice, _bm) in enumerate(brainmodelAxis.iter_structures()):
            print(idx, str(name), _slice, _bm)

    index_map = atlas.header.get_index_map(0)
    named_map=list(index_map.named_maps)[0]
    keys = []
    labels = []
    rgba = []

    # Iterate over label_table items and get relevat values
    for key, label in named_map.label_table.items():
        if key == 0:
            continue # skip background
        labels.append(label.label)
        rgba.append(label.rgba)
        keys.append(key)

    keys = np.asarray(keys)

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

# Plotting helpers
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
