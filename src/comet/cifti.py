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

_SHADING_PRESETS: dict[str, dict] = {
    "default": {},                                                                    # pyvista defaults
    "matte":   dict(lighting=True, ambient=0.65, diffuse=0.4, specular=0.0, specular_power=15),
    "glossy":  dict(lighting=True, ambient=0.3, diffuse=0.7, specular=0.3, specular_power=30),
    "flat":    dict(lighting=False),
}
_SHARPEN_CACHE: dict[tuple, tuple] = {}


def _shading(style: str) -> dict:
    if style not in _SHADING_PRESETS:
        raise ValueError(f"Unknown style '{style}'. Available: {list(_SHADING_PRESETS)}")
    return dict(_SHADING_PRESETS[style])


def _set_camera(pl, center, axis: str, sign: int, up, projection: str, distance: float, zoom: float):
    """Place the camera of the active subplot on the given side of ``center``."""
    axis_idx = {"x": 0, "y": 1, "z": 2}
    cam_pos = list(np.asarray(center, dtype=float))
    cam_pos[axis_idx[axis]] += sign * distance
    pl.camera_position = [tuple(cam_pos), tuple(np.asarray(center, dtype=float)), up]
    if projection == "parallel":
        pl.camera.parallel_projection = True
        pl.reset_camera()            # fit everything in the subplot, keeping the viewing direction
        pl.camera.zoom(zoom)
    elif projection != "perspective":
        raise ValueError("projection must be 'parallel' or 'perspective'.")


def _crop_to_content(img: np.ndarray, pad: int = 8) -> np.ndarray:
    """Crop a screenshot to its content (alpha > 0 for RGBA, non-white for RGB)."""
    mask = img[..., 3] > 0 if img.shape[-1] == 4 else np.any(img[..., :3] < 250, axis=-1)
    if not mask.any():
        return img
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def _finish_plot(pl, interactive: bool, fname: str | None, transparent: bool, scale: int, crop: bool,
                 return_array: bool):
    """Show, save and/or return the rendered figure (shared by surface_plot and subcortical_plot)."""
    if interactive:
        pl.show(auto_close=False)
    elif not return_array:           # a returned array is meant to be composed by the caller
        if _in_notebook():
            pl.show(jupyter_backend="static", auto_close=False)
        else:
            pl.show(auto_close=False)

    img = None
    if fname is not None and fname.endswith(("svg", "pdf", "eps", "ps")):
        if transparent:
            print("Warning: transparent backgrounds are not supported for vectorised output, saving opaque.")
        pl.save_graphic(fname, raster=False)
    elif fname is not None or return_array:
        img = pl.screenshot(None, transparent_background=transparent, return_img=True, scale=scale)
        if crop:
            img = _crop_to_content(img)
        if fname is not None:
            from PIL import Image
            Image.fromarray(img).save(fname)
    pl.close()
    return img if return_array else None


def _sharpen_parcels(mesh: pv.PolyData, labels: np.ndarray, levels: int = 2, smooth_iter: int = 4):
    """Subdivide a hemisphere mesh and relabel the finer vertices from neighbourhood-smoothed parcel
    masks, so that parcel borders become smooth curves instead of vertex-level stair-steps.

    Returns the subdivided mesh and an integer label per new vertex (0 = medial wall / background).
    The masks are kept sparse, so this works for any number of parcels.
    """
    import scipy.sparse as sp
    pts = np.asarray(mesh.points, dtype=np.float64)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    n, n_lab = pts.shape[0], int(labels.max()) + 1
    masks = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, n_lab))

    # neighbourhood averaging of the one-hot parcel masks on the original mesh
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = sp.csr_matrix((np.ones(i.size), (i, j)), shape=(n, n))
    adj.data[:] = 1.0
    adj = adj + sp.eye(n, format="csr")
    weights = sp.diags(1.0 / np.asarray(adj.sum(axis=1)).ravel()) @ adj
    for _ in range(smooth_iter):
        masks = weights @ masks

    # linear subdivision: every new vertex is an edge midpoint whose mask is the mean of its endpoints
    for _ in range(levels):
        edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        keys, inv = np.unique(edges[:, 0] * n + edges[:, 1], return_inverse=True)   # 1-D unique is much faster
        edges = np.column_stack([keys // n, keys % n])
        m_ab, m_bc, m_ca = np.asarray(inv).ravel().reshape(3, -1) + n
        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        faces = np.vstack([np.column_stack([a, m_ab, m_ca]), np.column_stack([b, m_bc, m_ab]),
                           np.column_stack([c, m_ca, m_bc]), np.column_stack([m_ab, m_bc, m_ca])])
        pts = np.vstack([pts, (pts[edges[:, 0]] + pts[edges[:, 1]]) / 2.0])
        masks = sp.vstack([masks, (masks[edges[:, 0]] + masks[edges[:, 1]]) * 0.5]).tocsr()
        n = pts.shape[0]

    new_labels = np.asarray(masks.argmax(axis=1)).ravel().astype(np.int64)
    sub = pv.PolyData(pts, np.column_stack([np.full(faces.shape[0], 3, dtype=np.int64), faces]).ravel())
    return sub, new_labels


def _sharpen_parcels_cached(surface: str, hemi: str, mesh: pv.PolyData, labels: np.ndarray):
    """_sharpen_parcels with an in-memory cache keyed by surface, hemisphere and label content.
    The cache holds plain arrays (a cached PolyData would be torn down after VTK at interpreter exit)."""
    import hashlib
    key = (surface, hemi, hashlib.sha1(np.ascontiguousarray(labels).tobytes()).hexdigest())
    if key not in _SHARPEN_CACHE:
        sub, sub_labels = _sharpen_parcels(mesh, labels)
        _SHARPEN_CACHE[key] = (np.asarray(sub.points), np.asarray(sub.faces), sub_labels)
    pts, faces, sub_labels = _SHARPEN_CACHE[key]
    return pv.PolyData(pts.copy(), faces.copy()), sub_labels


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
                 anatomical_extent: bool=True,
                 view_names: tuple[str, ...]=("lateral", "medial"), 
                 ncols: int|None=None, 
                 colwise: bool=True, 
                 cmap: str="viridis", 
                 clim: tuple[float, float]|None=None,
                 border_color: None|str=None, 
                 border_width: int=5,
                 surface_color: str="white",
                 surface_alpha: float=1.0,
                 sharpen: bool=True,
                 style: str="matte",
                 projection: str="parallel",
                 zoom: float=1.3,
                 distance: float=400.0, 
                 titles: bool=True,
                 size : list[int]|None=None,
                 labelsize : int=18,
                 colorbar: None|str="bottom",
                 colorbar_label : str|None=None,
                 interactive : bool=True,
                 fname : str|None=None,
                 transparent : bool=False,
                 scale: int=1,
                 crop: bool=False,
                 return_array: bool=False):
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
    anatomical_extent : bool
        Fit inflated surfaces to the anatomical (midthickness_mni) size and position, default True.
        The raw inflated surfaces are ~1.4x larger and centred on the midline.
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
    clim : tuple or None
        Colour limits (min, max). Default is the data range. For integer maps (e.g. network ids) pass
        the full id range, e.g. ``clim=(1, 7)``, so that a map containing a single id keeps its colour.
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
    surface_color : str
        Colour of the surface where no data is shown (medial wall, parcels with value 0 or NaN).
    surface_alpha : float
        Opacity of that surface (0-1). Below 1 the data parcels are drawn as a separate opaque actor on
        top of the translucent surface (depth peeling), so parcels on the far side shine through.
    sharpen : bool
        Subdivide the surface and smooth the parcel borders (default True). Otherwise the borders
        follow the 32k vertices and look jagged.
    style : {"matte", "default", "glossy", "flat"}
        Lighting preset. "matte" (default) is soft without highlights, "default" is pyvista's lighting.
    projection : {"parallel", "perspective"}
        "parallel" (default) fits every panel to the hemisphere without perspective distortion.
        "perspective" places the camera at ``distance``.
    zoom : float
        Zoom factor for the parallel projection (> 1 zooms in).
    distance : float
        Camera distance (perspective projection only).
    titles : bool
        Write "<hemisphere> <view>" into every panel (default True). Set False for clean panels.
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
    transparent : bool
        Save the plot with a transparent background. Raster formats only, as the vectorised
        export has no transparency support. Default is False.
    scale : int
        Resolution multiplier for saved and returned raster images.
    crop : bool
        Crop saved and returned raster images to their content.
    return_array : bool
        Return the rendered image as an RGB(A) uint8 array instead of showing it, e.g. to compose
        several panels in matplotlib.

    Returns
    -------
    numpy.ndarray or None
        The image when ``return_array=True``, else None.
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
    meshes = _get_surface(surface=surface, anatomical_extent=anatomical_extent)  # get the surface mesh(es)
    hemi_order = [hemi] if hemi in ("left", "right") else ["left", "right"] # which hemispheres to plot
    panels = [(h, v) for h in hemi_order for v in view_names]  # list of (hemisphere, view) pairs
    
    # Define default camera positions
    # (axis, side the camera is placed on, up vector) for the LEFT hemisphere; lateral/medial flip for the right
    base_cams = {
        "lateral":   ("x", -1, (0, 0, 1)),
        "medial":    ("x", +1, (0, 0, 1)),
        "anterior":  ("y", +1, (0, 0, 1)),
        "posterior": ("y", -1, (0, 0, 1)),
        "superior":  ("z", +1, (0, 1, 0)),
        "inferior":  ("z", -1, (0, 1, 0))
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
    
    # Parcel labels per hemisphere, optionally on a subdivided mesh with smoothed borders
    hemi_labels = {}
    if vertex_labels is not None:
        hemi_labels = {"left": np.asarray(vertex_labels[:32492]), "right": np.asarray(vertex_labels[32492:])}
        if sharpen:
            for h in hemi_order:
                meshes[h], hemi_labels[h] = _sharpen_parcels_cached(surface, h, meshes[h], hemi_labels[h])

    # Build per-vertex arrays for each hemisphere
    scalars = {}

    if node_values is not None and vertex_labels is not None:
        def _map_to_vertices(parc):
            out = np.full(parc.size, np.nan, dtype=float)
            mask = (parc > 0) & (parc <= node_values.size)
            out[mask] = node_values[parc[mask] - 1]  # labels are 1-based
            return out

        scalars["left"] = _map_to_vertices(hemi_labels["left"])
        scalars["right"] = _map_to_vertices(hemi_labels["right"])
 
    # Use one shared colour scale across all plotted hemispheres
    vals_list = [scalars[h][~np.isnan(scalars[h])] for h in scalars if h in meshes]
    vals_list = [v for v in vals_list if v.size > 0]
    user_clim = None if clim is None else (float(clim[0]), float(clim[1]))
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
            discrete_ref_max = max(prev_max, local_max, int(user_clim[1]) if user_clim else 0)
            _DISCRETE_CMAP_REF_MAX[cmap] = discrete_ref_max
        if user_clim is not None:
            clim = user_clim

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
    if return_array:
        interactive = False

    pv.global_theme.font.family = "times"
    pl = pv.Plotter(shape=plot_shape, window_size=size, title="Comet Toolbox Surface Viewer", border=False, line_smoothing=True,
                    notebook=_in_notebook() and not interactive, off_screen=not interactive, row_weights=row_weights, col_weights=col_weights, groups=groups)

    if surface_alpha < 1.0:
        # Translucency needs depth peeling, which needs a non-multisampled framebuffer (so no MSAA)
        pl.ren_win.SetMultiSamples(0)
        pl.ren_win.SetAlphaBitPlanes(True)
    else:
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
        if titles:
            pl.add_text(f"{h} {v}", font_size=int(labelsize*0.7))
        if surface_alpha < 1.0:
            pl.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
        shading = _shading(style)

        if node_values is None or vertex_labels is None:
            pl.add_mesh(mesh, color="lightgray", opacity=surface_alpha, smooth_shading=True, **shading)
            colorbar_mapper = None
        else:
            values = scalars[h].copy()

            # Make 0 values white by masking them
            zero_mask = values == 0
            values = values.astype(float)
            values[zero_mask] = np.nan  # treat zeros as NaN

            # Plot data to the surface
            mesh_kwargs = dict(scalars=values, clim=clim, nan_color=surface_color, nan_opacity=1.0,
                               show_scalar_bar=False, interpolate_before_map=False, smooth_shading=True,
                               **shading)
            
            if discrete_values is not None:
                # Keep ID->color mapping stable for single-ID subsets.
                if int(discrete_values.size) == 1:
                    v = int(round(float(discrete_values[0])))
                    n_ref = max(int(discrete_ref_max) if discrete_ref_max is not None else v, v, 2)
                    cmap_ref = mpl_cm.get_cmap(cmap, n_ref)
                    mesh_kwargs["cmap"] = ListedColormap([cmap_ref(v - 1)])
                    mesh_kwargs["clim"] = (float(v) - 0.5, float(v) + 0.5)
                else:
                    # One colour per integer of the colour range (the used categories by default).
                    n_col = int(round(clim[1] - clim[0])) + 1 if user_clim else int(discrete_values.size)
                    mesh_kwargs["cmap"] = mpl_cm.get_cmap(cmap, max(n_col, 2))
            else:
                mesh_kwargs["cmap"] = cmap

            if surface_alpha < 1.0:
                # Two actors: one actor cannot mix opacities per vertex. A triangle counts as data when all
                # three vertices carry a value, so no NaN colour bleeds into the data actor.
                faces = mesh.faces.reshape(-1, 4)[:, 1:]
                is_data = np.all(~np.isnan(values)[faces], axis=1)
                carrier = mesh.copy(deep=False)
                carrier.point_data["values"] = values
                data_mesh = carrier.extract_cells(np.flatnonzero(is_data)).extract_surface(algorithm="dataset_surface")
                surf_mesh = carrier.extract_cells(np.flatnonzero(~is_data)).extract_surface(algorithm="dataset_surface")
                pl.add_mesh(surf_mesh, color=surface_color, opacity=surface_alpha, smooth_shading=True,
                            show_scalar_bar=False, **shading)
                mesh_kwargs["scalars"] = "values"
                actor = pl.add_mesh(data_mesh, **mesh_kwargs)
            else:
                actor = pl.add_mesh(mesh, **mesh_kwargs)
            colorbar_mapper = actor.mapper

            # Draw parcel outlines
            if border_color is not None:
                border_lines = _parcel_border_lines(mesh, hemi_labels[h])
                pl.add_mesh(border_lines, color=border_color, line_width=border_width, 
                            render_lines_as_tubes=True, lighting=True, show_scalar_bar=False)

        _set_camera(pl, center, axis, sign, up, projection, distance, zoom)

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

    return _finish_plot(pl, interactive, fname, transparent, scale, crop, return_array)

def subcortical_plot(node_values: np.ndarray | list[float] | None = None,
                     scale: str = "S1",
                     hemi: str = "both",
                     surface: str | None = "inflated",
                     anatomical_extent: bool = True,
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
                     style: str = "matte",
                     projection: str = "parallel",
                     zoom: float = 1.3,
                     distance: float = 600.0,
                     titles: bool = True,
                     size: list[int] | None = None,
                     labelsize: int = 18,
                     colorbar: None | str = "bottom",
                     colorbar_label: str | None = None,
                     interactive: bool = True,
                     fname: str | None = None,
                     transparent: bool = False,
                     scale_factor: int = 1,
                     crop: bool = False,
                     return_array: bool = False):
    """
    Plot Tian subcortical structures for scale ``S1`` to ``S4``.

    If ``node_values`` contains cortex + subcortex values from ``parcellate(..., subcortical=scale)``,
    the first N values are used automatically, where N is the number of Tian meshes for that scale.
    The combined atlases place the subcortical parcels in the leading columns, and both the parcels
    and the meshes are ordered by Tian region id, so the i-th value maps to the i-th mesh.

    ``anatomical_extent`` fits an inflated context surface to the anatomical size and position (see
    ``surface_plot``), so the structures sit inside it where they belong.
    ``hemi`` ("both", "left", "right") selects the structures and the context hemisphere to show;
    lateral/medial views then follow that hemisphere. ``style``, ``projection``, ``zoom``, ``titles``,
    ``scale_factor`` (image resolution), ``crop`` and ``return_array`` work as in ``surface_plot``.

    ``transparent=True`` saves the plot with a transparent background. Raster formats only,
    as the vectorised export has no transparency support.
    """
    meshes = _get_subcortical(scale=scale, smooth_iter=smooth_iter, smooth_relaxation=smooth_relaxation)
    all_names = sorted(meshes)
    n_sub = len(all_names)
    if hemi not in ("both", "left", "right"):
        raise ValueError("hemi must be 'both', 'left' or 'right'.")
    side_suffix = {"left": "-lh", "right": "-rh"}
    mesh_names = [m for m in all_names if hemi == "both" or m.endswith(side_suffix[hemi])]
    value_index = {m: j for j, m in enumerate(all_names)}

    values = None
    if node_values is not None:
        values = np.asarray(node_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("node_values must be a 1D array-like or None.")
        if values.size < n_sub:
            raise ValueError(f"node_values must contain at least {n_sub} values for Tian {scale}.")
        values = values[:n_sub]

    panels = list(view_names)
    # (axis, side the camera is placed on, up vector); lateral = seen from the left, medial = from the right
    base_cams = {
        "lateral":   ("x", -1, (0, 0, 1)),
        "medial":    ("x", +1, (0, 0, 1)),
        "anterior":  ("y", +1, (0, 0, 1)),
        "posterior": ("y", -1, (0, 0, 1)),
        "superior":  ("z", +1, (0, 1, 0)),
        "inferior":  ("z", -1, (0, 1, 0))
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

    context_meshes = _get_surface(surface=surface, anatomical_extent=anatomical_extent) if surface is not None else {}
    context_meshes = {k: v for k, v in context_meshes.items() if hemi == "both" or k == hemi}

    pv.global_theme.font.family = "times"
    pl = pv.Plotter(shape=plot_shape, window_size=size, title=f"Comet Toolbox Tian {scale} Viewer", border=False,
                    notebook=_in_notebook() and not interactive, off_screen=not interactive,
                    row_weights=row_weights, col_weights=col_weights, groups=groups)
    translucent = surface_alpha < 1.0 or nan_alpha < 1.0
    if translucent:
        # Translucency needs depth peeling, which needs a non-multisampled framebuffer (so no MSAA)
        pl.ren_win.SetMultiSamples(0)
        pl.ren_win.SetAlphaBitPlanes(True)
    else:
        pl.enable_anti_aliasing("msaa")

    colorbar_mapper = None
    shading = _shading(style)
    all_centers = [np.asarray(meshes[m].center) for m in mesh_names]
    for mesh in context_meshes.values():
        all_centers.append(np.asarray(mesh.center))
    center = np.mean(np.vstack(all_centers), axis=0) if all_centers else np.zeros(3, dtype=float)

    for i, view_name in enumerate(panels):
        row, col = (i % panel_nrows, i // panel_nrows) if colwise else (i // panel_ncols, i % panel_ncols)
        axis, sign, up = base_cams[view_name]
        if hemi == "right" and view_name in ("lateral", "medial"):
            sign *= -1

        pl.subplot(row, col)
        if titles:
            pl.add_text(view_name if hemi == "both" else f"{hemi} {view_name}", font_size=int(labelsize * 0.7))
        if translucent:
            pl.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)

        for mesh in context_meshes.values():
            pl.add_mesh(mesh, color=surface_color, opacity=surface_alpha, show_scalar_bar=False,
                        smooth_shading=True, **shading)

        for name in mesh_names:
            mesh = meshes[name]
            if values is None:
                actor = pl.add_mesh(mesh, color=surface_color, opacity=1.0, show_scalar_bar=False,
                                    smooth_shading=True, **shading)
            else:
                val = float(values[value_index[name]])
                mesh["Data"] = np.full(mesh.n_points, val, dtype=float)
                actor = pl.add_mesh(mesh, scalars="Data", cmap=cmap, clim=clim, nan_color=nan_color,
                                    opacity=1.0 if np.isfinite(val) else nan_alpha, show_scalar_bar=False,
                                    smooth_shading=True, **shading)
            colorbar_mapper = actor.mapper

        _set_camera(pl, center, axis, sign, up, projection, distance, zoom)
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

    return _finish_plot(pl, interactive, fname, transparent, scale_factor, crop, return_array)

def _get_surface(surface: str = "very_inflated", anatomical_extent: bool = True) -> dict[str, pv.PolyData]:
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
    anatomical_extent : bool, default=True
        The inflated surfaces come enlarged (~1.4x) and centred on the midline. If True, each inflated
        hemisphere is fitted to the bounding box of its midthickness_mni counterpart, so that its size and
        position stay anatomical relative to the other hemisphere and to subcortical meshes (MNI space).

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

    if anatomical_extent and surface in ("inflated", "very_inflated", "super_inflated"):
        ref = _get_surface("midthickness_mni", anatomical_extent=False)
        for h, m in meshes.items():
            b = np.asarray(m.bounds, dtype=float).reshape(3, 2)
            r = np.asarray(ref[h].bounds, dtype=float).reshape(3, 2)
            scale = (r[:, 1] - r[:, 0]) / (b[:, 1] - b[:, 0])
            m.points = (np.asarray(m.points, dtype=float) - b[:, 0]) * scale + r[:, 0]

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
                    if not parts:
                        continue
                    if len(parts) == 1:  # Tian label files: one name per line, the id is the line number
                        labels[len(labels) + 1] = parts[0]
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
