import urllib
import numpy as np
import nibabel as nib
nib.imageglobals.logger.setLevel(40)
import importlib_resources
from scipy.io import loadmat
from surfplot import Plot

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

def surface_plot(node_values, vertex_labels, surface="super_inflated", size=(800, 600)):
    lh_surf, rh_surf = _get_surface(surface)

    lh_parc = vertex_labels[:32492]
    rh_parc = vertex_labels[32492:]

    lh_data = np.zeros(32492, dtype=float)
    rh_data = np.zeros(32492, dtype=float)

    lh_mask = lh_parc > 0
    rh_mask = rh_parc > 0

    lh_data[lh_mask] = node_values[lh_parc[lh_mask].astype(int) - 1]
    rh_data[rh_mask] = node_values[rh_parc[rh_mask].astype(int) - 1]

    # Create and save the figure
    p = Plot(lh_surf, rh_surf, zoom=1.5, flip=False, brightness=0.8, size=size)
    p.add_layer({"left": lh_data, "right": rh_data}, cmap="viridis", cbar=True)
    p.add_layer({"left": lh_data, "right": rh_data}, as_outline=True, cmap="gray", cbar=False)
    fig = p.build()
    
    return fig

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

def _get_surface(surface: str = "very_inflated") -> tuple[str, str]:
    """
    Helper function: download + load fs_LR 32k surfaces from CBIG.

    Parameters
    ----------
    surface : str
        Surface type (CBIG filenames). Options include:
        - "inflated"
        - "very_inflated"
        - "super_inflated"
        - "midthickness_mni"
        - "midthickness_orig"
        - "sphere"

        These map to files like:
        fsaverage.L.<surface>.32k_fs_LR.surf.gii
        fsaverage.R.<surface>.32k_fs_LR.surf.gii

    Returns
    -------
    lh_surf, rh_surf : str
        Path to surface files.
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

    return lh_path, rh_path

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
