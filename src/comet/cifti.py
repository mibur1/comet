import urllib
import numpy as np
import nibabel as nib
nib.imageglobals.logger.setLevel(40)
import importlib_resources
from scipy.io import loadmat

def parcellate(dtseries:str|nib.cifti2.cifti2.Cifti2Image, 
               atlas         : str="schaefer", 
               resolution    : int=100, 
               networks      : int=7, 
               subcortical   : None|str="S1",
               kong          : bool=False,
               standardize   : bool=True,
               method        = np.mean,
               return_labels : bool=False,
               debug         : bool=False
    ) -> np.ndarray | tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """ 
    Parcellate cifti data (.dtseries.nii) using a given atlas. 
    66 different atlases are available and will be downloaded on demand (see References).
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

    networks : int
        Number of networks in the atlas. Only used with Schaefer atlas.
        Available options are: 7, 17
    
    subcortical : None or string
        If a string containing the scale is provided, the Tian subcortical parcels are included.
        Available options are: 'S1' (16 ROIs), 'S2' (32 ROIs), 'S3' (50 ROIs), 'S4' (54 ROIs).
    
    kong : bool
        Use the Kong 2022 version of the Schaefer atlas (only for Schaefer cortical atlas with 17 networks).
        Reference: https://doi.org/10.1093/cercor/bhab101

    standardize : bool
        Standardize the time series to zero (temporal) mean and unit
        standard deviation before(!) parcellation.

    method : function
        Aggregation function to use for parcellation. Default (and the only tested function) is np.mean

    debug : bool
        Flag to provide additional debugging information. Default is False.

    Returns
    -------
    ts_parc : TxP np.ndarray or tuple
        If return_labels is False (default):
            - ts_parc : TxP np.ndarray
                Parcellated time series data
        If return_labels is True, a tuple containing:
            - ts_parc : TxP np.ndarray
                Parcellated time series data
            - labels : list
                List of label names for each parcel
            - rgba : list
                RGBA values of each label
            - rois : np.ndarray
                ROI indices for each vertex in the CIFTI
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

    rois, keys, labels, rgba = _get_atlas(atlas=atlas, resolution=resolution, networks=networks, subcortical=subcortical, kong=kong, debug=debug)
    
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
    ts_parc = np.zeros((len(ts), len(labels)), dtype=ts.dtype)

    i = 0
    for k, lab in zip(keys, labels):
        if k == 0:
            continue

        mask = (rois == k)
        if not np.any(mask):
            ts_parc[:, i] = 0
            i += 1
            print(f"[WARN] ROI {lab} is empty and was set to zero.")
            continue

        ts_parc[:, i] = method(ts[:, mask], axis=1)
        i += 1

    return (ts_parc, labels, rgba, rois) if return_labels else ts_parc

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

    # Check input parameters
    if resolution not in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        raise ValueError(f"Resolution '{resolution}' not available. Please choose from [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000].")
    if resolution not in  [100, 200, 400] and subcortical is not None:
        raise ValueError(f"Schaefer + Tian subcortical parcellation is only available for resolutions 100, 200, and 400.")
    if atlas != "schaefer" and subcortical is not None and networks != 17 and kong is True:
        raise ValueError(f"Kong 2022 version is only available for Schaefer atlases with 17 networks and without subcortical parcels.")
    if atlas == "schaefer" and networks == 7 and kong is True:
        raise ValueError(f"Kong 2022 version is only available for Schaefer atlases with 17 networks and without subcortical parcels.")
    if networks not in [7, 17]:
        raise ValueError(f"Networks '{networks}' not available. Please choose from [7, 17].")
    if atlas in ["glasser", "gordon"] and subcortical is None:
        raise ValueError(f"Atlas '{atlas}' includes subcortical parcels. Please provide a valid subcortical scale.")
    if resolution == 100 and networks == 17 and atlas == "schaefer" and subcortical is not None:
        raise ValueError(f"Schaefer 100 not available for 17 networks and subcortical parcels. Use 7 networks or cortical only instead.")
    if subcortical is not None and subcortical not in ['S1', 'S2', 'S3', 'S4']:
        raise ValueError(f"Subcortical scale '{subcortical}' not available. Please choose from ['S1', 'S2', 'S3', 'S4'] or set to None.")
    if subcortical is None and atlas == "glasser":
        raise ValueError(f"Glasser atlas requires subcortical parcels. Please provide a valid subcortical scale.")
    if not isinstance(debug, bool):
        raise ValueError(f"Debug flag must be a boolean value (True or False).")
    
    # All checks passed, prepare the atlas url
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
