import urllib
import numpy as np
import nibabel as nib
nib.imageglobals.logger.setLevel(40)
import importlib_resources
from scipy.io import loadmat


def parcellate(dtseries, atlas="schaefer_200_cortical", method=np.mean, standardize=True):
    """
    Parcellate cifti data (.dtseries.nii) using a given atlas.

    The parcellated time series is calculated as the mean over all grayordinates within a parcel.

    Parameters
    ----------
    dtseries : nibabel.cifti2.cifti2.Cifti2Image
        nibabel cifti image object

    atlas : string
        name of the atlas to use for parcellation. Options are:
            - schaefer_{x}_cortical     with x = 100, 200, 300 400, 500, 600, 700, 800, 900, 1000
            - schaefer_{x}_subcortical  with x = 100, 200, 300 400, 500, 600, 700, 800, 900, 1000
            - glasser_mmp_subcortical

    method : function
        function to use for parcellation. Only available option is np.mean

    standardize : bool
        Standardize the time series to zero (temporal) mean and unit
        standard deviation before(!) parcellation.

    Returns
    -------
    ts_parc : TxP np.ndarray
        parcellated time series data
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

    rois, keys, _, _ = _get_atlas(atlas)

    # Schaefer cortical includes the medial wall which we have to insert into the data
    if atlas.startswith("schaefer") and atlas.endswith("_cortical"):
        with importlib_resources.path("comet.data.atlas", "fs_LR_32k_medial_mask.mat") as maskdir:
            medial_mask = loadmat(maskdir)['medial_mask'].squeeze().astype(bool)
        idx = np.where(medial_mask == 0)[0]

        # prepare idices and insert them into the HCP data
        for i, value in enumerate(idx):
            idx[i] = value - i

        cortical_vertices = 59412 # HCP data has 59412 cortical vertices
        ts = ts[:,:cortical_vertices]
        ts = np.insert(ts, idx, np.nan, axis=1)

    # Standardize the time series
    # TODO: Check if it should be done somewhere else
    if standardize:
        ts = _stdize(ts)

    # Parcellation
    n = np.sum(keys!=0)
    ts_parc = np.zeros((len(ts), n), dtype=ts.dtype)

    i = 0
    for k in keys:
        if k!=0:
            ts_parc[:, i] = method(ts[:, rois==k], axis=1)
            i += 1

    return ts_parc

def _get_atlas(atlas_name, debug=False):
    """
    Helper function: Get and prepare a CIFTI-2 atlas for parcellation.

    Parameters
    ----------
    atlas_name : string
        name of the atlas to use for parcellation. Options are:
            - schaefer_{x}_cortical     with x = 100, 200, 300 400, 500, 600, 700, 800, 900, 1000
            - schaefer_{x}_subcortical  with x = 100, 200, 300 400, 500, 600, 700, 800, 900, 1000
            - glasser_mmp_subcortical
    debug : bool, optional
        Flag to provide additional debugging information. Default is False.

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
        "schaefer_cortical": "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_{parcels}Parcels_17Networks_order.dlabel.nii",
        "schaefer_subcortical": "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Cortex-Subcortex/Schaefer2018_{parcels}Parcels_7Networks_order_Tian_Subcortex_S4.dlabel.nii",
        "glasser_mmp_subcortical": "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors_with_Atlas_ROIs2.32k_fs_LR.dlabel.nii"
    }

    # Prepare and check atlas file names
    if "schaefer" in atlas_name:
        try:
            parts = atlas_name.split("_")
            parcels = parts[1]
            atlas_type = f"schaefer_{parts[2]}"
            url = base_urls[atlas_type].format(parcels=parcels)

        except (IndexError, KeyError):
            raise ValueError(f"Invalid atlas name format or unsupported type '{atlas_name}'.")

    elif atlas_name in base_urls:
        url = base_urls[atlas_name]

    else:
        raise ValueError(f"Atlas '{atlas_name}' not recognized. Please choose a valid atlas name.")

    atlas_file_name = f"{atlas_name}.dlabel.nii" if "schaefer" in atlas_name else base_urls[atlas_name]

    # Download (or load) the atlas
    with importlib_resources.path("comet.data.atlas", atlas_file_name) as atlas_path:
        if not atlas_path.exists():
            if "glasser" in atlas_name:
                raise FileNotFoundError(
                    f"Glasser atlas file '{atlas_file_name}' was not found\n"
                    f"  Please download manually from: https://balsa.wustl.edu/file/87B9N\n"
                    f"  and place it in the comet/src/comet/resources/atla folder.")
            else:
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
        labels.append(label.label)
        rgba.append(label.rgba)
        keys.append(key)

    keys = np.asarray(keys)

    return (rois, keys, labels, rgba)

def _stdize(ts):
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

    return (ts - np.mean(ts,axis=0))/np.std(ts,axis=0)
