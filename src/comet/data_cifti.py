import sys
import numpy as np
import nibabel as nib
nib.imageglobals.logger.setLevel(40)
from scipy.io import loadmat
import importlib_resources

"""
SECTION: Functiond for parcellating cifti data
 - Currently limited to a few included parcellatons, will be expanded in the future
"""
def parcellate(dtseries, atlas="glasser", method=np.mean, standardize=True):
    """
    Parcellate cifti data (.dtseries.nii) using a given atlas.

    The parcellated time series is calculated as the mean over all grayordinates within a parcel.

    Parameters
    ----------
    dtseries : nibabel.cifti2.cifti2.Cifti2Image
        nibabel cifti image object

    atlas : string
        name of the atlas to use for parcellation. Options are:
         - glasser (Glasser MMP cortical parcellation + subcortical)
         - schaefer_kong (Schaefer 200 cortical parcellation)
         - schaefer_tian (Schaefer 200 cortical parcellation + subcortical)

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

    ts = dtseries.get_fdata()
    rois, keys, _, _ = _prepare_atlas(atlas)

    # schaefer_kong includes the medial wall which we have to insert into the data
    if atlas == "schaefer_kong":
        with importlib_resources.path("comet.resources.atlas", "fs_LR_32k_medial_mask.mat") as maskdir:
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

def _prepare_atlas(atlas_name, debug=False):
    """
    Helper functio: Prepare a CIFTI-2 atlas for parcellation.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to use for parcellation. Options are: 'glasser', 'schaefer_kong', 'schaefer_tian'.
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

    if atlas_name == "glasser":
        with importlib_resources.path("comet.resources.atlas",
                                      "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors_with_Atlas_ROIs2.32k_fs_LR.dlabel.nii") as path:
            atlas = nib.load(path)
    elif atlas_name == "schaefer_kong":
        with importlib_resources.path("comet.resources.atlas",
                                      "Schaefer2018_200Parcels_Kong2022_17Networks_order.dlabel.nii") as path:
            atlas = nib.load(path)
    elif atlas_name == "schaefer_tian":
        with importlib_resources.path("comet.resources.atlas",
                                      "Schaefer2018_200Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii") as path:
            atlas = nib.load(path)
    else:
        sys.exit("Atlas must be any of glasser, schaefer_kong, or schaefer_tian")

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

    print(named_map.label_table.items())

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

def _get_fdata(dtseries):
    """
    Helper function: Get the cifti data

    Parameters
    ----------
    dtseries : nibabel.cifti2.cifti2.Cifti2Image
        nibabel cifti image objet

    Returns
    -------
    fdata : np.ndarray
        Time series data of the cifti image
    """

    data = nib.load(dtseries)
    return data.get_fdata()

def _get_header(dtseries):
    """
    Helper function: Get the cifti header

    Parameters
    ----------
    dtseries : nibabel.cifti2.cifti2.Cifti2Image
        nibabel cifti image objet

    Returns
    -------
    header : nibabel.cifti2.cifti2.Cifti2Header
        nibabel cifti header object
    """

    data = nib.load(dtseries)
    return data.header

def _get_labels(atlas):
    """
    Helper function: Get atlas labels and default rgba values

    Parameters
    ----------
    atlas : nibabel.cifti2.cifti2.Cifti2Image
        nibabel cifti image objet

    Returns
    -------
    labels : tuple
        Tuple containing the labes and rgba values
    """

    _, _, labels, rgba = _prepare_atlas(atlas)
    return labels, rgba
