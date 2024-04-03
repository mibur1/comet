import sys
import numpy as np
import nibabel as nib
nib.imageglobals.logger.setLevel(40)
from scipy.io import loadmat
import importlib.resources as pkg_resources

def parcellate(dtseries, atlas="glasser", method=np.mean, standardize=True):
    """
    Parcellation, calculates the mean over all grayordinates within a parcel. Z-standardization is enabled by default
    """
    ts = dtseries.get_fdata()
    rois, keys, labels, rgba = prepare_atlas(atlas)

    # schaefer_kong includes the medial wall which we have to insert into the data
    if atlas == "schaefer_kong":
        with pkg_resources.path("comet.resources.atlas", "fs_LR_32k_medial_mask.mat") as maskdir:
            medial_mask = loadmat(maskdir)['medial_mask'].squeeze().astype(bool)
        idx = np.where(medial_mask == 0)[0]

        # prepare idices and insert them into the HCP data
        for i in range(len(idx)):
            idx[i] = idx[i] - i

        cortical_vertices = 59412 # HCP data has 59412 cortical vertices
        ts = ts[:,:cortical_vertices]
        ts = np.insert(ts, idx, np.nan, axis=1)

    # Standardize the time series
    # TODO: Check if it should be done somewhere else
    if standardize:
        ts = stdize(ts)
    
    # Parcellation
    n = np.sum(keys!=0)
    ts_parc = np.zeros((len(ts), n), dtype=ts.dtype)

    i = 0
    for k in keys:
        if k!=0:
            ts_parc[:, i] = method(ts[:, rois==k], axis=1)
            i += 1

    return ts_parc

def prepare_atlas(atlas_name, debug=False):
    """
    Prepare a cifti 2 atlas to be used in parcellation
    """
    if atlas_name == "glasser":
        with pkg_resources.path("comet.resources.atlas", "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors_with_Atlas_ROIs2.32k_fs_LR.dlabel.nii") as path:
            atlas = nib.load(path)
    elif atlas_name == "schaefer_kong":
        with pkg_resources.path("comet.resources.atlas", "Schaefer2018_200Parcels_Kong2022_17Networks_order.dlabel.nii") as path:
            atlas = nib.load(path)
    elif atlas_name == "schaefer_tian":
        with pkg_resources.path("comet.resources.atlas", "Schaefer2018_200Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii") as path:
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
        for idx, (name, slice, bm) in enumerate(brainmodelAxis.iter_structures()):
            print(idx, str(name), slice)

    index_map = atlas.header.get_index_map(0)
    named_map=list(index_map.named_maps)[0]

    keys = []
    labels = []
    rgba = []

    # Iterate over label_table and get relevat values
    for i in range(len(named_map.label_table)):
        roi = named_map.label_table[i]
        labels.append(roi.label)
        rgba.append(roi.rgba)
        keys.append(roi.key)
    keys = np.asarray(keys)
    
    return (rois, keys, labels, rgba)

def get_fdata(dtseries):
    """
    Get the cifti 2 data
    """
    data = nib.load(dtseries)
    return data.get_fdata()

def get_header(dtseries):
    """
    Get the cifti 2 headers
    """
    data = nib.load(dtseries)
    return data.header

def get_labels(atlas):
    """
    Get atlas labels and default rgba values
    """
    _, _, labels, rgba = prepare_atlas(atlas)
    return labels, rgba

def stdize(ts):
    """
    Standardize to zero (temporal) mean and unit standard deviation.
    """
    return (ts - np.mean(ts,axis=0))/np.std(ts,axis=0)
