#!/usr/bin/env python
"""
:Summary:
I/O utilities for EISRAD: loading a NIfTI, reorienting to a canonical
(“RAS”) orientation, extracting its data and voxel dimensions.

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:License: MIT
"""

import nibabel as nib
import numpy as np

def load_and_reorient_pair(path, binarize=None):
    """
    Load a NIfTI file, reorient it to RAS canonical space, extract the
    data array and voxel dimensions.

    Parameters
    ----------
    path : str
        Path to a .nii or .nii.gz file
    binarize : float or None
        If given, threshold data > binarize → 1 (else 0)

    Returns
    -------
    data : np.ndarray
        The image data (possibly binarized)
    zooms : tuple of float
        Voxel dimensions along each axis (in mm)
    """
    img = nib.load(path)
    # reorder array to RAS “canonical” orientation
    img = nib.as_closest_canonical(img)
    arr = img.get_fdata()

    if binarize is not None:
        arr = (arr > binarize).astype(np.uint8)

    # voxel sizes in mm
    zooms = img.header.get_zooms()[:3]
    return arr, zooms
