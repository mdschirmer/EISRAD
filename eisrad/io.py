#!/usr/bin/env python
"""
:Summary:
I/O utilities for EISRAD: loading a NIfTI, reorienting to a canonical
(“RAS”) orientation, extracting its data and voxel dimensions.

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:Version: 1.0
:Date: 2025-06-24
:License: MIT
"""

import nibabel as nib
import numpy as np

def load_and_reorient_pair(path, binarize=False):
    """
    Load a NIfTI file, reorient it to RAS canonical space, extract the
    data array and voxel dimensions.

    Parameters
    ----------
    path : str
        Path to a .nii or .nii.gz file
    binarize : bool
        If True, threshold data > 0 → 1

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

    if binarize:
        arr = (arr > 0).astype(np.uint8)

    # voxel sizes in mm
    zooms = img.header.get_zooms()[:3]
    return arr, zooms
