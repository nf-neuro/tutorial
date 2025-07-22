#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import tqdm

import itertools
from dipy.io.streamline import load_tractogram
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             assert_headers_compatible,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Input tractogram file to decompose')
    p.add_argument('in_labels',
                   help='Input labels file containing the connectivity signatures.')
    p.add_argument('out_dir',
                   help='Output directory to save the decomposed TDI files.')
    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_tractogram, args.in_labels])

    sft = load_tractogram(args.in_tractogram, 'same')
    labels = nib.load(args.in_labels).get_fdata().astype(np.uint16)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    unique_labels = np.unique(labels)[1:]
    comb_list = list(itertools.combinations(unique_labels, 2))
    comb_list.extend([(i, i) for i in unique_labels])

    for i, j in tqdm.tqdm(comb_list, desc='Decomposing signatures'):
        mask = np.logical_or(labels == i, labels == j)
        mode = 'both_ends'
        is_exclude = False
        distance = 0.0

        _, filtered_sft = filter_grid_roi(
                    sft, mask, mode, is_exclude, distance, return_sft=True)
        filtered_sft.to_vox()
        filtered_sft.to_corner()
        density = compute_tract_counts_map(filtered_sft.streamlines,
                                           sft.dimensions)

        nib.save(nib.Nifti1Image(density.astype(np.int32), sft.affine),
                 os.path.join(args.out_dir, f'{i}_{j}.nii.gz'))

if __name__ == '__main__':
    main()
