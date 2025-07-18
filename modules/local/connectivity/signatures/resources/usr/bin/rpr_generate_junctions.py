#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import tqdm

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


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_assignement_files', nargs=2,
                   metavar='ASSIGN_FILES',
                   help='Both assignment files:\n'
                        '\t1. MAT file with all possible connectivity signatures\n'
                        '\t2. MAT file with mapping of original to new labels.')
    p.add_argument('in_dir',
                   help='Input directory containing subject decomposed '
                        'TDI files.')
    p.add_argument('in_wm_mask',
                   help='Input WM mask file.')
    p.add_argument('in_nufo',
                   help='Input NUFO file.')
    p.add_argument('out_labels',
                   help='Output directory to save the results.')
    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_assignement_files + [args.in_wm_mask,
                                                             args.in_nufo])
    if not os.path.isdir(args.in_dir):
        raise ValueError(
            f"Input directory not found: {args.in_dir}. Skipping subject.")
    assert_outputs_exist(parser, args, args.out_labels)

    # mapping_labels = scipy.io.loadmat(args.in_assignement_files[1])
    with open(args.in_assignement_files[1], 'r') as f:
        mapping_labels = json.load(f)
    orig_labels = np.squeeze(mapping_labels['orig_label'])
    new_labels = np.squeeze(mapping_labels['new_label'])
    mapping_labels = dict(zip(orig_labels, new_labels))

    # Define LABEL matrix (ensure dtype is appropriate, e.g., int)
    # Using the second LABEL matrix provided in the MATLAB code
    labels = np.ones((15, 15), dtype=int) * -1
    comb_list = np.triu_indices(15, k=0)
    for i, coord in enumerate(zip(*comb_list)):
        labels[coord] = i + 1

    # mat_data = scipy.io.loadmat(args.in_assignement_files[0])
    # B = mat_data['B']
    global all_signatures, all_signatures_dict
    all_possibles_signatures = np.loadtxt(
        args.in_assignement_files[0]).astype(np.uint8)
    print(f"Loaded {len(all_possibles_signatures)} signatures from "
          f"{args.in_assignement_files[0]}")

    all_signatures = all_possibles_signatures[orig_labels].astype(np.uint8)
    all_signatures_dict = {hash(tuple(row)): i for i,
                           row in enumerate(all_signatures)}
    print(f"Using {len(all_signatures)} signatures from "
          f"{args.in_assignement_files[0]}")

    nufo_img = nib.load(args.in_nufo)
    nufo_data = nufo_img.get_fdata().astype(np.uint8)
    nufo_data = np.clip(nufo_data, 0, 3)

    wm_img = nib.load(args.in_wm_mask)
    wm_data = wm_img.get_fdata().astype(np.float32)
    wm_mask = wm_data > 0.0

    max_label = np.max(labels[labels > 0])
    tdi_data = np.zeros(wm_data.shape + (max_label,), dtype=float)

    assert_headers_compatible(wm_img, [nufo_img])
    print(f"Initialized labels array with shape: {tdi_data.shape}")

    print("Grabbing TDI files...")
    count = 0
    comb_list = np.triu_indices(15, k=0)
    for id_1, id_2 in tqdm.tqdm(zip(*comb_list), total=len(comb_list[0])):
        tdi_path = os.path.join(args.in_dir, f'{id_1+1}_{id_2+1}.nii.gz')

        if not os.path.isfile(tdi_path):
            print(f"Warning: TDI file not found: {tdi_path}")
            continue

        img = nib.load(tdi_path)
        assert_headers_compatible(wm_img, [img])
        data = img.get_fdata().astype(np.uint32)

        label_index = labels[id_1, id_2] - 1
        tdi_data[..., label_index] = data
        print(np.max(tdi_data))
        count += 1

    if count != max_label:
        print(f"Warning: Expected {max_label} TDI files, but found {count}.")

    THR = 0.10
    mask_sum = np.sum(tdi_data, axis=-1)
    for ind in np.argwhere(mask_sum > 0):
        ind = tuple(ind)
        tmp_tdi_data = tdi_data[ind] / mask_sum[ind]
        # print(ind, tdi_data[ind], mask_sum[ind], tmp_tdi_data)
        tmp_tdi_data[tmp_tdi_data < THR] = 0
        tmp_tdi_data /= np.sum(tmp_tdi_data)
        if mask_sum[ind] > 2:
            print(ind, tdi_data[ind], mask_sum[ind], tmp_tdi_data)
        tdi_data[ind] = tmp_tdi_data

    tdi_data[tdi_data > 0] = 1

    def _process_voxel(signature):
        """
        Process a single voxel's signature against the NUFO signatures.
        """
        global all_signatures, all_signatures_dict
        if np.sum(signature) == 0:
            return -1

        curr_hash = hash(tuple(signature))
        # Check if the current signature is in the dictionary
        if curr_hash in all_signatures_dict:
            best_match = all_signatures_dict[curr_hash]
            return best_match + 1
        else:
            for i in range(1, 4):  # Max NUFO is 3
                # Check if the current signature is a subset of any signature in all_signatures
                if i == signature[0]:
                    continue
                elif curr_hash in all_signatures_dict:
                    best_match = all_signatures_dict[curr_hash]
                    return best_match + 1

        # Calculate distances (Cityblock = Manhattan) if not found
        # Too small signature does not have enough information
        if np.sum(signature[1:]) < 3:
            return -1

        D = cdist(signature[1:].reshape(1, -1), all_signatures[:, 1:],
                  metric='cityblock')[0]

        best_match = np.argmin(D)
        # min_dist = D[best_match]
        return best_match + 1

    # Flatten except the last dimension
    intersection_mask = np.sum(tdi_data, axis=-1) > 0 & wm_mask
    tdi_data = tdi_data[intersection_mask]
    nufo_data = nufo_data[intersection_mask]
    num_voxels = np.count_nonzero(intersection_mask)
    labels_ravel = np.zeros_like(nufo_data, dtype=np.int16)

    # Voxel-wise processing of signatures
    for pos in tqdm.tqdm(range(num_voxels), total=num_voxels):
        curr_signature = tdi_data[pos]
        curr_signature = np.insert(curr_signature, 0, nufo_data[pos])
        best_match = _process_voxel(curr_signature)

        # labels_ravel[pos] = mapping_labels.get(best_match, -2)
        labels_ravel[pos] = best_match

    labels = np.zeros_like(wm_data, dtype=np.int16)
    labels[intersection_mask] = labels_ravel
    print(f"Labels unmatched: {np.count_nonzero(labels == -1)}")
    print(f"Labels matched: {np.count_nonzero(labels > 0)}")
    labels[labels == -1] = 0

    # Remove unconnected island for each label
    min_voxel_count = 6
    voxel_to_remove = np.ones_like(labels, dtype=np.uint8)

    for label_id in tqdm.tqdm(np.unique(labels)[1:]):
        curr_data = np.zeros_like(labels, dtype=np.uint8)
        curr_data[labels == label_id] = 1
        components, nb_structures = ndi.label(curr_data)
        # For each label, remove small components
        for label in range(1, nb_structures + 1):
            if np.count_nonzero(components == label) < min_voxel_count:
                voxel_to_remove[components == label] = 0
    labels *= voxel_to_remove

    coord_unfound = np.argwhere((wm_mask > 0) & (labels == 0))
    coord_found = np.argwhere(labels > 0)

    tree = KDTree(coord_found)
    _, idx = tree.query(coord_unfound, k=1, distance_upper_bound=5)

    # # Filter out invalid indices (e.g., those that exceed the length of coord_found)
    valid_idx_mask = idx < len(coord_found)
    valid_idx = idx[valid_idx_mask]

    # Extract the labels at the neighbor coordinates
    labels_found = labels[coord_found[valid_idx, 0],
                          coord_found[valid_idx, 1],
                          coord_found[valid_idx, 2]]
    # Assign the labels to the unfound coordinates
    labels[coord_unfound[valid_idx_mask, 0],
           coord_unfound[valid_idx_mask, 1],
           coord_unfound[valid_idx_mask, 2]] = labels_found

    print(f"Saving labels to: {args.out_labels}")
    nib.save(nib.Nifti1Image(labels.astype(
        np.uint16), wm_img.affine), args.out_labels)


if __name__ == '__main__':
    main()
