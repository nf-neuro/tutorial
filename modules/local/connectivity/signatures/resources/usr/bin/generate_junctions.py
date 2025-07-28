#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import tqdm
from glob import glob

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
    p.add_argument('in_assignement_file',
                   help='Both assignment files:\n'
                        '\tTXT file with all possible connectivity signatures')
    p.add_argument('in_dir',
                   help='Input directory containing subject decomposed '
                        'TDI files.')
    p.add_argument('in_wm_mask',
                   help='Input WM mask file.')
    p.add_argument('out_labels',
                   help='Output directory to save the results.')
    
    p.add_argument('--only_signatures', action='store_true',
                   help='If set, only the signatures will be processed '
                        'and saved, skipping the labels generation.')
    p.add_argument('--skip_spatial_filtering', action='store_true',
                   help='If set, spatial filtering of signatures will be skipped.')
    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_assignement_file, args.in_wm_mask])
    if args.only_signatures:
        basename, ext = os.path.splitext(os.path.basename(args.out_labels))
        if ext == '.gz':
            basename, ext = os.path.splitext(os.path.basename(basename))
        
        if os.path.isfile(f"{basename}_signature.txt"):
            raise FileExistsError(
                f"Signature file {basename}_signature.txt already exists. "
                "Please remove it or use a different output name.")
    if not os.path.isdir(args.in_dir):
        raise ValueError(
            f"Input directory not found: {args.in_dir}. Skipping subject.")
    assert_outputs_exist(parser, args, args.out_labels)

    # Define LABEL matrix (ensure dtype is appropriate, e.g., int)
    # Using the second LABEL matrix provided in the MATLAB code
    matrix_pos = np.ones((15, 15), dtype=int) * -1
    comb_list = np.triu_indices(15, k=0)
    for i, coord in enumerate(zip(*comb_list)):
        matrix_pos[coord] = i + 1

    global all_signatures, all_signatures_dict
    all_possibles_signatures = np.loadtxt(
        args.in_assignement_file).astype(np.uint8)
    print(f"Loaded {len(all_possibles_signatures)} signatures from "
          f"{args.in_assignement_file}")

    all_signatures = all_possibles_signatures.astype(float)
    # +1 to ensure labels start from 1
    all_signatures_dict = {hash(tuple(row)): i+1 for i,
                           row in enumerate(all_signatures)}
    print(f"Using {len(all_signatures)} signatures from "
          f"{args.in_assignement_file}")

    wm_img = nib.load(args.in_wm_mask)
    wm_data = wm_img.get_fdata().astype(np.float32)
    wm_mask = wm_data > 0.0

    max_label = np.max(matrix_pos[matrix_pos > 0])
    tdi_data = np.zeros(wm_data.shape + (max_label,), dtype=np.float16)

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
        data = img.get_fdata()

        label_index = matrix_pos[id_1, id_2] - 1
        tdi_data[..., label_index] = data

        count += 1
        continue

    if count != max_label:
        print(f"Warning: Expected {max_label} TDI files, but found {count}.")

    # Voxel-wise normalization of TDI data and thresholding,
    # contribution below THR is set to 0
    print("Normalizing TDI files...")
    THR = 0.10
    mask_sum = np.sum(tdi_data, axis=-1).astype(float)
    for ind in tqdm.tqdm(np.argwhere(mask_sum > 0),
                         total=np.count_nonzero(mask_sum > 0)):
        ind = tuple(ind)
        tmp_tdi_data = tdi_data[ind] / mask_sum[ind]
        tmp_tdi_data[tmp_tdi_data < THR] = 0
        tmp_sum = np.sum(tmp_tdi_data).astype(float)
        if tmp_sum < 1e-6:
            tmp_tdi_data[:] = 0
        else:
            tmp_tdi_data /= tmp_sum
        
        # This should be ceil to ensure integer values (binarize)
        tdi_data[ind] = np.ceil(tmp_tdi_data)

    tdi_data = tdi_data.astype(np.uint8)

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
            return all_signatures_dict[curr_hash]

        # Calculate distances (Cityblock = Manhattan) if not found
        # Too small signature does not have enough information
        curr_sum = np.sum(signature)
        if curr_sum < 3:
            return -1

        # Filter signatures based on the sum of the signature
        D = cdist(signature.reshape(1, -1), all_signatures,
                  metric='cityblock')[0]
        
        best_val = np.min(D)
        # If the best match is too high, return -1
        if best_val > curr_sum // 2:
            return -1

        return np.argmin(D) + 1

    # Flatten except the last dimension
    intersection_mask = np.sum(tdi_data, axis=-1) > 0 & wm_mask
    tdi_data = tdi_data[intersection_mask].astype(float)
    num_voxels = np.count_nonzero(intersection_mask)
    labels_ravel = np.zeros(num_voxels, dtype=np.int32)

    # Voxel-wise processing of signatures
    print("Processing signatures...")
    unique_signatures_count = {}
    for pos in tqdm.tqdm(range(num_voxels), total=num_voxels):
        curr_signature = tdi_data[pos]
        if args.only_signatures:
            # If only signatures are processed, save them directly
            unique_signatures_count[tuple(curr_signature)] = \
                unique_signatures_count.get(tuple(curr_signature), 0) + 1
            continue

        labels_ravel[pos] = _process_voxel(curr_signature)

    if args.only_signatures:
        print("Only signatures processed, skipping label generation.")
        # Save unique signatures to a txt file and count to json
        unique_signatures = list(unique_signatures_count.keys())
        unique_signatures = np.array(unique_signatures, dtype=np.uint8)
        basename, ext = os.path.splitext(os.path.basename(args.out_labels))
        if ext == '.gz':
            basename, ext = os.path.splitext(os.path.basename(basename))
        np.savetxt(f"{basename}_signature.txt",
                   unique_signatures, fmt='%d', delimiter=' ')
        
        tmp_dict = {}
        for key in unique_signatures_count:
            str_key = ' '.join(map(str, key))
            tmp_dict[str_key] = unique_signatures_count[key]
        with open(f"{basename}_count.json", 'w') as f:
            json.dump(tmp_dict, f, indent=4)

        return

    labels = np.zeros_like(wm_data, dtype=np.int32)
    labels[intersection_mask] = labels_ravel
    print(f"Labels unmatched: {np.count_nonzero(labels == -1)}")
    print(f"Labels matched: {np.count_nonzero(labels > 0)}")

    if not args.skip_spatial_filtering:
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

        # Filter out invalid indices (e.g., those that exceed the length of coord_found)
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

    else:
        print("Skipping spatial filtering of signatures.")

    print(f"Saving labels to: {args.out_labels}")
    nib.save(nib.Nifti1Image(labels.astype(
        np.int32), wm_img.affine), args.out_labels)


if __name__ == '__main__':
    main()
