process GENERATE_JUNCTION_SIGNATURES {
    tag "$meta.id"

    input:
    tuple val(meta), path(h5_file), path(wm), path(nufo), path(signatures), path(mapping)

    output:
    tuple val(meta), path("junction_labels.nii.gz"), emit: junction_labels

    script:
    """
    scil_tractogram_convert_hdf5_to_trk.py ${h5_file} split/ --save_empty

    for i in split/*.trk;
        do scil_tractogram_compute_density_map.py \${i} \${i/.trk/.nii.gz};
    done

    generate_junctions.py ${signatures} ${mapping} split/ \
        ${wm} ${nufo} junction_labels.nii.gz

    rm -rf split/
    """
}
