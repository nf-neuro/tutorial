process GENERATE_JUNCTION_SIGNATURES {
    tag "$meta.id"

    input:
    tuple val(meta), path(trk), path(labels), path(wm), path(nufo), path(signatures), path(mapping)

    output:
    tuple val(meta), path("*__junction_labels.nii.gz"), emit: junction_labels
    tuple val(meta), path("*.txt"), emit: junction_labels
    path ("split/")               , emit: split

    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    decompose_simple.py ${trk} ${labels} split/

    touch ${prefix}__junction_labels.nii.gz
    # generate_junctions.py ${signatures} ${mapping} split/ \
    #     ${wm} ${nufo} ${prefix}__junction_labels.nii.gz --only_signatures
    """
}
