
apptainer run --nv -B /tmp/camelyon17_v1.0.sqfs:/data/camelyon17_v1.0:image-src=/ \
    ../containers/dispatch-new.sif \
    python \
        train.py \
            --config-name debug