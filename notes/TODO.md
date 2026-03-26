# TODO

## Dead code: `_tile_transform.npy` saved but never used
- In `annotating_data/annotation_aided.py`, `img_toseg_tile_edges` (the locally-normalized edge-enhanced image) is saved as `*_tile_transform.npy`, but this file is never loaded or used anywhere in the repo.
- The old `_previous_version/mypytorch_fullseg/dataset_classes_fullseg.py` constructs `thefilelist_extra` with the `_tile_transform.npy` suffix but never actually loads it in `__getitem__`.
- **Decision needed**: either start using this edge transform as an extra input channel for training, or remove the saving step to avoid generating unused files.
