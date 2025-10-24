# Tile segmentation and fluorescence quantification of big 2D images.

Segmentation relies on Stardist (https://github.com/stardist/stardist). Examples of the three main ussages of the software can be found on the examples folder:

* `compute_tile_overlap.py` gives you an estimation of the tile overlap that should be used.
* `remove_debris.ipynb` gives you an estimation of the size on the debris in your data so that it can be thresholded out.
* `fluo_quantification_and_thresholding.ipynb` is an example of quantification of the fluorescence inside the segmented nuclei using the overlap and thresholding obtained with the other two scripts. 

## Installation
`$ python -m pip install 'tilesegment @ git+https://github.com/dsb-lab/tilesegment'`

## Usage

For a ready to use example check `examples/fluo_quantification_and_thresholding.ipynb`
