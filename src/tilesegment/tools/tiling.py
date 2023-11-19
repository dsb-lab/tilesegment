from .dataclasses import padding_info, tiling_info
from .tools import rotating_calipers
import numpy as np


def generate_tiling_info(IMG, Ts, O):
    """Function to generate tiling info
    Parameters
    ----------
    IMG : ndarray
        2D ndarray with the unsquared, unpadded image
    Ts: int
        Tile size 
    O: int
        Tile overlap size


    Returns
    -------
    tile_info: tiling_info
        Contains the parameters of the tiling. run help(tiling_info) for more info
    """

    # Define overlap by running a segmentation test and checking mean cell diameter after removing small objects
    # Total image size
    sh = IMG.shape
    D = compute_closest_larger_multiple(max(sh), Ts)

    Tso = Ts + 2*O
    DO = D + 2*O

    # Number of tiles
    n = int(D/Ts)
    N = int(n**2)

    tile_info = tiling_info(Ts, O, D, Tso, DO, n, N)
    return tile_info

def compute_closest_larger_multiple(number, multiple):
    """Computed closes larger multiple
    Parameters
    ----------
    number : int
        Number to be changed to make if multiple to `multiple`
    multiple: int
        Base multiple 

    Returns
    -------
    closest_multiple:  int
        Closest larger multiple of `multiple` to `number`. 
        That is `number` is scaled up until it's a multiple of `multiple`.
    """
    
    if not (number < 0 or multiple <= 0):
        closest_multiple = (number // multiple) * multiple  # Find the largest multiple less than or equal to the number
        if closest_multiple < number:
            closest_multiple += multiple
        return closest_multiple
    else: 
        raise Exception("number and multiple cannot be negative. Multiple cannot be 0")

def pad_image_and_square_array(IMG, tile_info):
    """Squares and pads input image
    
    Parameters
    ----------
    IMG : ndarray
        2D ndarray with the unsquared, unpadded image
    tile_info: tiling_info
        See help(tiling_info) for more information

    Returns
    -------
    IMG_padded:  ndarray
        2d ndarray with the image squared and padded
    pad_info: padding_info
        Contains the information required by the tiling function to reconstruct
        the results of the segmentation from the padded images
    """
    
    # indexed to be used for image reconstruction
    sh = IMG.shape
    ishdiff = tile_info.D-sh[0]
    top_ishdiff = np.int32(np.ceil(ishdiff/2))
    bot_ishdiff = np.int32(np.floor(ishdiff/2))
    jshdiff = tile_info.D-sh[1]
    lef_jshdiff = np.int32(np.ceil(jshdiff/2))
    rig_jshdiff = np.int32(np.floor(jshdiff/2))

    pad_info = padding_info(ishdiff, top_ishdiff, bot_ishdiff, jshdiff, lef_jshdiff, rig_jshdiff)
    
    IMG_padded = np.zeros((tile_info.DO,tile_info.DO))
    IMG_padded[tile_info.O+top_ishdiff:-tile_info.O-bot_ishdiff, tile_info.O+lef_jshdiff:-tile_info.O-rig_jshdiff] = IMG
    return IMG_padded, pad_info

from embdevtools.celltrack.core.tools.tools import get_outlines_masks_labels
from csbdeep.utils import normalize

def tile_segment(IMG_padded, model, tile_info, pad_info, xyres, diam_th, verbose=False):
    labels = np.zeros_like(IMG_padded)
    maxlab=0

    centers = []
    masks = []
    diameters = []
    tiles_left = tile_info.N

    for i in range(tile_info.n):
        idsi= slice((i*tile_info.Ts),((i+1)*tile_info.Ts+2*tile_info.O))
        for j in range(tile_info.n):
            idsj= slice((j*tile_info.Ts),((j+1)*tile_info.Ts+2*tile_info.O))
            img = IMG_padded[idsi, idsj]
            
            labs, _ = model.predict_instances(normalize(img))
            outlines, _masks, _labs = get_outlines_masks_labels(labs) 
            for l, lab in enumerate(_labs):
                outline = outlines[l]
                mask = _masks[l]
                diam = rotating_calipers(outline)
                center = np.mean(mask, axis=0)
                # check if label center falls out of the limits
                mins = tile_info.O, tile_info.O
                maxs = tile_info.Ts+tile_info.O, tile_info.Ts+tile_info.O
                
                if (((center < mins).any() or (center > maxs).any()) or (diam*xyres  < diam_th)):
                    labs[mask[:,0], mask[:,1]] = 0
                else:
                    offset = np.array([i*tile_info.Ts-tile_info.O-pad_info.top_ishdiff, j*tile_info.Ts-tile_info.O-pad_info.lef_jshdiff])
                    centers.append(center+offset)
                    masks.append(mask+offset)
                    diameters.append(diam)

            labs += maxlab
            background = np.transpose(np.where(labs == maxlab))
            labs[background[:,0], background[:,1]] = 0
            maxlab = labs.max()
            labels[idsi, idsj] += labs
            tiles_left -=1
            if verbose: print("tiles left =", tiles_left)
    return centers, masks, diameters