from dataclasses import dataclass

from numba import b1, typed, typeof
from numba.experimental import jitclass
from numba.types import Array, ListType, float32, int64, uint16


@dataclass
class padding_info:
    """Padding info dataclass
    
    Dataclass to store padding information used for reconstruction
    posterior to the segmentation.

    Attributes:
        ishdiff (int)
        top_ishdiff (int)
        bot_ishdiff (int)
        jshdiff (int)
        lef_jshdiff (int)
        rig_jshdiff (int)
    """
    ishdiff: int
    top_ishdiff: int
    bot_ishdiff: int
    jshdiff: int
    lef_jshdiff: int
    rig_jshdiff: int

@dataclass
class tiling_info:
    """Tiling info dataclass
    
    Dataclass to store tiling information.

    Attributes:
        Ts (int): Tilse size
        O (int): Overlap between tiles
        D (int): Image size after squaring and padding
        Tso (int): Tile size with overlap
        DO (int): Image size after squaring and padding with overlap
        n (int): Number of tiles in a row or column
        N (int): Total number of tiles
    """
    Ts: int
    O: int
    D: int
    Tso: int
    DO: int
    n: int
    N: int
    
