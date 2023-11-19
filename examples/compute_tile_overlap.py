
import tilesegment.tools.tools as tools
import tilesegment.tools.tiling as tiling
import matplotlib.pyplot as plt

from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack
from embdevtools.celltrack.core.tools.tools import get_outlines_masks_labels
import numpy as np


### LOAD STARDIST MODEL ###
from csbdeep.utils import normalize
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/belas/2D/Sina/activin/crops/movies/'
path_save='/home/pablo/Desktop/PhD/projects/AVEDifferentiation/results/2D/activin/'

concentration = "50ng"

### GET FULL FILE NAME AND FILE CODE ###
file, embcode, files = get_file_embcode(path_data, "_"+concentration, allow_file_fragment=True, returnfiles=True)

### LOAD HYPERSTACKS ###
IMGS, xyres, zres = read_img_with_resolution(path_data+file, stack=False, channel=0)
try: 
    xyresdif = np.abs(np.abs(xyres[0]) - np.abs(xyres[1]))
    print("xyres dif =", xyresdif)
    xyres = np.mean(xyres)
except: 
    pass
IMG = IMGS[0,0]

img = IMG[:2024, :2024]
            
labs, _ = model.predict_instances(normalize(img))
outlines, masks, _labs = get_outlines_masks_labels(labs) 

_diams = []
for outline in outlines:
    diam = tools.rotating_calipers(outline)
    _diams.append(diam)

diams = np.array(_diams)*xyres

from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from scipy.signal import argrelextrema

x = np.arange(0, step=0.01,stop=np.max(diams))
local_minima=[]
bw = 2.0
modelo_kde = KernelDensity(kernel='linear', bandwidth=bw)
modelo_kde.fit(X=diams.reshape(-1, 1))
densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1,1))))

idmax = np.argmax(densidad_pred)
x_th = np.ones(len(x))*x[idmax]
y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
plt.plot(x_th, y_th, c='k', ls='--')
plt.hist(diams, bins=50, density=True, label="hist")
plt.plot(x, densidad_pred, lw=5, label="kde")
plt.legend()
plt.xlabel("diam (µm)")
plt.yticks([])
plt.title("mean cell diam = {:0.2f}µm".format(x[idmax]))
plt.show()

print("Mean diameter = {:0.2f} µm".format(x[idmax]))
print("Tile overlap = 3 * mean diameter = {:d} µm".format(np.int32(np.rint(x[idmax]*3))))
