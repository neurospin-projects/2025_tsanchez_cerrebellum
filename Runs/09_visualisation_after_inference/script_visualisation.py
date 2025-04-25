from visualisation_anatomist import VisualiserAnatomist
from soma import aims
import matplotlib.pyplot as plt 
import anatomist.api as ana
from pathlib import Path
import numpy as np

SLICE_CLIP = aims.Quaternion([0.6427876096865394,
  -0.3420201433256688,
  0.6427876096865394,
  0.3420201433256688])

VIEW_SAGITTAL = [0.5,-0.5,-0.5,0.5]
OBLIC_VIEW = [0.6,-0.2,-0.25,0.75] 

# Palette array
PALETTE = np.zeros((512,1,1,1,4))
PALETTE[100:400,:,:,:,3] = 0 # Settings middle alpha to 0
PALETTE[:100,:,:,:,2] = 255 # Settings begin to blue max
PALETTE[:100,:,:,:,1] = 255 # Settings begin to green max
PALETTE[400:,:,:,:,0] = 255 # Settings end to red max

PATH = Path("/neurospin/dico/tsanchez/tmp/input_inference.npy")
PATH_SAVING = Path("/neurospin/dico/tsanchez/tmp")

DICT_VIEWS = {
    "normal" : (
        SLICE_CLIP,
        VIEW_SAGITTAL,
        1,
        PALETTE),

    "oblic" : (
        SLICE_CLIP,
        OBLIC_VIEW,
        1,
        PALETTE)
}

if __name__ == "__main__" :
    a = ana.Anatomist()
    win = a.createWindow("3D")
    

    visu = VisualiserAnatomist(
        path_or_obj = PATH,
        saving_path= PATH_SAVING,
        dict_views= DICT_VIEWS,
        anatomist=a,
        window=win
    )
    image = visu.show(buffer = False, view_settings = DICT_VIEWS["normal"])
    print(image)
