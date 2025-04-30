from visualisation_anatomist import VisualiserAnatomist
import torch 
from soma import aims
import anatomist.api as ana
from pathlib import Path
import numpy as np

SLICE_CLIP = aims.Quaternion([0.6427876096865394,
  -0.3420201433256688,
  0.6427876096865394,
  0.3420201433256688])

VIEW_SAGITTAL = [0.5,-0.5,-0.5,0.5]
OBLIC_VIEW = [0.6,-0.2,-0.25,0.75] 

PATH_INPUT = Path("/neurospin/dico/tsanchez/tmp/input_inference.npy")
PATH_OUTPUT = Path("/neurospin/dico/tsanchez/tmp/output_inference.npy")

DICT_VIEWS = {
    "normal" : (
        SLICE_CLIP,
        VIEW_SAGITTAL,
        1),

    "oblic" : (
        SLICE_CLIP,
        OBLIC_VIEW,
        1)
}

if __name__ == "__main__" :
    a = ana.Anatomist()
    win = a.createWindow("3D")
    win2 = a.createWindow("3D")
    

    visu_input = VisualiserAnatomist(
        path_or_obj = PATH_INPUT,
        dict_views= DICT_VIEWS,
        anatomist=a,
        window=win
    )

    output = np.load(PATH_OUTPUT) 
    visu_output = VisualiserAnatomist(
        path_or_obj = output,
        dict_views= DICT_VIEWS,
        anatomist=a,
        window=win2
    )

    # image = visu.show(buffer = False, view_settings = DICT_VIEWS["normal"])
    image_output = visu_output.tensor_image("normal", buffer = True)
    print(image_output)
    # image_input = visu_input.tensor_image("normal", buffer = True)

    # diff = torch.unique(image_input == image_output)
    # print(f"Output : {image_output}")
    # print(f"InpuInput : {image_input}")
    # print(f"{diff}")