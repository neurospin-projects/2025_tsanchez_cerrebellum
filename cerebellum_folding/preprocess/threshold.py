from pathlib import Path
from typing import Union

import numpy as np
from soma import aims

from cerebellum_folding.data.path import SubjectPath, MaskPath
from deep_folding.brainvisa.utils.resample import resample
import deep_folding.brainvisa.utils.dilate_mask as dl 

WM_THRESH = -0.3967
SULCI_THRESH = 0.464

def threshold_mean_curv(subject_path : SubjectPath, to_save : bool = False, white_matter_thresh : float = WM_THRESH, sulci_thresh : int = SULCI_THRESH) -> np.ndarray :

    #Read mean curvature file 
    obj = aims.read(str(subject_path.mc))
    #To numpy 
    vol = obj.np

    # Mask for WM and Sulci
    white_matter_mask = (vol <= white_matter_thresh) 
    sulci_mask = (vol >= sulci_thresh)
    rest = (vol > white_matter_thresh) & (vol < sulci_thresh) 

    #Apply Mask : 
    vol[white_matter_mask] = -1
    vol[sulci_mask] = 1
    vol[rest] = 0

    # Convert from float type to S16 more convinient
    conv = aims.Converter(intype=obj, outtype=aims.Volume("S16")) 
    vol = conv(obj)

    if to_save : 
        print(f"Saving to {subject_path.thresh}")
        aims.write(obj, filename=str(subject_path.thresh))
    
    return vol

def get_ICBM2009c_transform(graph_path) : 
    graph = aims.read(str(graph_path))
    transf = aims.GraphManip.getICBM2009cTemplateTransform(graph)
    return transf


RESAMPLE_VALUES = [0, -1, 1]
OUTPUT_VOXEL_SIZE = (1,1,1)

def transform_ICBM2009c(paths : SubjectPath, resample_values = RESAMPLE_VALUES, output_voxel = OUTPUT_VOXEL_SIZE ,  save : bool = False, verbose : bool = False) : 

    native_obj = aims.read(str(paths.thresh))
    # c = aims.Converter(intype=native_obj, outtype=aims.Volume('S16'))
    # native_obj = c(native_obj)

    # Apply transform
    transf = get_ICBM2009c_transform(paths.graph)
    resampled_to_ICBM2009c = resample(
        input_image=native_obj, 
        transformation=transf,
        output_vs=output_voxel,
        background=0,
        values= resample_values,
        verbose=verbose
    )

    if save : 
        aims.write(resampled_to_ICBM2009c, filename=str(paths.ICBM2009c))

    return resampled_to_ICBM2009c


# Function to retrieve the cerebellum only in Clara Fisher's masks (cf. Ataxia)
def only_mask_cerebellum(vol) -> None :
    vol_np = vol.np

    cerebellum = np.isin(vol_np, [1,2,3])
    rest = np.isin(vol_np, [0,5,6,7,8])

    vol_np[cerebellum] = 1
    vol_np[rest] = 0


DILATATION = 5 #mm

def dilatate_mask(mask, saving_path, dilatation : int = DILATATION, save : bool = False,):
    dilated_mask = dl.dilate(mask, radius=dilatation)
    if save :
        aims.write(dilated_mask, filename = saving_path)

def get_binary_val(thresh_path : Union[Path, str], to_isolate : str , saving_path : Union[Path, str], to_save : bool = False):

    assert to_isolate in ["sulci", "white_matter"], "Should be in ['sulci', 'white_matter']"

    # ! To validate
    # Value corresponding the the encoding in the MC Thresholded
    val = -1 if to_isolate == "sulci" else 1

    thresh_vol = aims.read(str(thresh_path))
    thresh_np = thresh_vol.np

    # Retrieving masks for the analysis
    mask = (thresh_np == val)

    thresh_np[mask] = 1
    thresh_np[~mask] = 0

    if to_save : 
        aims.write(thresh_vol, saving_path)


def full_pipe_subject(subject_path : SubjectPath) : 
    pass
