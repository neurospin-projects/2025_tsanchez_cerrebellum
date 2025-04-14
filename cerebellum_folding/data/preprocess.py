"""Library for the preprocessing pipeline"""


from cerebellum_folding.data.path import SubjectPath, MaskPath, check_file

import deep_folding.brainvisa.utils.dilate_mask as dl 
from deep_folding.brainvisa.utils.resample import resample
import subprocess as sp

from typing import List, Dict, Union
from pathlib import Path
from soma import aims, aimsalgo
import numpy as np
import gc


class PipelineSubject :
    def __init__(self,
                 subject_path : SubjectPath,
                 masks_path : Dict[str, MaskPath], 
                 white_matter_threshold : float, 
                 sulci_threshold : float, 
                 resample_values_icbm : List[int], 
                 resample_values_bin : List[int],
                 output_voxel : List[int], 
                 verbose : bool = False,
                ):
        # Retrieving paths of the subject
        self.path = subject_path
        self.id = self.path.id

        # Creating saving folders
        self.path.create_saving_paths()

        self.masks_path = masks_path
        
        # Parameters of the pipeline
        self.wm_thresh = white_matter_threshold
        self.sulci_thresh = sulci_threshold
        self.resample_val_icbm = resample_values_icbm
        self.resample_val_bin = resample_values_bin
        self.resample_output_voxel = output_voxel

        self.verbose = verbose
    
    def print(self, string):
        if self.verbose : 
            print(string)


    def threshold_mean_curvature(self,
                                overwrite : bool = False,
                                ):
        
        if not check_file(self.path.mc):
            raise Exception(f"Mean curvature file not available : {self.path.mc}")
        
        if check_file(self.path.native["threshold"]):
            if not overwrite : 
                raise Exception("File already exists")
            else : 
               self.print(f"Overwriting : {self.path.native['threshold']}") 

        white_matter_thresh = self.wm_thresh
        sulci_thresh = self.sulci_thresh

        obj = aims.read(str(self.path.mc))
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

        self.print(f"Saving to {self.path.native['threshold']}")
        aims.write(obj, filename=str(self.path.native['threshold']))


    def get_binary_val(self,
                      overwrite : bool = False,
                    ):

        if not check_file(self.path.native["threshold"]):
            raise Exception("Threshold path doesn't exist")


        for to_isolate in ["sulci", "white_matter"]:
            
            if check_file(self.path.native[to_isolate]):
                if not overwrite : 
                    raise Exception("File already exists")
                else : 
                    self.print(f"Overwriting : {self.path.native[to_isolate]} ") 

            # Value corresponding the the encoding in the MC Thresholded
            val = 1 if to_isolate == "sulci" else -1
            saving_path = self.path.native[to_isolate]

            thresh_vol = aims.read(str(self.path.native["threshold"]))
            thresh_np = thresh_vol.np

            # Retrieving masks for the analysis
            mask = (thresh_np == val)

            thresh_np[mask] = 1
            thresh_np[~mask] = 0

            self.print(f"Saving {saving_path}")
            aims.write(thresh_vol, str(saving_path))
    
    def _get_transform_mat(self):
        if not self.path.transform_mat :
            graph = aims.read(str(self.path.graph))
            transf = aims.GraphManip.getICBM2009cTemplateTransform(graph)
            return transf
        else :
            print("Transformation matrix available")
            return None
    
    
    def resample(self, overwrite : bool = False):
        
        if check_file(self.path.icbm["resampled_icbm"]):
            if not overwrite : 
                raise Exception("File already exists")
            else : 
                self.print(f"Overwriting : {self.path.icbm['resampled_icbm']} ") 

        vol = aims.read(str(self.path.raw))
        vol_dt = vol.__array__() # Volume dtype 

        transform_mat = self._get_transform_mat()

        # New dimensions of the volume 
        output_vs = np.array(self.resample_output_voxel)
        header = aims.StandardReferentials.icbm2009cTemplateHeader()

        resampling_ratio = np.array(header["voxel_size"][:3]) / output_vs
        origin_dim = header["volume_dimension"][:3]

        new_dim = list((resampling_ratio * origin_dim).astype(int))

        # Generating the new volume
        resampled = aims.Volume(new_dim, dtype=vol_dt.dtype)
        resampled.copyHeaderFrom(header)
        resampled.header()["voxel_size"][:3] = output_vs

        #Resampler 
        resampler = aimsalgo.ResamplerFactory(vol).getResampler(3) #Interpolation Cubic 
        resampler.setDefaultValue(0)
        resampler.setRef(vol)

        #Resampling
        resampled = resampler.doit(
            transform_mat, 
            new_dim[0],
            new_dim[1],
            new_dim[2],
            output_vs
        )

        self.print(f"Saving {self.path.icbm['resampled_icbm']}")
        aims.write(resampled, filename = str(self.path.icbm["resampled_icbm"]))
        header["voxel_size"][:3] = [1,1,1]

    def _apply_mask(self,
                    source_path : Path,
                    saving_path : Path,
                    mask_path : Path,
                    ): 
        
        # Reading objects
        obj = aims.read(str(source_path))
        mask = aims.read(str(mask_path))

        # Convert mask to bool
        mask_bool = np.where(mask.np == 1, True, False)

        # Apply mask
        obj.np[~mask_bool] = 0

        self.print(f"Saving {saving_path}")
        aims.write(obj, filename=str(saving_path))
    
    def apply_masks(self, 
                    overwrite : bool = False):
        for type_mask, mask_path in self.masks_path.items(): 

            path_mask = mask_path.icbm2009

            for type_file in ["threshold", "sulci", "white_matter"]:

                if check_file(self.path.masked[type_mask][type_file]):
                    if not overwrite : 
                        raise Exception("File already exists")
                    else : 
                        self.print(f"Overwriting : {self.path.masked[type_mask][type_file]} ") 
                self._apply_mask(
                    source_path=self.path.icbm[type_file],
                    saving_path=self.path.masked[type_mask][type_file],
                    mask_path= path_mask
                )