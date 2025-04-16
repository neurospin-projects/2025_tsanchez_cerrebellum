"""Library for the preprocessing pipeline"""


from cerebellum_folding.data.path import SubjectPath, MaskPath, check_file

import deep_folding.brainvisa.utils.dilate_mask as dl 
from deep_folding.brainvisa.utils.resample import resample
import subprocess as sp

from typing import List, Dict
from pathlib import Path
from soma import aims, aimsalgo
import numpy as np


class PipelineSubject :
    def __init__(self,
                 subject_path : SubjectPath,
                 masks_path : Dict[str, MaskPath], 
                 white_matter_threshold : float, 
                 sulci_threshold : float, 
                 resample_output_voxel : List[int], 
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
        self.resample_output_voxel = resample_output_voxel

        self.verbose = verbose
    
    def print(self, string):
        if self.verbose : 
            print(string)


    def threshold_mean_curvature(self,
                                overwrite : bool = False,
                                ):
        
        if not check_file(self.path.mc):
            raise Exception(f"Mean curvature file not available : {self.path.mc}")
        
        if check_file(self.path.icbm["threshold"]):
            if not overwrite : 
                raise Exception("File already exists")
            else : 
               self.print(f"Overwriting : {self.path.icbm['threshold']}") 

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

        self.print(f"Saving to {self.path.icbm['threshold']}")
        aims.write(obj, filename=str(self.path.icbm['threshold']))


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
        resampled.copyHeaderFrom(header) #Be careful it is a shallow copy 
        resampled.header()["voxel_size"] = list(output_vs) + [1] 

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

    def compute_mean_curvature(self, sigma : float = 1, overwrite : bool = False):

        if check_file(self.path.icbm["mean_curvature"]):
            if not overwrite : 
                raise Exception("File already exists")
            else : 
                self.print(f"Overwriting : {self.path.icbm['mean_curvature']} ") 
        
        CMD_MEAN_CURV = ["VipGeometry",
                "-m", "mc",
                "-s", f"{sigma}", 
                "-i", self.path.icbm["resampled_icbm"],
                "-o", self.path.icbm["mean_curvature"]
                ]

        output = sp.run(CMD_MEAN_CURV, capture_output=True)
        self.print(output)
        self.print(f"Saving {self.path.icbm['mean_curvature']}")

    def _apply_mask(self,
                    source_path : Path,
                    saving_path : Path,
                    mask_path : Path,
                    dilatation : int,
                    ): 
        
        # Reading objects
        obj = aims.read(str(source_path))
        mask = aims.read(str(mask_path))
        dilated_mask = dl.dilate(mask, radius=dilatation)

        # Convert mask to bool
        mask_bool = np.where(dilated_mask.np == 1, True, False)

        # Apply mask
        obj.np[~mask_bool] = 0

        self.print(f"Saving {saving_path}")
        aims.write(obj, filename=str(saving_path))
    
    def apply_masks(self,
                    dilatation : int, 
                    overwrite : bool = False):
        for type_mask, mask_path in self.masks_path.items(): 

            path_mask = mask_path.icbm2009

            for type_file in ["threshold", "resampled_icbm"]:

                if check_file(self.path.masked[type_mask][type_file]):
                    if not overwrite : 
                        raise Exception("File already exists")
                    else : 
                        self.print(f"Overwriting : {self.path.masked[type_mask][type_file]} ") 
                self._apply_mask(
                    source_path=self.path.icbm[type_file],
                    saving_path=self.path.masked[type_mask][type_file],
                    mask_path= path_mask,
                    dilatation=dilatation
                )

    def _compute_bbox(self, dilatation):
        type_masks = self.masks_path.keys()
        bbox_masks = dict()
        for type_mask in type_masks : 
            mask = aims.read(str(self.masks_path[type_mask].icbm2009))
            dilated_mask = dl.dilate(mask, radius=dilatation)
            val = np.array(np.where(dilated_mask.np == 1)) #Dim (4, flatten of 3D)
            bbox_masks[type_mask] = np.min(val, axis = 1), np.max(val, axis = 1)
        return bbox_masks
    
    def apply_bbox(self,dilatation, overwrite : bool = False):

        # Compute bounding box for each mask
        bbox_masks = self._compute_bbox(dilatation=dilatation)
        self.print("Bbox computed" )
        print(bbox_masks)

        for type_mask, bbox in bbox_masks.items():
            for type_file in ["threshold", "resampled_icbm"]: 

                if check_file(self.path.cropped[type_mask][type_file]):
                    if not overwrite : 
                        raise Exception("File already exists")
                    else : 
                        self.print(f"Overwriting : {self.path.cropped[type_mask][type_file]} ") 

                obj = aims.read(str(self.path.masked[type_mask][type_file]))
                cropped = aims.VolumeView(obj, bbox[0], bbox[1] - bbox[0])
                self.print(f"Cropped : {cropped.header()}")

                self.print(f"Saving {self.path.cropped[type_mask][type_file]}")
                aims.write(cropped, str(self.path.cropped[type_mask][type_file]))


                
                if type_file == "threshold" :
                    np_crop = cropped.np.copy()
                    np_crop = np_crop.astype(np.int16)
                    np.save(self.path.numpy[type_mask], np_crop)
                

            
    def run_pipe(self, overwrite : bool = False, dilatation : int = 2):
        try :
            self.resample(overwrite=overwrite)
        except Exception as e :
            self.print(e)

        try :
            self.compute_mean_curvature(overwrite=overwrite)
        except Exception as e :
            self.print(e)

        try :
            self.threshold_mean_curvature(overwrite=overwrite)
        except Exception as e :
            self.print(e)

        try :
            self.apply_masks(overwrite=overwrite, dilatation=dilatation)
        except Exception as e :
            self.print(e)

        try :
            self.apply_bbox(overwrite=overwrite, dilatation=dilatation)
        except Exception as e :
            self.print(e)


class PipelineMask :
    def __init__(self,
                 mask_path : MaskPath,
                 sub_struct_mask : List[int],
                 resample_values: List[int],
                 output_voxel : List[int], 
                 verbose : bool = False
                 ):

        self.path = mask_path

        self.path.create_saving_paths()

        self.sub_struct_mask = sub_struct_mask
        self.resample_val = resample_values
        self.output_voxel = output_voxel
        self.verbose = verbose

    def print(self, string):
        if self.verbose :
            print(string)
    
    def retrieve_structure_mask(self, overwrite : bool = False):
        if check_file(self.path.native):
            if not overwrite : 
                raise Exception("File already exists")
            else : 
                self.print(f"Overwriting : {self.path.native} ") 

        vol = aims.read(str(self.path.raw))

        vol_np = vol.np

        to_keep = np.isin(vol_np, self.sub_struct_mask)
        vol_np[to_keep] = 1
        vol_np[~to_keep] = 0

        self.print(f"Saving : {self.path.native}")
        aims.write(vol, filename=str(self.path.native))

    def _get_transform_mat(self):
        graph = aims.read(str(self.path.graph))
        transf = aims.GraphManip.getICBM2009cTemplateTransform(graph)
        return transf
    
    def _apply_ICBM2009c(self,
                         source_path : Path,
                         saving_path : Path,
                         resample_values : List[int],
                         ):

        # Loading object
        native_obj = aims.read(str(source_path))
        mat_transform = self._get_transform_mat()

        #Convert for sanity
        c = aims.Converter(intype=native_obj, outtype=aims.Volume('S16'))
        native_obj = c(native_obj)

        resampled_to_ICBM2009c = resample(
            input_image=native_obj, 
            transformation=mat_transform,
            output_vs=self.output_voxel,
            background=0,
            values=resample_values,
            verbose=self.verbose
        )

        self.print(f"Saving {saving_path}")
        aims.write(resampled_to_ICBM2009c, filename=str(saving_path))

    def transform_ICBM2009c(self, overwrite : bool = False):

        if check_file(self.path.icbm2009):
            if not overwrite : 
                raise Exception("File already exists")
            else : 
                self.print(f"Overwriting : {self.path.icbm2009} ") 

        self._apply_ICBM2009c(source_path= self.path.native,
                                saving_path=self.path.icbm2009,
                                resample_values=self.resample_val)
    
    def run_pipeline(self,overwrite : bool = False):

        try :
            self.retrieve_structure_mask(overwrite=overwrite)
        except Exception as e : 
            self.print(e)

        try :
            self.transform_ICBM2009c(overwrite=overwrite)
        except Exception as e : 
            self.print(e)