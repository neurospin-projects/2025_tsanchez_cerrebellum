"""Sub lib containing all the file management for the subjects"""

from pathlib import Path
import shutil
from typing import Union, List
import os

def check_file(path : Path):
    """Check if a file exists

    Args:
        path (Path): Path of the file

    Returns:
        bool: True if file exists, False otherwise
    """
    return (os.path.exists(path) and os.path.isfile(path))

class BasePath :
    def __init__(self,
                 subject_id : str,
                 graph_folder : Union[str, Path],
                 tree_graph : Union[str, Path],
                 raw_folder : Union[str, Path],
                 tree_raw : Union[str, Path],
                 nomenclature_raw : str,
                 transform_folder : Union[str,None],
                 type_transform : Union[str,None],
                 ):
        """Base class for path management

        Args:
            subject_id (str): Id of the subject in the different databases

            graph_folder (Union[str, Path]): Folder containing the subject folder (with the graph file)

            tree_graph (Union[str, Path]): Path to the folder containing the graph file relative to the subject folder

            raw_folder (Union[str, Path]): Folder containing the raw subject files

            tree_raw (Union[str, Path]): Folder containing the subject folder (with the raw file)

            nomenclature_raw (str): Nomenclature of the subject raw file -> The file name has this structure at the end `<subject_id><nomenclature>`
            e.g. 199133_raw.nii.gz will have this features : 
            - subject_id = 199133
            - nomenclature_raw = "_raw.nii.gz"    

            transform_folder(Union[str,None]) : Folder containing the transform file (.trm)

            type_transform (Union[str,None]) : Type of transform (Talairach, icbm, ...)
        """                                       
        
        self.id = subject_id
        self.folders = {"raw" : Path(raw_folder),
                        "graph" : Path(graph_folder)}
        
        self.graph = graph_folder / subject_id / tree_graph / f"R{subject_id}.arg" #Always the same structure of name
        self.raw = raw_folder / subject_id / tree_raw / f"{subject_id}{nomenclature_raw}"
        
        if type_transform and transform_folder: 
            if type_transform in ["ACPC", "MNI"] :
                self.transform_mat = transform_folder / subject_id / tree_raw / "registration" / f"RawT1-{subject_id}_default_acquisition_TO_Talairach-{type_transform}.trm"
                if not os.path.exists(self.transform_mat) :
                    raise ValueError(f"Object does not exist : {self.transform_mat}")
            else :
                raise ValueError("Transform type available : ACPC or MNI")
        else : 
            self.transform_mat = None
        
        self.type_transform = type_transform

        self._validate_path()

    def _validate_path(self) :
        """Check if paths exists

        Raises:
            ValueError: Path does not exist
        """
        if not os.path.exists(self.graph) and not self.transform_mat:
            raise ValueError(f"Object does not exist : {self.graph}")
        elif not os.path.exists(self.raw) :
            raise ValueError(f"Object does not exist : {self.raw}")
    
    def __repr__(self):
        return f"Paths({self.id})"

        
class SubjectPath(BasePath) :
    def __init__(self,
                 subject_id : str,
                 graph_folder : Union[str, Path],
                 tree_graph : Union[str, Path],
                 raw_folder : Union[str, Path],
                 tree_raw : Union[str, Path],
                 nomenclature_raw : str,
                 masks_type : List[str],
                 saving_folder: Union[str, Path],
                 transform_folder : Union[str | Path, None] = None,
                 type_transform : Union[str, None] = None, 
                 ):
        """Individual class that contains / create all the required paths for a subject

        Args:
            subject_id (str): Id of the subject in the different databases

            graph_folder (Union[str, Path]): Folder containing the subject folder (with the graph file)

            tree_graph (Union[str, Path]): Path to the folder containing the graph file relative to the subject folder

            raw_folder (Union[str, Path]): Folder containing the raw subject files

            tree_raw (Union[str, Path]): Folder containing the subject folder (with the raw file)

            nomenclature_raw (str): Nomenclature of the subject raw file -> The file name has this structure at the end `<subject_id><nomenclature>`
            e.g. 199133_raw.nii.gz will have this features : 
            - subject_id = 199133
            - nomenclature_raw = "_raw.nii.gz"    

            transform_folder(Union[str,None]) : Folder containing the transform file (.trm)

            type_transform (Union[str,None]) : Type of transform (Talairach, icbm, ...)
        """
        # Init the base folder
        super().__init__(subject_id, graph_folder, tree_graph, raw_folder , tree_raw, nomenclature_raw, transform_folder, type_transform)

        self._ICBM2009_FOLDER = "ICBM2009c"
        self._MASKED_FOLDER = "masked"
        self._CROP_FOLDER = "cropped"

        # TODO : Add reports for the transform with the matrix that is used to register from native 

        # Path where all the data is saved
        self.save = Path(saving_folder) / self.id
        self.available_masks = masks_type
        self.icbm = dict()
        self.masked = dict([(key, dict()) for key in masks_type]) 
        self.cropped = dict([(key, dict()) for key in masks_type]) 
        self.numpy = dict([(key, dict()) for key in masks_type]) 

        # File in the ICBM2009 space (1st step of the pipeline)
        #Without skel
        self.icbm["resampled_icbm"] = self.save / self._ICBM2009_FOLDER / f"{self.id}_resampled_icbm.nii.gz"
        self.icbm["mean_curvature"] = self.save / self._ICBM2009_FOLDER / f"{self.id}_mean_curvature_icbm.nii.gz"
        self.icbm["threshold"] = self.save / self._ICBM2009_FOLDER / f"{self.id}_tresh_mc.nii.gz"

        for key in self.masked.keys():
            self.masked[key]["threshold"] = self.save / self._MASKED_FOLDER / key / f"{self.id}_masked_tresh_{key}.nii.gz"
            self.masked[key]["resampled_icbm"] = self.save / self._MASKED_FOLDER / key / f"{self.id}_masked_t1mri_{key}.nii.gz"
            self.masked[key]["mean_curvature"] = self.save / self._MASKED_FOLDER / key / f"{self.id}_masked_mean_curvature_{key}.nii.gz"

        for key in self.cropped.keys():
            self.cropped[key]["threshold"] = self.save / self._CROP_FOLDER/ key / f"{self.id}_crop_tresh_{key}.nii.gz"
            self.cropped[key]["resampled_icbm"] = self.save / self._CROP_FOLDER / key / f"{self.id}_crop_t1mri_{key}.nii.gz"
            self.cropped[key]["mean_curvature"] = self.save / self._CROP_FOLDER / key / f"{self.id}_crop_mean_curvature_{key}.nii.gz"
        
        for key in self.cropped.keys():
            self.numpy[key] = self.save / f"{self.id}_{key}.npy"


    def _icbm_exists(self): 
        return dict([(key, os.path.exists(self.icbm[key])) for key in self.icbm.keys()])

    def _masked_exists(self): 
        return dict([
            (mask_type, dict([(key, os.path.exists(self.masked[mask_type][key])) for key in self.masked[mask_type].keys()]))
            for mask_type in self.available_masks
        ])


    @property
    def dict_exists(self):
        return {
            "icbm" : self._icbm_exists(),
            "masked" : self._masked_exists(),
        }
    
    @property
    def mc(self):
        return self.icbm["mean_curvature"]
    
    @mc.setter
    def mc(self, val):
        self.icbm["mean_curvature"] = val


    def _create_masked_saving_folders(self):
        for type_mask in self.available_masks : 
            os.mkdir(self.save / "masked" / type_mask)

    def _create_cropped_saving_folders(self):
        for type_mask in self.available_masks : 
            os.mkdir(self.save / "cropped" / type_mask)
    
    def create_saving_paths(self) :
        if not os.path.exists(self.save):
            os.mkdir(self.save)

        if not os.path.exists(self.save / "ICBM2009c"):
            os.mkdir(self.save / "ICBM2009c")

        if not os.path.exists(self.save / "masked"):
            os.mkdir(self.save / "masked")
            self._create_masked_saving_folders()

        if not os.path.exists(self.save / "cropped"):
            os.mkdir(self.save / "cropped")
            self._create_cropped_saving_folders()

    def clear_folder(self,
                     rm_icbm : bool = False,
                     rm_mean_curvature : bool = True,
                     rm_masked : bool = False,
                     rm_crop : bool = False): 
        # Removing ICBM folder
        if rm_icbm : 
            try :
                shutil.rmtree(self.save / self._ICBM2009_FOLDER)
            except Exception as e : 
                print(e)
        elif not rm_icbm and rm_mean_curvature : 
            try : 
                os.remove(self.icbm["mean_curvature"])
                os.remove(str(self.icbm["mean_curvature"]) + ".minf")
            except Exception as e: 
                print(e)


        # Removing masked
        if rm_masked : 
            try :
                shutil.rmtree(self.save / self._MASKED_FOLDER)
            except Exception as e : 
                print(e)

        # Removing crop
        if rm_crop : 
            try :
                shutil.rmtree(self.save / self._CROP_FOLDER)
            except Exception as e : 
                print(e)


class MaskPath(BasePath):
    def __init__(self,
                subject_id : str,
                graph_folder : Union[Path, str],
                tree_graph : Union[Path, str],
                raw_folder : Union[Path, str],
                tree_raw : Union[Path, str],
                nomenclature_raw : str, 
                mask_type : str ,
                saving_path : Union[str, Path]
                ):
        super().__init__(subject_id, graph_folder, tree_graph, raw_folder, tree_raw, nomenclature_raw)

        self.save = saving_path #Folder to save the files
        self.type = mask_type #Mask type 

        self.native = self.save / self.type / f"{self.id}_{self.type}_native.nii.gz"
        self.dilated = self.save / self.type / f"{self.id}_{self.type}_native_dilatation.nii.gz"
        self.icbm2009 = self.save / self.type / f"{self.id}_{self.type}_ICBM2009c.nii.gz"

        self.create_saving_paths()

    def create_saving_paths(self) :
        if not os.path.exists(self.save / self.type):
            os.mkdir(self.save / self.type)

        
        # TODO : Add reports for the transform with the matrix that is used to register from native 

class MergedMaskPath: 
    def __init__(self, path):
        self.icbm2009 = Path(path)
        self.id = self.icbm2009.name