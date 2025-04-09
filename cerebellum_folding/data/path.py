"""Sub lib containing all the file management for the subjects"""

from pathlib import Path
from typing import Union, List
import os

def check_file(path : Path):
    return (os.path.exists(path) and os.path.isfile(path))

class BasePath :
    def __init__(self,
                 subject_id : str,
                 graph_folder : Union[str, Path],
                 tree_graph : Union[str, Path],
                 raw_folder : Union[str, Path],
                 tree_raw : Union[str, Path],
                 nomenclature_raw : str,
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
        """
        
        self.id = subject_id
        self.folders = {"raw" : Path(raw_folder),
                        "graph" : Path(graph_folder)}
        
        self.graph = graph_folder / subject_id / tree_graph / f"R{subject_id}.arg" #Always the same structure of name
        self.raw = raw_folder / subject_id / tree_raw / f"{subject_id}{nomenclature_raw}"

        self._validate_path()

    def _validate_path(self) :
        """Check if paths exists

        Raises:
            ValueError: Path does not exist
        """
        if not os.path.exists(self.graph) :
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
                 transform_path : Union[str, Path, None] = None
                 ):
        # Init the base folder
        super().__init__(subject_id, graph_folder, tree_graph, raw_folder , tree_raw, nomenclature_raw)

        NATIVE_FOLDER = "native"
        ICBM2009_FOLDER = "ICBM2009c"
        MASKED_FOLDER = "masked"

        # TODO : Add reports for the transform with the matrix that is used to register from native 

        # Path where all the data is saved
        self.save = Path(saving_folder) / self.id
        self.available_masks = masks_type
        self.native = dict()
        self.icbm = dict()
        self.masked = dict([(key, dict()) for key in masks_type]) 

        # File in the native space
        self.native["mean_curvature"] = self.save / NATIVE_FOLDER / f"{self.id}_mean_curvature.nii.gz"

        self.native["threshold"] = self.save / NATIVE_FOLDER / f"{self.id}_thresh_native.nii.gz"
        self.native["white_matter"] = self.save / NATIVE_FOLDER / f"{self.id}_white_matter_native.nii.gz"
        self.native["sulci"] = self.save / NATIVE_FOLDER / f"{self.id}_sulci_native.nii.gz"

        # File in the ICBM2009 space
        self.transform_mat = transform_path

        #Without skel
        self.icbm["threshold"] = self.save / ICBM2009_FOLDER / f"{self.id}_thresh_icbm2009.nii.gz"
        self.icbm["white_matter"] = self.save / ICBM2009_FOLDER / f"{self.id}_white_matter_icbm2009.nii.gz"
        self.icbm["sulci"] = self.save / ICBM2009_FOLDER / f"{self.id}_sulci_icbm2009.nii.gz"

        for key in self.masked.keys():
            self.masked[key]["threshold"] = self.save / MASKED_FOLDER / key / f"{self.id}_masked_tresh_{key}.nii.gz"
            self.masked[key]["white_matter"] = self.save / MASKED_FOLDER / key / f"{self.id}_masked_white_matter_{key}.nii.gz"
            self.masked[key]["sulci"] = self.save / MASKED_FOLDER / key / f"{self.id}_masked_sulci_{key}.nii.gz"

    def _native_exists(self): 
        return dict([(key, os.path.exists(self.native[key])) for key in self.native.keys()])
        
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
            "native" : self._native_exists(),
            "icbm" : self._icbm_exists(),
            "masked" : self._masked_exists(),
        }
    
    @property
    def mc(self):
        return self.native["mean_curvature"]
    
    @mc.setter
    def mc(self, val):
        self.native["mean_curvature"] = val


    def _create_masked_saving_folders(self):
        for type_mask in self.available_masks : 
            os.mkdir(self.save / "masked" / type_mask)
    
    def create_saving_paths(self) :
        if not os.path.exists(self.save):
            os.mkdir(self.save)
            os.mkdir(self.save / "native")
            os.mkdir(self.save / "ICBM2009c")
            os.mkdir(self.save / "masked")
            self._create_masked_saving_folders()


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

        
        # TODO : Add reports for the transform with the matrix that is used to register from native 