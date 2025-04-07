"""Sub lib containing all the file management for the subjects"""

from pathlib import Path
from typing import Union
import os

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
                 saving_folder: Union[str, Path]
                 ):
        # Init the base folder
        super().__init__(subject_id, graph_folder, tree_graph, raw_folder , tree_raw, nomenclature_raw)

        NATIVE_FOLDER = "native"
        ICBM2009_FOLDER = "ICBM2009c"
        CROP_FOLDER = "crop"

        # TODO : Add reports for the transform with the matrix that is used to register from native 

        # Path where all the data is saved
        self.save = Path(saving_folder) / self.id

        # File in the native space
        self.mc = self.save / NATIVE_FOLDER / f"{self.id}_mean_curvature.nii.gz"
        self.thresh = self.save / NATIVE_FOLDER / f"{self.id}_thresh_native.nii.gz"
        self.white_matter_native = self.save / NATIVE_FOLDER / f"{self.id}_white_matter_native.nii.gz"
        self.sulci_native = self.save / NATIVE_FOLDER / f"{self.id}_sulci_native.nii.gz"

        # File in the ICBM2009 space

        #Without skel
        self.thresh_ICBM = self.save / ICBM2009_FOLDER / f"{self.id}_thresh_icbm2009.nii.gz"
        self.white_matter_ICBM = self.save / ICBM2009_FOLDER / f"{self.id}_white_matter_icbm2009.nii.gz"
        self.sulci_ICBM = self.save / ICBM2009_FOLDER / f"{self.id}_sulci_icbm2009.nii.gz"

        #With skel
        self.thresh_wSkel_ICBM = self.save / ICBM2009_FOLDER / f"{self.id}_thresh_wSkel_icbm2009.nii.gz"
        self.white_matter_wSkel_ICBM = self.save / ICBM2009_FOLDER / f"{self.id}_white_matter_wSkel_icbm2009.nii.gz"
        self.sulci_wSkel_ICBM = self.save / ICBM2009_FOLDER / f"{self.id}_sulci_wSkel_icbm2009.nii.gz"

        # Cropped files : 
        self.thresh_crop = self.save / CROP_FOLDER / f"{self.id}_crop_tresh.nii.gz"
        self.white_matter_crop = self.save / CROP_FOLDER / f"{self.id}_crop_white_matter.nii.gz"
        self.sulci_crop = self.save / CROP_FOLDER / f"{self.id}_crop_sulci.nii.gz"


class MaskPath(BasePath):
    def __init__(self,
                subject_id,
                graph_folder,
                tree_graph,
                raw_folder,
                tree_raw,
                nomenclature_raw, 
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