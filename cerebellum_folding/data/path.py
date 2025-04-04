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
        
        self.graph = graph_folder / subject_id / tree_graph / f"{subject_id}.arg" #Always the same structure of name
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
        
        
    
