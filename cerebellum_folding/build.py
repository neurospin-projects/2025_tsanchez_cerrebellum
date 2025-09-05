"""Building SubjectPath objects specific to each dataset"""

from pathlib import Path
from typing import List, Union
from cerebellum_folding.path import SubjectPath

### Paths

GRAPH_FOLDER_UKB = Path("/tmp/tsanchez") #Mounted on the local server
TREE_GRAPH_UKB = Path("ses-2/anat/t1mri/default_acquisition/default_analysis/folds/3.1")
RAW_FOLDER_UKB = Path("/tmp/tsanchez")
TREE_RAW_UKB = Path("ses-2/anat/t1mri/default_acquisition")

GRAPH_FOLDER_ATAXIA = Path("/neurospin/dico/zsun/ataxie/etudes_AlexandraDurr/database_brainvisa/biosca")
TREE_GRAPH_ATAXIA = Path("t1mri/E1/default_analysis/folds/3.1")
RAW_FOLDER_ATAXIA = Path("/neurospin/dico/zsun/ataxie/etudes_AlexandraDurr/database_brainvisa/biosca")
TREE_RAW_ATAXIA = Path("t1mri/E1")

GRAPH_FOLDER_ABCD = Path("/neurospin/dico/tsanchez/tmp") # Mounted on local server
TREE_GRAPH_ABCD = Path("ses-1/anat/t1mri/default_acquisition/default_analysis/folds/3.1")
RAW_FOLDER_ABCD = Path("/neurospin/dico/tsanchez/tmp")
TREE_RAW_ABCD = Path("ses-1/anat/t1mri/default_acquisition")

MASKS_TYPE = ["cerebellum", "vermis"] 
NOMENCLATURE_RAW = ".nii.gz"


class AtaxiaSubject(SubjectPath):
    def __init__(self,
                 subject_id : str, 
                 saving_folder : Union[str, Path],
                 masks_type : List[str] = MASKS_TYPE,
                 ):
        super().__init__(
            subject_id=subject_id,
            graph_folder=GRAPH_FOLDER_ATAXIA,
            tree_graph=TREE_GRAPH_ATAXIA,
            raw_folder=RAW_FOLDER_ATAXIA,
            tree_raw=TREE_RAW_ATAXIA,
            nomenclature_raw=NOMENCLATURE_RAW,
            masks_type = masks_type,
            saving_folder=saving_folder
        )

class UkbSubject(SubjectPath):
    def __init__(self,
                 subject_id : str, 
                 saving_folder : Union[str, Path],
                 masks_type : List[str] = MASKS_TYPE,
                 ):
        super().__init__(
            subject_id=subject_id,
            graph_folder=GRAPH_FOLDER_UKB,
            tree_graph=TREE_GRAPH_UKB,
            raw_folder=RAW_FOLDER_UKB,
            tree_raw=TREE_RAW_UKB,
            nomenclature_raw=NOMENCLATURE_RAW,
            masks_type = masks_type,
            saving_folder=saving_folder
        )


class ABCDSubject(SubjectPath):
    def __init__(self,
                 subject_id : str, 
                 saving_folder : Union[str, Path],
                 masks_type : List[str] = MASKS_TYPE,
                 ):
        super().__init__(
            subject_id=subject_id,
            graph_folder=GRAPH_FOLDER_ABCD,
            tree_graph=TREE_GRAPH_ABCD,
            raw_folder=RAW_FOLDER_ABCD,
            tree_raw=TREE_RAW_ABCD,
            nomenclature_raw=NOMENCLATURE_RAW,
            masks_type = masks_type,
            saving_folder=saving_folder
        )