from pathlib import Path
from cerebellum_folding.data.preprocess import PipelineSubject
from cerebellum_folding.data.path import MergedMaskPath, SubjectPath
import gc

SAVING_PATH = "/neurospin/dico/tsanchez/preprocessed/ataxia"
PATH_CEREBELLUM_MASK = "/neurospin/dico/tsanchez/mask/cerebellum/27_ataxia_control_cerebellum.nii.gz"
PATH_VERMIS_MASK = "/neurospin/dico/tsanchez/mask/vermis/27_ataxia_control_vermis.nii.gz"

GRAPH_FOLDER_ATAXIA = Path("/neurospin/dico/zsun/ataxie/etudes_AlexandraDurr/database_brainvisa/cermoi")
TREE_GRAPH_ATAXIA = Path("t1mri/V1/default_analysis/folds/3.1")
RAW_FOLDER_ATAXIA = Path("/neurospin/dico/zsun/ataxie/etudes_AlexandraDurr/database_brainvisa/cermoi")
TREE_RAW_ATAXIA = Path("t1mri/V1")

NOMENCLATURE_RAW = ".nii.gz"
MASKS_TYPE = ["cerebellum", "vermis"] 


WM_THRESH = -0.3967
SULCI_THRESH = 0.464
RESAMPLE_VALUES = [0, -1, 1]
RESAMPLE_BIN = [0,1]
OUTPUT_VOXEL_SIZE = (0.5,0.5,0.5)


subjects = [
    "00004PA",
    "00002PV",
    "00005PS",
    "00006PG",
    "00020CT",
]

masks = {
    "cerebellum" : MergedMaskPath(
        path = PATH_CEREBELLUM_MASK
    ),
    "vermis" : MergedMaskPath(
        path = PATH_VERMIS_MASK
    )
}

paths_subjects = [
    SubjectPath(
                subject_id = subject,
                graph_folder = GRAPH_FOLDER_ATAXIA,
                tree_graph = TREE_GRAPH_ATAXIA,
                raw_folder = RAW_FOLDER_ATAXIA,
                tree_raw = TREE_RAW_ATAXIA,
                nomenclature_raw = NOMENCLATURE_RAW,
                masks_type = MASKS_TYPE,
                saving_folder= SAVING_PATH
    ) for subject in subjects
]

if __name__ == "__main__":
    for sub_path in paths_subjects:
        pipe = PipelineSubject(
            subject_path= sub_path,
            masks_path=masks,
            white_matter_threshold=WM_THRESH,
            sulci_threshold=SULCI_THRESH,
            # resample_values_icbm = RESAMPLE_VALUES,
            # resample_values_bin=RESAMPLE_BIN,
            resample_output_voxel=OUTPUT_VOXEL_SIZE,
            downsample_output_voxel=(1,1,1),
            verbose = True
        )
        pipe.run_pipe(overwrite=True, dilatation=5)