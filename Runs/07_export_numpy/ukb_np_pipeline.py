from pathlib import Path
from cerebellum_folding.data.preprocess import PipelineSubject
from cerebellum_folding.data.path import MergedMaskPath
from cerebellum_folding import UkbSubject

SAVING_PATH = Path("/neurospin/dico/tsanchez/preprocessed/ukb_np")
PATH_CEREBELLUM_MASK = Path("/neurospin/dico/tsanchez/mask/cerebellum/27_ataxia_control_cerebellum.nii.gz")
PATH_VERMIS_MASK = Path("/neurospin/dico/tsanchez/mask/vermis/27_ataxia_control_vermis.nii.gz")

WM_THRESH = -0.3967
SULCI_THRESH = 0.464
RESAMPLE_VALUES = [0, -1, 1]
RESAMPLE_BIN = [0,1]
OUTPUT_VOXEL_SIZE = (0.5,0.5,0.5)

subjects = [
    "sub-1000021",
    "sub-1000325",
    "sub-1000458",
    "sub-1000575",
    "sub-1000606",
    "sub-1000715",
    "sub-1000963",
    "sub-1001107",
    "sub-1001393",
]

path_subjects = [UkbSubject(sub_id, SAVING_PATH) for sub_id in subjects]

masks = {
    "vermis" : MergedMaskPath(
        path = PATH_VERMIS_MASK
    )
}

delete_tmp_settings = {
    "rm_icbm" : False,
    "rm_masked" : True,
    "rm_crop" : True,
}

if __name__ == "__main__":
    for path_sub in path_subjects : 
        PipelineSubject(
            subject_path=path_sub,
            masks_path=masks,
            white_matter_threshold=WM_THRESH,
            sulci_threshold=SULCI_THRESH,
            resample_output_voxel=OUTPUT_VOXEL_SIZE,
            verbose = True
            ).run_pipe(overwrite=True, dilatation=5,delete_tmp_settings=delete_tmp_settings)
    