from cerebellum_folding.data.preprocess import PipelineSubject
from soma import aims
import numpy as np
from cerebellum_folding import UkbSubject
from cerebellum_folding.data.path import MergedMaskPath

PATH_CEREBELLUM_MASK = "/neurospin/dico/tsanchez/mask/cerebellum/27_ataxia_control_cerebellum.nii.gz"
PATH_VERMIS_MASK = "/neurospin/dico/tsanchez/mask/vermis/27_ataxia_control_vermis.nii.gz"


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

masks = {
    "cerebellum" : MergedMaskPath(
        path = PATH_CEREBELLUM_MASK
    ),
    "vermis" : MergedMaskPath(
        path = PATH_VERMIS_MASK
    )
}
masks["cerebellum"].icbm2009, masks["vermis"].icbm2009

SAVING_PATH = "/neurospin/dico/tsanchez/preprocessed/ukb"

WM_THRESH = -0.3967
SULCI_THRESH = 0.464
RESAMPLE_VALUES = [0, -1, 1]
RESAMPLE_BIN = [0,1]
OUTPUT_VOXEL_SIZE = (0.5,0.5,0.5)


paths_ukb = [UkbSubject(subject, SAVING_PATH) for subject in subjects ]


if __name__ == "__main__":

    for sub_path in paths_ukb:
        pipe = PipelineSubject(
            subject_path= sub_path,
            masks_path=masks,
            white_matter_threshold=WM_THRESH,
            sulci_threshold=SULCI_THRESH,
            resample_values_icbm = RESAMPLE_VALUES,
            resample_values_bin=RESAMPLE_BIN,
            output_voxel=OUTPUT_VOXEL_SIZE,
            verbose = True
        )

        pipe.run_pipe(overwrite= True, dilatation =5)