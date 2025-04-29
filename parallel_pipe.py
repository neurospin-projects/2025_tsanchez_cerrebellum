from pathlib import Path
import pandas as pd
from cerebellum_folding.data.preprocess import PipelineSubject
from cerebellum_folding.data.path import MergedMaskPath
from cerebellum_folding import UkbSubject
from joblib import Parallel, delayed
from tqdm import tqdm

SAVING_PATH = Path("/neurospin/dico/data/cerebellum/datasets/Ukbio")
PATH_VERMIS_MASK = Path("/neurospin/dico/data/cerebellum/mask/vermis/27_ataxia_control_vermis.nii.gz")

WM_THRESH = -0.3967
SULCI_THRESH = 0.464
OUTPUT_VOXEL_SIZE = (0.5,0.5,0.5)

qc_file = Path("/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/qc_post.tsv")

df_qc = pd.read_csv(qc_file, sep = "\t")
clean_subjects = df_qc[df_qc["qc"] == 1]
subjects = list(clean_subjects["participant_id"])

print(f"Number subjects : {len(subjects)}")

print("Check graph paths and raw...")
path_subjects = [UkbSubject(sub_id, SAVING_PATH) for sub_id in tqdm(subjects)]

masks = {
    "vermis" : MergedMaskPath(
        path = PATH_VERMIS_MASK
    )
}

delete_tmp_settings = {
    "rm_icbm" : False,
    "rm_mean_curvature" : True,
    "rm_masked" : True,
    "rm_crop" : True,
}


def run_pipe(path_sub):
    PipelineSubject(
        subject_path=path_sub,
        masks_path=masks,
        white_matter_threshold=WM_THRESH,
        sulci_threshold=SULCI_THRESH,
        resample_output_voxel=OUTPUT_VOXEL_SIZE,
        verbose = False,
        ).run_pipe(overwrite=False, dilatation=5,delete_tmp_settings=delete_tmp_settings)
    print(f"Done : {path_sub.id} ")

if __name__ == "__main__":
    Parallel(n_jobs=30, verbose=0)(delayed(run_pipe)(sub_path) for sub_path in path_subjects)
