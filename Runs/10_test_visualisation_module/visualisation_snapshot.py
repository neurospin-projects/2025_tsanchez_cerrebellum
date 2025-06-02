from betaVAE.visualisation_anatomist import VisualiserAnatomist, VisualiseExperiment, adjust_in_shape
from pathlib import Path
import anatomist.headless as ana
from soma import aims

anatomist = ana.Anatomist()
win = anatomist.createWindow("3D")

SLICE_CLIP = aims.Quaternion([0.6427876096865394,
  -0.3420201433256688,
  0.6427876096865394,
  0.3420201433256688])

VIEW_SAGITTAL = [0.5,-0.5,-0.5,0.5]
OBLIC_VIEW = [0.6,-0.2,-0.25,0.75] 

DICT_VIEWS = {
    "normal" : (
        SLICE_CLIP,
        VIEW_SAGITTAL,
        1),

    "oblic" : (
        SLICE_CLIP,
        OBLIC_VIEW,
        1)
}

#ROOT_EXPERIMENT = Path("/neurospin/dico/tsanchez/Test_BetaVAE/2025-05-16/15-45-05")
ROOT_EXPERIMENT = Path("/neurospin/dico/tsanchez/Test_BetaVAE/2025-05-27/17-31-16_")
ROOT_DATA = Path("/neurospin/dico/tsanchez/preprocessed/UKBio1000")

DEPTH = 3
IN_SHAPE_WOUT_ADJUST = [1, 54, 120, 139] #One from the config.yaml file
IN_SHAPE = adjust_in_shape(IN_SHAPE_WOUT_ADJUST, depth=DEPTH)
# ! Specific to the different models
N_LATENT = 64 

vae_settings = {
    "in_shape" : IN_SHAPE, 
    "n_latent" : N_LATENT,
    "depth" : DEPTH}

to_plot = [
"kl=2_n=1024_weights=[2, 1, 2]",
"kl=4_n=1024_weights=[2, 1, 2]",
"kl=8_n=1024_weights=[2, 1, 2]",
]


if __name__ == "__main__" : 
    for exp in to_plot :
        visu_exp = VisualiseExperiment(
            root_experiment=ROOT_EXPERIMENT / exp, 
            root_dataset= ROOT_DATA,
            vae_settings= vae_settings
        )
        visu_exp.plt_plot(start = 20 ,stop = 59) 
        #visu_exp.plot_training(save_full=False)
