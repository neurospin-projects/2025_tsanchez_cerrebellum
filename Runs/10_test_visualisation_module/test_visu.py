from betaVAE.visualisation_anatomist import VisualiseExperiment, adjust_in_shape
import anatomist.api as anapi
from pathlib import Path

ROOT_EXPERIMENT = Path("/neurospin/dico/tsanchez/Test_BetaVAE/2025-04-28/12-06-47")
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

visu_exp = VisualiseExperiment(
    root_experiment=ROOT_EXPERIMENT, 
    root_dataset= ROOT_DATA,
    vae_settings= vae_settings
)

if __name__ == "__main__":
    anatomist = anapi.Anatomist()
    subject = "sub-3824568"
    visu_exp.view_inference(subject=subject,
                            anatomist=anatomist,
                            plot = False,
                            checkpoint=True)
    input("Press any key")