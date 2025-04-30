from typing import Dict, List, Tuple
import PIL
import io
from soma import aims
import numpy as np
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import anatomist.headless as anahead
from betaVAE.beta_vae import VAE
from betaVAE.preprocess import UkbDataset

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

def converter_RBGA(tensor : torch.Tensor) -> np.ndarray:
    """Convert tensor to RGBA encoded numpu

    Args:
        tensor (torch.Tensor): Tensor [rgba, width, heigth]

    Returns:
        np.ndarray : Array of shape [width, height, rgba]
    """
    np_tens = tensor.numpy()
    rgba = np.transpose(np_tens, (1,2,0))
    return rgba

def adjust_in_shape(in_shape, depth):
    """
    Function to make sure that the output of the encoder is composed of integers 
    In this case : Each block (conv_x + conv_x_a) reduce by 2 the dimension of the volume.
    """

    dims=[]
    for idx in range(1, 4):
        dim = in_shape[idx]
        r = dim%(2**depth)
        if r!=0:
            dim+=(2**depth-r)
        dims.append(dim)
    return((1, dims[0]+4, dims[1], dims[2])) 

class VisualiserAnatomist :
    def __init__(self,
                 path_or_obj : Path,
                 dict_views : Dict,
                 anatomist,
                 window,
                 ):
        
        self.ana = anatomist
        self.win = window
        self.win.setHasCursor(0)

        self.path_or_obj = path_or_obj

        self.dict_views = dict_views

        self.vol_np = self.load_np()
        self.clipped = self.toAnatomist()

    def load_np(self):


        if isinstance(self.path_or_obj, np.ndarray):
            vol_np = self.path_or_obj
        else : #Path to the numpy array
            vol_np = np.load(self.path_or_obj)
        # Adapting the shape to fit anatomist format
        vol_tensor = torch.from_numpy(vol_np)
        vol_sq = vol_tensor.squeeze(0)
        vol = vol_sq.unsqueeze(-1)
        # vol_int = vol.to(dtype=torch.int16)

        return vol


    
    def toAnatomist(self):

        numpy_vol = self.vol_np
        # Transformation to anatomist volume
        aims_vol = aims.Volume(numpy_vol.numpy()) # ! Bug when conversion
        ana_vol = self.ana.toAObject(aims_vol)

        # 3D rendering 
        vol_3d = self.ana.fusionObjects(
            objects=[ana_vol],
            method= "VolumeRenderingFusionMethod"
        )
        
        clipped = self.ana.fusionObjects(
            objects=[vol_3d],
            method= "FusionClipMethod"
        )

        clipped.releaseAppRef()

        return clipped


    def positioning(self, 
                    clip_quaternion : aims.Quaternion,
                    view_quaternion : List,
                    zoom : float,
                    ):

        palette = self.gen_palette()

        self.clipped.setQuaternion(clip_quaternion)
        self.clipped.setPalette(palette)

        self.win.camera(view_quaternion = view_quaternion, zoom = zoom)


    def gen_palette(self) :

        # ! Fix the modification using : 
        #! palette_array = np.zeros((512,1,1,1,4))
        #! palette_array[100:400,:,:,:,3] = 0 # Settings middle alpha to 0
        #! palette_array[:100,:,:,:,2] = 255 # Settings begin to blue max
        #! palette_array[:100,:,:,:,1] = 255 # Settings begin to green max
        #! palette_array[400:,:,:,:,0] = 255 # Settings end to red max

        #! palette.np["v"][:] = palette_array[:]

        # Creating Palette
        pal = self.ana.createPalette("CustomPal")

        # Initialize colors
        colors = [0,0,0] * 512
        pal.setColors(colors)
        
        #Setting colors
        pal.np["v"][100:400,:,:,:,3] = 0 # Settings middle alpha to 0
        pal.np["v"][:100,:,:,:,2] = 255 # Settings begin to blue max
        pal.np["v"][:100,:,:,:,1] = 255 # Settings begin to green max
        pal.np["v"][400:,:,:,:,0] = 255 # Settings end to red max

        return pal
        
    @staticmethod
    def buffer_to_image(buffer):
        plt.savefig(buffer, format = "png")
        buffer.seek(0)
        plt.close("all")
        image = PIL.Image.open(buffer)
        image = ToTensor()(image).unsqueeze(0)[0]
        return image
    
    def show(self, buffer :bool, view_settings : Tuple):
        self.positioning(
            view_settings[0], # Slice Quaternion
            view_settings[1], # View Quaternion
            view_settings[2], # Zoom
            )

        self.win.addObjects(self.clipped)
        self.win.imshow(show = False)

        if buffer : 
            self.win.removeObjects(self.clipped)
            return self.buffer_to_image(buffer = io.BytesIO())
        else :
            plt.show()
    
    def tensor_image(self, name_setting : str) : 
        return self.show(buffer = True, view_settings = self.dict_views[name_setting])


class VisualiseExperiment :
    def __init__(
            self, 
            root_experiment : Path | str,
            root_dataset : Path | str,
            vae_settings = Dict,
            ):
            # Define the paths for the visualiser

            self.root_exp = Path(root_experiment)
            self.root_data = Path(root_dataset)
            self.vae_settings = vae_settings
            self.dataloader = UkbDataset(config = {
                "root" : self.root_data,
                "in_shape" : self.vae_settings["in_shape"]
            })

            self.paths = {}
            self.paths["model"] = self.root_exp / "vae.pt"
            self.paths["checkpoint"] = self.root_exp / "checkpoint.pt"
            self.paths["training"] = {
                "input" : self.root_exp / "input.npy",
                "output" : self.root_exp / "output.npy",
                "phase" : self.root_exp / "phase.npy",
                "id" : self.root_exp / "id.npy",
            }

            self.phase_arr = np.squeeze(np.load(self.paths["training"]["phase"]))
            self.id_arr = np.squeeze(np.load(self.paths["training"]["id"]))

    def plot_training(self):
        anatomist = anahead.Anatomist()
        win = anatomist.createWindow("3D")

        #Loading images
        inputs_arr = np.squeeze(np.load(self.paths["training"]["input"]).astype(np.int16))
        outputs_arr = np.squeeze(np.load(self.paths["training"]["output"]).astype(np.int16))

        visu_inputs = [converter_RBGA(VisualiserAnatomist(
            path_or_obj=np_obj, 
            dict_views= DICT_VIEWS, 
            anatomist = anatomist,
            window = win
        ).tensor_image("normal")) for np_obj in inputs_arr]

        visu_outputs = [converter_RBGA(VisualiserAnatomist(
            path_or_obj=np_obj, 
            dict_views= DICT_VIEWS, 
            anatomist = anatomist,
            window = win
        ).tensor_image("normal")) for np_obj in outputs_arr]

        n_sub = len(self.id_arr)
        fig, axes = plt.subplots(n_sub, 2, figsize=(8, n_sub*4))
        for i in range(n_sub):
            axes[i,0].set_axis_off()
            axes[i,1].set_axis_off()
            axes[i,0].set_title(self.id_arr[i])
            axes[i,0].imshow(visu_inputs[i], aspect="equal")
            axes[i,1].imshow(visu_outputs[i], aspect="equal")
        plt.subplots_adjust(wspace=0.01, hspace=0.05)
        fig.savefig(self.root_exp / "fig_evolution.png", format = "png")
        
    
    def view_inference(self,
                    subject : str,
                    anatomist,
                    plot : bool,
                    checkpoint : bool = False
                    ):

        # assert subject in self.dataloader.list_subjects, "Subject not in the preprocessed sub"
        win_input = anatomist.createWindow("3D")
        win_output = anatomist.createWindow("3D")

        # if torch.cuda.is_available():
        #     device = "cuda:0"
        # else : 
        #     device = "cpu"
        device = "cpu" #! Seems that cuda doesn't work on carafon 
        
        to_load = "checkpoint" if checkpoint else "model"
        # Loading model
        model = VAE(**self.vae_settings, device=device)
        model.load_state_dict(torch.load(self.paths[to_load]))
        model.to(device=device)
        index_patient = list(self.dataloader.list_subjects).index(subject)
        target = self.dataloader[index_patient]
        output = model(target[0].to(dtype = torch.float).unsqueeze(0))
        argmax_out = torch.argmax(output[0], dim = 1) #Retrieving only the reconstruction -> output[0]

        visu_input = VisualiserAnatomist(
            path_or_obj = target[0].to(dtype = torch.int16).numpy(),
            dict_views=DICT_VIEWS,
            anatomist = anatomist,
            window = win_input
        )

        visu_output = VisualiserAnatomist(
            path_or_obj = argmax_out.to(dtype = torch.int16).numpy() - 1,
            dict_views=DICT_VIEWS,
            anatomist = anatomist,
            window = win_output
        )
        
        if plot :
            fig, axes = plt.subplots(1,2, figsize = (8, 8))
            axes[0,0].set_axis_off()
            axes[0,1].set_axis_off()
            axes[0,0].set_title(subject)
            axes[0,0].imshow(converter_RBGA(visu_input.tensor_image("normal")), aspect="equal")
            axes[0,1].imshow(converter_RBGA(visu_output.tensor_image("normal")), aspect="equal")
            plt.subplots_adjust(wspace=0.01, hspace=0.05)
            plt.show()
        
        else :
            visu_input.show(buffer=False, view_settings=DICT_VIEWS["normal"])
            visu_output.show(buffer=False, view_settings=DICT_VIEWS["normal"])

