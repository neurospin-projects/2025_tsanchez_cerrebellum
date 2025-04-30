from typing import Dict, List, Tuple
import PIL
import io
from soma import aims
import numpy as np
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Produce for one image all the views
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
        vol_int = vol.to(dtype=torch.int16)

        return vol_int


    
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
    
    def tensor_image(self, name_setting : str, buffer = True) : 
        return self.show(buffer = buffer, view_settings = self.dict_views[name_setting])
    
    
    
