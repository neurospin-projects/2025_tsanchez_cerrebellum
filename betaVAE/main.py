# /usr/bin/env python3
# coding: utf-8
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
#
# https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate/


import os
import hydra
from pathlib import Path

import numpy as np
import torch

from train import train_vae
from utils.config import process_config
from preprocess import UkbDataset

from datetime import datetime

now = datetime.now()


def adjust_in_shape(config):
    """
    Function to make sure that the output of the encoder is composed of integers 
    In this case : Each block (conv_x + conv_x_a) reduce by 2 the dimension of the volume.
    """

    dims=[]
    for idx in range(1, 4):
        dim = config.in_shape[idx]
        r = dim%(2**config.depth)
        if r!=0:
            dim+=(2**config.depth-r)
        dims.append(dim)
    return((1, dims[0]+4, dims[1], dims[2])) # ! removed dim[0]+4 because it was strange

@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def train_model(config):

    config = process_config(config)

    torch.manual_seed(3) #same seed = same training ? yes, pourtant différents entraînements donnent des outputs diff ?
    # take random seed like contrastive and save it in logs / config ?

    # * Adjusting model to adjust to model speficication
    config.in_shape = adjust_in_shape(config)

    now = datetime.now()
    config.run_time = f"{now:%H-%M-%S}" #Making sure that output dict are matching
    print(f"[INFO] Current working directory : {os.getcwd()}")

    SAVING_PATH = Path(".")

    """ Load data and generate torch datasets """
    dataset = UkbDataset(config)
    
    #### * Splitting the data
    # ! From here the shape of the tensor or the config.in_shape
    train_set, val_set = torch.utils.data.random_split(dataset,
                            [round(0.8*len(dataset)), round(0.2*len(dataset))])

    print("Prepare dataset : DONE !")
    #### * Making the data loader
    trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=config.batch_size,
                  num_workers=1,
                  shuffle=True)
    valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=1,
                num_workers=1,
                shuffle=True)

    val_label = []
    for _,_,path in valloader:
        val_label.append(path[0])
    np.savetxt( SAVING_PATH / "val_label.csv", np.array(val_label), delimiter =", ", fmt ='% s')

    """ Train model for given configuration """
    ### * Training the model
    vae, final_loss_val = train_vae(config, trainloader, valloader)

if __name__ == '__main__':
    train_model()
