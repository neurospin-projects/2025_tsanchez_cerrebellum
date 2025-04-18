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
from omegaconf import OmegaConf

import numpy as np
import torch

from datetime import datetime
now = datetime.now()

from train import train_vae
from utils.config import process_config
from preprocess import UkbDataset

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

    config.save_dir = config.save_dir + f"/{now:%Y-%m-%d}/{now:%H-%M-%S}/"
    config.in_shape = adjust_in_shape(config)

    print(config)

    # create the save dir
    try:
        os.makedirs(config.save_dir)
    except FileExistsError:
        print("Directory " , config.save_dir ,  " already exists")
        pass

    # save config as a yaml file
    with open(config.save_dir+"/config.yaml", "w") as f:
        OmegaConf.save(config, f)

    """ Load data and generate torch datasets """
    # TODO Change this function to have a DataLoader
    # ! The data stored in subset1.df is not reshaped, the reshaping is done in the __getitem__
    dataset = UkbDataset(config)


    #### * Splitting the data
    # ! From here the shape of the tensor or the config.in_shape
    train_set, val_set = torch.utils.data.random_split(dataset,
                            [round(0.8*len(dataset)), round(0.2*len(dataset))])

    #### * Making the data loader
    trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=config.batch_size,
                  num_workers=8,
                  shuffle=True)
    valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=1,
                num_workers=8,
                shuffle=True)

    val_label = []
    for _, path in valloader:
        val_label.append(path[0])
    np.savetxt(f"{config.save_dir}/val_label.csv", np.array(val_label), delimiter =", ", fmt ='% s')

    """ Train model for given configuration """
    ### * Training the model
    vae, final_loss_val = train_vae(config, trainloader, valloader,
                                    root_dir=config.save_dir)


    # """ Evaluate model performances """
    # dico_set_loaders = {'train': trainloader, 'val': valloader}
    #
    # tester = ModelTester(model=vae, dico_set_loaders=dico_set_loaders,
    #                      kl_weight=config.kl, loss_func=criterion,
    #                      n_latent=config.n, depth=3)
    #
    # results = tester.test()
    # encoded = {loader_name:[results[loader_name][k] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
    # df_encoded = pd.DataFrame()
    # df_encoded['latent'] = encoded['train'] + encoded['val']
    # X = np.array(list(df_encoded['latent']))
    #
    # cluster = Cluster(X, save_dir)
    # res = cluster.plot_silhouette()
    # res['loss_val'] = final_loss_val
    #
    # with open(f"{save_dir}results_test.json", "w") as json_file:
    #     json_file.write(json.dumps(res, sort_keys=True, indent=4))

if __name__ == '__main__':
    train_model()
