# -*- coding: utf-8 -*-
# /usr/bin/env python3
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

import numpy as np
import pandas as pd
import torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from beta_vae import *
from utils.pytorchtools import EarlyStopping

from postprocess import plot_loss


def train_vae(config, trainloader, valloader, root_dir=None):
    """ Trains beta-VAE for a given hyperparameter configuration
    Args:
        config: instance of class Config
        trainloader: torch loader of training data
        valloader: torch loader of validation data
        root_dir: str, directory where to save model
    Returns:
        vae: trained model
        final_loss_val
    """
    torch.manual_seed(5)
    writer = SummaryWriter(log_dir= config.save_dir+'logs/',
                            comment="")

    lr = config.lr
    vae = VAE(config.in_shape, config.n, depth=3)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    vae.to(device)
    summary(vae, list(config.in_shape))

    weights = [1, 2]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    nb_epoch = config.nb_epoch
    early_stopping = EarlyStopping(patience=12, verbose=True, root_dir=root_dir)

    list_loss_train, list_loss_val = [], []

    # arrays enabling to see model reconstructions
    id_arr, phase_arr, input_arr, output_arr = [], [], [], []

    for epoch in range(config.nb_epoch):
        running_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        epoch_steps = 0
        for inputs, path in trainloader:
            optimizer.zero_grad()

            inputs = Variable(inputs).to(device, dtype=torch.float32)
            target = torch.squeeze(inputs, dim=1).long()
            output, z, logvar = vae(inputs)
            partial_recon_loss, partial_kl, loss = vae_loss(output, target, z,
                                    logvar, criterion,
                                    kl_weight=config.kl)
            output = torch.argmax(output, dim=1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            recon_loss += partial_recon_loss
            kl_loss += partial_kl
            epoch_steps += 1
        running_loss = running_loss / epoch_steps
        recon_loss = recon_loss / epoch_steps
        kl_loss = kl_loss / epoch_steps

        images = [inputs[0][0][10][:][:], output[0][10][:][:]]
        grid = torchvision.utils.make_grid(images)
        writer.add_image('inputs', images[0].unsqueeze(0), epoch)
        writer.add_image('output', images[1].unsqueeze(0), epoch)
        writer.add_scalar('Loss/train', running_loss, epoch)
        writer.add_scalar('KL Loss/train', kl_loss, epoch)
        writer.add_scalar('recon Loss/train', recon_loss, epoch)
        writer.close()

        print("[%d] KL loss: %.2e" % (epoch + 1, kl_loss))
        print("[%d] recon loss: %.2e" % (epoch + 1, recon_loss))
        #print(kl_loss * config.kl + recon_loss)
        print("[%d] loss: %.2e" % (epoch + 1,
                                        running_loss))
        list_loss_train.append(running_loss)
        running_loss = 0.0

        """ Saving of reconstructions for visualization in Anatomist software """
        if epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('train')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

        # Validation loss
        val_loss = 0.0
        recon_loss_val = 0.0
        kl_val = 0.0
        val_steps = 0
        total = 0
        vae.eval()
        for inputs, path in valloader:
            with torch.no_grad():
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, z, logvar = vae(inputs)
                target = torch.squeeze(inputs, dim=1).long()
                partial_recon_loss_val, partial_kl_val, loss = vae_loss(output, target,
                                        z, logvar, criterion,
                                        kl_weight=config.kl)
                output = torch.argmax(output, dim=1)

                val_loss += loss.cpu().numpy()
                recon_loss_val += partial_recon_loss_val
                kl_val += partial_kl_val
                val_steps += 1
        valid_loss = val_loss / val_steps
        recon_loss_val = recon_loss_val / val_steps
        kl_val = kl_val / val_steps

        images = [inputs[0][0][10][:][:],\
                  output[0][10][:][:]]
        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_scalar('KL Loss/val', kl_val, epoch)
        writer.add_scalar('recon Loss/val', recon_loss_val, epoch)
        writer.add_image('inputs VAL', images[0].unsqueeze(0), epoch)
        writer.add_image('output VAL', images[1].unsqueeze(0), epoch)
        writer.close()

        # prints on the terminal
        print("[%d] KL validation loss: %.2e" % (epoch + 1, kl_val))
        print("[%d] recon validation loss: %.2e" % (epoch + 1, recon_loss_val))
        #print(kl_val * config.kl + recon_loss_val)
        print("[%d] validation loss: %.2e" % (epoch + 1, valid_loss))
        list_loss_val.append(valid_loss)

        early_stopping(valid_loss, vae)
        print("")

        """ Saving of reconstructions for visualization in Anatomist software """
        if early_stopping.early_stop or epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('val')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())
            break
    for key, array in {'input': input_arr, 'output' : output_arr,
                           'phase': phase_arr, 'id': id_arr}.items():
        np.save(config.save_dir+key, np.array([array]))

    plot_loss(list_loss_train[1:], config.save_dir+'tot_train_')
    plot_loss(list_loss_val[1:], config.save_dir+'tot_val_')
    final_loss_val = list_loss_val[-1:]

    """Saving of trained model"""
    torch.save((vae.state_dict(), optimizer.state_dict()),
                config.save_dir + 'vae.pt')

    print("Finished Training")
    return vae, final_loss_val
