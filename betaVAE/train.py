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
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from pathlib import Path

from beta_vae import *
from utils.pytorchtools import EarlyStopping

CHECK_OUTPUT_NP = 10

def retrieve_counts(torch_unique):
    # torch.Tensor.unique(return_counts= True)
    values = torch_unique[0].tolist()
    counts = torch_unique[1].tolist()
    tmp_dict = dict(zip(values,counts))
    dict_count = dict()
    for i in [0,1,2] : # Specific to the cerebellym case (values are in this list)
        if i in tmp_dict.keys():
            dict_count[i] = tmp_dict[i]
        else :
            dict_count[i] = 0
    return dict_count



def train_vae(config, trainloader, valloader):
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
    SAVING_PATH = Path(os.getcwd())
    LOGDIR = SAVING_PATH / "logs"
    

    writer = SummaryWriter(log_dir=LOGDIR,
                            comment="")

    print(f"Saving Tensorboard : {LOGDIR}")

    # * Retrieving settings
    lr = config.lr

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    vae = VAE(config)
    vae.to(device)
    
    weights = config.weights
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    # criterion = nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=config.weight_decay)

    nb_epoch = config.nb_epoch
    early_stopping = EarlyStopping(patience=35, verbose=True, root_dir=SAVING_PATH)

    list_loss_train, list_loss_val = [], []

    # arrays enabling to see model reconstructions
    id_arr, phase_arr, input_arr, output_arr, epoch_arr, wm_count, emp_count, sulci_count= [], [], [], [], [], [], [], []

    for epoch in range(config.nb_epoch):
        running_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        epoch_steps = 0
        for both_split, full, path in trainloader:
            optimizer.zero_grad()

            inputs = both_split[:,0,:,:,:].unsqueeze(1)
            inputs = Variable(inputs).to(device, dtype=torch.float32)

            # ! Cross entropy doesn't take negative values so added 1 to each class
            target = torch.squeeze(inputs, dim=1).long()
            logits, z, logvar = vae(inputs)
            partial_recon_loss, partial_kl, loss = vae_loss(logits, target, z,
                                    logvar, criterion,
                                    kl_weight=config.kl,)
            output = torch.argmax(logits, dim=1)
            loss.backward()

            nn.utils.clip_grad_norm_(vae.parameters(), max_norm = config.gradient_max_norm)
            optimizer.step()

            running_loss += loss.item()
            recon_loss += partial_recon_loss
            kl_loss += partial_kl
            epoch_steps += 1


        if (epoch % 1 == 0) : 
            for name, param in vae.named_parameters():
                if param is not None:
                    writer.add_histogram(f"grad/{name}", param.grad, global_step=epoch)
                    # print(f" [GRAD] {name} : {param.grad}")
        writer.add_histogram(f"logits/train", logits, global_step=epoch)

        probs = torch.softmax(logits, dim=1)
        writer.add_histogram("probs_emp/train", probs[:,0,:,:,:], global_step=epoch)
        writer.add_histogram("probs_wm/train", probs[:,1,:,:,:], global_step=epoch)

        running_loss = running_loss / epoch_steps
        recon_loss = recon_loss / epoch_steps
        kl_loss = kl_loss / epoch_steps


        # Retrieving counts of the output
        counts_output = retrieve_counts(output.unique(return_counts=True))

        writer.add_scalar('Count_empt/train', counts_output[0], epoch)
        writer.add_scalar('Count_wm/train', counts_output[1], epoch)

        # Writing loss
        writer.add_scalar('Loss/train', running_loss, epoch)
        writer.add_scalar('KL Loss/train', kl_loss, epoch)
        writer.add_scalar('recon Loss/train', recon_loss, epoch)
        writer.close()

        print(f"[TRAIN] Values in output : {list(counts_output.keys())}")
        print(f"[TRAIN] Counts in output : {list(counts_output.values())}")
        print("[%d] KL loss: %.2e" % (epoch + 1, kl_loss))
        print("[%d] recon loss: %.2e" % (epoch + 1, recon_loss))
        print("[%d] loss: %.2e" % (epoch + 1, running_loss))

        list_loss_train.append(running_loss)
        running_loss = 0.0

        if (epoch%CHECK_OUTPUT_NP) == 0:
            id_arr.append(path[0])
            phase_arr.append('train')
            input_arr.append(np.array(np.squeeze(inputs[0]).cpu().detach().numpy()))
            output_arr.append(np.squeeze(output[0]).cpu().detach().numpy())
            epoch_arr.append(epoch)

        # Validation loss
        val_loss = 0.0
        recon_loss_val = 0.0
        kl_val = 0.0
        val_steps = 0
        vae.eval()

        for both_split, full, path in valloader:
            with torch.no_grad():
                inputs = both_split[:,0,:,:,:].unsqueeze(1)
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                logits, z, logvar = vae(inputs)
            # ! Cross entropy doesn't take negative values so added 1 to each class
                target = torch.squeeze(inputs, dim=1).long()
                partial_recon_loss_val, partial_kl_val, loss = vae_loss(logits, target,
                                        z, logvar, criterion,
                                        kl_weight=config.kl,)
                output = torch.argmax(logits, dim=1)
                val_loss += loss.cpu().numpy()
                recon_loss_val += partial_recon_loss_val
                kl_val += partial_kl_val
                val_steps += 1

        valid_loss = val_loss / val_steps
        recon_loss_val = recon_loss_val / val_steps
        kl_val = kl_val / val_steps


        counts_output = retrieve_counts(output.unique(return_counts=True))

        writer.add_histogram(f"logits/val", logits, global_step=epoch)

        probs = torch.softmax(logits, dim=1)
        writer.add_histogram("probs_emp/val", probs[:,0,:,:,:], global_step=epoch)
        writer.add_histogram("probs_wm/val", probs[:,1,:,:,:], global_step= epoch)

        writer.add_scalar('Count_empt/val', counts_output[0], epoch)
        writer.add_scalar('Count_wm/val', counts_output[1], epoch)

        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_scalar('KL Loss/val', kl_val, epoch)
        writer.add_scalar('recon Loss/val', recon_loss_val, epoch)

        writer.close()

        # prints on the terminal
        print(f"[VAL] Values in output : {list(counts_output.keys())}")
        print(f"[VAL] Counts in output : {list(counts_output.values())}")
        print("[%d] KL validation loss: %.2e" % (epoch + 1, kl_val))
        print("[%d] recon validation loss: %.2e" % (epoch + 1, recon_loss_val))
        print("[%d] validation loss: %.2e" % (epoch + 1, valid_loss))

        list_loss_val.append(valid_loss)

        # * Checking if early stopping needed
        early_stopping(valid_loss, vae)

        print("\n __________________ \n")

        """ Saving of reconstructions for visualization in Anatomist software """
        if (epoch%CHECK_OUTPUT_NP) == 0:
            id_arr.append(path[0])
            phase_arr.append('val')
            input_arr.append(np.array(np.squeeze(inputs[0]).cpu().detach().numpy()))
            output_arr.append(np.squeeze(output[0]).cpu().detach().numpy())
            epoch_arr.append(epoch)
            wm_count.append(counts_output[0])
            emp_count.append(counts_output[1])
            sulci_count.append(counts_output[2])
            # if custom_metric != 0 and epoch%20==0 :
            #     torch.save((vae.state_dict(), optimizer.state_dict()),
                        # SAVING_PATH / f'training_{epoch}.pt')

            for key, array in {'input': input_arr,
                                'output' : output_arr,
                                'phase': phase_arr,
                                'id': id_arr,
                                'epoch_val' : epoch_arr,
                                'emp_count' : wm_count,
                                'wm_count' : emp_count,
                                }.items():
                np.save(SAVING_PATH / key, np.array([array]))

        if early_stopping.early_stop or epoch == nb_epoch -1:
            break


    final_loss_val = list_loss_val[-1:]

    """Saving of trained model"""
    torch.save((vae.state_dict(), optimizer.state_dict()),
                SAVING_PATH / 'vae.pt')

    print("Finished Training")
    return vae, final_loss_val
