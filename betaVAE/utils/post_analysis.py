from betaVAE.beta_vae import VAE
from betaVAE.preprocess import UkbDataset, AtaxiaDataset, SAF1T5Dataset
from pathlib import Path
from omegaconf import OmegaConf
import torch

def adjust_in_shape(config):
    """
    Function to make sure that the output of the encoder is composed of integers 
    In this case : Each block (conv_x + conv_x_a) reduce by 2 the dimension of the volume.
    """

    dims=[]
    for idx in range(1, 4):
        dim = config.in_shape[idx]
        r = dim%(2**config.encoder_depth)
        if r!=0:
            dim+=(2**config.encoder_depth-r)
        dims.append(dim)
    return((1, dims[0]+(2**(config.encoder_depth - 1)), dims[1], dims[2])) # ! removed dim[0]+4 because it was strange

def read_config_exp(path : Path): 
    with open(path / ".hydra" / "config.yaml") as f:
        config = OmegaConf.load(f)
    return config

def load_model(path : Path, device : str):
    config = read_config_exp(path)
    config.in_shape = adjust_in_shape(config)
    return VAE(config, device = device)

def load_dataset(data_path : Path, exp_path : Path, dataset_name : str): 
    config = read_config_exp(exp_path)
    config.data_root = data_path
    config.in_shape = adjust_in_shape(config)
    match dataset_name : 
        case  "ukb" : 
            config_dataset = UkbDataset
        case "ataxia" : 
            config_dataset = AtaxiaDataset
        case "saf1t5" : 
            config_dataset = SAF1T5Dataset
        case _ : 
            raise ValueError("Dataset available : ukb, ataxia, saf1t5")
    return config_dataset(config)

def load_trained_model(exp_path : Path, to_load : str, device : str):
    model = load_model(path=exp_path, device = device)

    match to_load : 
        case "vae" : #No optimizer state dict in the .pt file
            model_state_dict, optimizer_state_dict = torch.load(exp_path / "vae.pt")
            model.load_state_dict(model_state_dict)

        case "checkpoint" : # w/ optimizer state dict in the .pt file
            model_state_dict = torch.load(exp_path / "checkpoint.pt")
            model.load_state_dict(model_state_dict)

        case _ :
            raise ValueError("Only [checkpoint | vae] available")
    return model
