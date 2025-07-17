from betaVAE.beta_vae import VAE
from betaVAE.preprocess import UkbDataset, AtaxiaDataset
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from tqdm import tqdm

EXP_PATH = Path("")
DATA_ROOT = Path("")

saveable = lambda x : x.cpu().detach().numpy()


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
        case _ : 
            raise ValueError("Dataset available : ukb, ataxia")
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

def generate_emb(data_path : Path, exp_path : Path, to_load :str, device : str, dataset_name : str) : 
    results = dict()
    mu_s, logvar_s, recons_s, input_s, subject_s = [], [], [], [], []

    model = load_trained_model(exp_path=exp_path, to_load= to_load, device=device)
    model.to(device)

    dataset = load_dataset(data_path=data_path, exp_path=exp_path, dataset_name=dataset_name)
    loader = DataLoader(dataset=dataset, batch_size=1)

    model.eval()
    for split, full_vol, subject in tqdm(loader) : 
        with torch.no_grad():
            wm_vol = split[:,0,:,:,:].unsqueeze(1)
            inputs = Variable(wm_vol).to(device, dtype = torch.float32)
            logits, mu, logvar = model(inputs)
            reconstruction = torch.argmax(logits, dim=1)
            subject_s.append(subject[0]) # ! batch size == 1
            input_s.append(saveable(inputs.squeeze(0).squeeze(0))) 
            mu_s.append(saveable(mu))
            logvar_s.append(saveable(logvar))
            recons_s.append(saveable(reconstruction.squeeze(0)))

    results["input"] = input_s
    results["id_sub"] = subject_s
    results["mu"] = mu_s
    results["logvar"] = logvar_s
    results["reconstruction"] = recons_s

    return results

if __name__ == "__main__": 
    EXP_PATH = Path("/neurospin/dico/tsanchez/Jean_zay_runs/08_model_choice/2025-06-14/08-13-28_14878")
    DATA_PATH = Path("/neurospin/dico/tsanchez/preprocessed/UKBio1000")

    results = generate_emb(data_path=DATA_PATH,
                        exp_path= EXP_PATH, 
                        to_load= "checkpoint",
                        device="cuda",
                        dataset_name="ukb"
                        )
    print(results)
