from betaVAE.utils.post_analysis import load_trained_model, load_dataset
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from tqdm import tqdm
import numpy as np

saveable = lambda x : x.cpu().detach().numpy()


def generate_output(data_path : Path, exp_path : Path, to_load :str, device : str, dataset_name : str) : 
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

    results["input"] = np.array(input_s)
    results["id_sub"] = np.array(subject_s)
    results["mu"] = np.array(mu_s)
    results["logvar"] = np.array(logvar_s)
    results["reconstruction"] = np.array(recons_s)

    return results


def compute_emb_only(data_path : Path, exp_path : Path, to_load :str, device : str, dataset_name : str) : 
    results = dict()
    mu_s, logvar_s, subject_s = [], [], []

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
            subject_s.append(subject[0]) # ! batch size == 1
            mu_s.append(saveable(mu))
            logvar_s.append(saveable(logvar))

    results["id_sub"] = np.array(subject_s)
    results["mu"] = np.array(mu_s)

    return results


def generate_emb(data_path : Path, exp_path : Path, to_load :str, device : str, dataset_name : str, only_emb : bool = False) :
    if only_emb : 
        results = compute_emb_only(data_path, exp_path, to_load, device, dataset_name)
    else : 
        results = generate_output(data_path, exp_path, to_load, device, dataset_name) 
    return results

if __name__ == "__main__": 
    EXP_PATH = Path("/neurospin/dico/tsanchez/Jean_zay_runs/08_model_choice/2025-06-14/08-13-28_14878")
    DATA_PATH = Path("/neurospin/dico/tsanchez/preprocessed/UKBio1000")

    results = generate_output(data_path=DATA_PATH,
                        exp_path= EXP_PATH, 
                        to_load= "checkpoint",
                        device="cuda",
                        dataset_name="ukb"
                        )
    print(results)
