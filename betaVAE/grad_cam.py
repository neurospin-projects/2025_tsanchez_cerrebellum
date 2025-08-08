from pathlib import Path
from betaVAE.utils.post_analysis import load_trained_model, load_dataset, read_config_exp, adjust_in_shape
from betaVAE.preprocess import Padding
import torch
from torch.autograd import Variable
from soma import aims
import numpy as np
import torch.nn.functional as F
from typing import Tuple, List, Dict

def only_volume(input_np : np.ndarray) -> np.ndarray : 
    """Transforms volume with more than 3 dimensions the volume of shape : [x,y,z]

    Args:
        input_np (np.ndarray): input_volume with excedent dims

    Returns:
        np.ndarray: volume with shape [x,y,z]
    """
    # Formatting the volume to fit right shape
    num_squeeze = input_np.ndim - 3 #Keeping 3 dimensions for the 3d volume
    for _ in range(num_squeeze): 
        input_np = np.squeeze(input_np)
    return input_np

def normal_difference_distribution(mu_x : torch.Tensor,
                                   mu_y : torch.Tensor,
                                   logvar_y : torch.Tensor,
                                   logvar_x : torch.Tensor) -> torch.Tensor:

    """Sample a latent from the normal difference distribution between two distributions

    Returns:
        torch.Tensor: Resampled tensor from normal difference distribution
    """
    mu_u = mu_x - mu_y
    logvar_u = logvar_x - logvar_y

    stddev = torch.exp(0.5 * logvar_u)
    noise = torch.randn_like(logvar_u)

    return (noise * stddev) + mu_u


class GradCam : 
    def __init__(self,
                 path_data : Path | str,
                 path_exp : Path | str, 
                 model_to_load : str,
                 dataset_name : str,
                 device : str,
                 ):
        # Loading data and model
        self.device = device
        self.path_data = path_data
        self.path_exp = path_exp
        
        self._model = load_trained_model(self.path_exp, to_load=model_to_load, device = self.device)
        self._model.to(device = self.device)
        self._dataset = load_dataset(self.path_data, self.path_exp, dataset_name)

        config_exp = read_config_exp(self.path_exp)
        self._input_shape = adjust_in_shape(config_exp)
        self._latent_dim = config_exp.n
    

    def save_numpy(self, np_vol : np.ndarray, saving_path : Path | str) -> str :
        assert str(saving_path).endswith(".nii.gz"), "Only saving to nii.gz"

        np_vol = only_volume(np_vol)
        aims_vol = aims.Volume(np_vol)
        aims.write(aims_vol, str(saving_path))

        return f"Saved at {saving_path}"
    
    def get_sub_tensor(self, sub_id : str) -> Variable:
        # Numpy to tensor and retrieving white matter only
        sub_index = self._dataset.get_index(sub_id)
        both_split, full, subject = self._dataset[sub_index]

        assert subject == sub_id, f"Failed retrieving, got {subject} instead of {sub_id}"

        wm_inp = both_split[0,:,:,:].unsqueeze(0).unsqueeze(0) #Retrieving white matter
        return Variable(wm_inp).to(device = self.device, dtype = torch.float32)
    
    def _forward(self, input_vol : Variable) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._model.zero_grad()
        mu, logvar, feat_map = self._model.encode(input_vol, return_feat = True)
        feat_map.retain_grad()
        return mu, logvar, feat_map

    def backward_feat_map(self, sub_id : str, latent_ind : int) -> torch.Tensor:
        assert sub_id in self._dataset.list_subjects.values, "Subject doesn't exist in dataset"

        input_vol = self.get_sub_tensor(sub_id)
        mu, logvar, feat_map = self._forward(input_vol)

        z = self._model.sample_z(mu, logvar)
        z_i = z[0,latent_ind]

        z_i.backward()
        return feat_map
    
    def compute_cam_variable(self, sub_id : str, latent_ind : int) -> np.ndarray : 
        feat_map = self.backward_feat_map(sub_id, latent_ind)
        grad = feat_map.grad #Shape = [1,nb_channel,x_feat, y_feat, z_feat]
        weights = grad.mean(dim=(2,3,4), keepdim = True) #shape = [1,nb_channel,1,1,1]
        cam = F.relu((weights*feat_map).sum(dim=1, keepdim=True)) #shape = [1,1,x_feat, y_feat, z_feat]

        # Normalisation
        cam /= cam.max()

        #Interpolation to reach input dims
        cam_interpolation = F.interpolate(cam, size = self._input_shape[1:], mode="trilinear", align_corners=False)

        return only_volume(cam_interpolation.detach().cpu().numpy())

    def compute_cam_list(self, sub_id : str, latent_list : List[int] | np.array) -> np.ndarray :
        list_cams = []
        for latent_ind in latent_list :
            assert latent_ind < self._latent_dim, f"{latent_ind} is not a latent variable index, should be inferior to {self._latent_dim}"
            list_cams.append(self.compute_cam_variable(sub_id, latent_ind))
        return np.array(list_cams)

    def compute_average_cam(self, sub_id : str) : 
        all_ind = np.arange(self._latent_dim)
        return np.nanmean(self.compute_cam_list(sub_id, latent_list=all_ind), axis=0)


    def backward_normal_diff_dist(self, sub_id : str,
                                         mu_x : torch.Tensor,
                                         logvar_x : torch.Tensor,
                                         latent_ind : int) :
        input_vol = self.get_sub_tensor(sub_id)
        mu_y, logvar_y, feat_map = self._forward(input_vol)
        z = normal_difference_distribution(mu_x, mu_y, logvar_x, logvar_y)
        z_i = z[0,latent_ind]
        z_i.backward()
        return feat_map

    def compute_cam_diff_dist(self, sub_id : str,
                             latent_ind : int,
                             mu_x : torch.Tensor,
                             logvar_x : torch.Tensor) -> np.ndarray : 

        feat_map = self.backward_normal_diff_dist(sub_id, mu_x, logvar_x, latent_ind)
        grad = feat_map.grad #Shape = [1,nb_channel,x_feat, y_feat, z_feat]
        weights = grad.mean(dim=(2,3,4), keepdim = True) #shape = [1,nb_channel,1,1,1]
        cam = F.relu((weights*feat_map).sum(dim=1, keepdim=True)) #shape = [1,1,x_feat, y_feat, z_feat]

        # Normalisation
        cam /= cam.max()

        #Interpolation to reach input dims
        cam_interpolation = F.interpolate(cam, size = self._input_shape[1:], mode="trilinear", align_corners=False)

        return only_volume(cam_interpolation.detach().cpu().numpy())

    def anomaly_cam_list(self, sub_id : str, 
                         latent_list : List[int] | np.array,
                         mu_x : torch.Tensor, 
                         logvar_x : torch.Tensor,) -> np.ndarray :
        list_cams = []
        for latent_ind in latent_list :
            assert latent_ind < self._latent_dim, f"{latent_ind} is not a latent variable index, should be inferior to {self._latent_dim}"

            list_cams.append(self.compute_cam_diff_dist(sub_id=sub_id,
                                                        latent_ind=latent_ind,
                                                        mu_x=mu_x,
                                                        logvar_x=logvar_x))
        return np.array(list_cams)

    def average_anomaly_cam(self, sub_id : str, mu_x : torch.Tensor, logvar_x : torch.Tensor) : 

        all_ind = np.arange(self._latent_dim)
        return np.nanmean(self.anomaly_cam_list(sub_id,
                                     latent_list=all_ind,
                                     mu_x=mu_x,
                                     logvar_x=logvar_x),axis=0)

    def save_original(self, sub_id : str , saving_path : Path): 
        path_t1w = self.path_data / sub_id / "cropped" / "vermis" / f"{sub_id}_crop_t1mri_vermis.nii.gz"
        padder = Padding(self._input_shape[1:], nb_channels=1, fill_value=0)
        aims_t1w = aims.read(str(path_t1w))
        t1w_padded = padder(aims_t1w.np[:,:,:,0])

        t1w_padded_aims = aims.Volume(t1w_padded)
        aims.write(t1w_padded_aims, str(saving_path))