import hydra
import torch

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
    return((1, dims[0]+4, dims[1], dims[2]))

@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def gen_loader(config):

    config = process_config(config)

    torch.manual_seed(3) #same seed = same training ? yes, pourtant différents entraînements donnent des outputs diff ?
    # take random seed like contrastive and save it in logs / config ?

    config.in_shape = adjust_in_shape(config)

    print(config)

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

if __name__ == "__main__" :
    gen_loader()