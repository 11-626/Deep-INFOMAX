#-----------------------------------------------------------------------------
# testing script
#-----------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

import torch, torchvision, torch.nn.functional as F
import argparse
from tqdm import tqdm
from pathlib import Path

from Model import DeepInfoMaxLoss

def corr_hist(Y):
    df = pd.DataFrame(Y.numpy())
    corr_values = df.corr().values.flatten()
    pd.DataFrame(corr_values).hist()

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    num_workers = 1
    model_path = "./Models/cifar10_v3/dim_epoch_0100.pt"

    #-- dataset and DataLoader
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.cifar.CIFAR10("~/.torch/", download=True, transform=transform)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    dim = DeepInfoMaxLoss(alpha=0.5, beta=1.0, gamma=0.1).to(device)
    dim.load_state_dict( torch.load(model_path) )
    dim.eval()

    Y_list = []

    with torch.no_grad():
        Batch = tqdm(train_loader, total=len(train_dataset) // batch_size)
        for i, (data, target) in enumerate(Batch, 1):
            data = data.to(device)
            Y, M = dim.encoder(data)
            Y_list.append(Y.detach().cpu())
    
    Y = torch.cat(Y_list, dim=0)
    prior = torch.rand_like(Y)

    print(f"Y     mean={Y.mean():}, std={Y.std():}")
    print(f"Prior mean={prior.mean():}, std={prior.std():}")

    # corr_hist(Y)
    # corr_hist(prior)