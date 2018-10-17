#-----------------------------------------------------------------------------
# training script
#-----------------------------------------------------------------------------

import torch, torchvision, torch.nn.functional as F
import argparse
from tqdm import tqdm
from pathlib import Path

from Model import DeepInfoMaxLoss

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    num_epochs = 1000
    num_workers = 4
    save_interval = 100
    version = "cifar10_v2"
    lr = 1e-4

    # image size (3,32,32)
    # batch size must be an even number
    # shuffle must be True
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    train_dataset = torchvision.datasets.cifar.CIFAR10("~/.torch/", download=True, transform=transform)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    dim = DeepInfoMaxLoss(alpha=0.5, beta=1.0, gamma=0.1).to(device)
    optimizer = torch.optim.Adam(dim.parameters(), lr=lr)

    dim.train()
    for epoch in range(1, num_epochs+1):
        Batch = tqdm(train_loader, total=len(train_dataset) // batch_size)
        for i, (data, target) in enumerate(Batch, 1):
            data = data.to(device)

            Y, M = dim.encoder(data)
            # shuffle batch to pair each element with another
            M_fake = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            loss = dim(Y, M, M_fake)
            Batch.set_description(f"[{epoch:>3d}/{num_epochs:<3d}]Loss/Train: {loss.item():1.5e}")
            dim.zero_grad()
            loss.backward()
            optimizer.step()

        # checkpoint and save models
        if epoch % save_interval == 0:
            file = Path(f"./Models/{version}/checkpoint_epoch_{epoch}.pkl")
            file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dim.state_dict(), str(file))
