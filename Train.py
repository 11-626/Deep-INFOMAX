#-----------------------------------------------------------------------------
# training script
#-----------------------------------------------------------------------------

import torch, torchvision, torch.nn.functional as F
import argparse
from tqdm import tqdm
from pathlib import Path
import argparse
import math
import copy

from Model import DeepInfoMaxLoss
from Model import PriorDiscriminatorLoss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=False, help='resume from saved models')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    num_epochs = 1000
    num_workers = 4
    save_interval = 100
    version = "cifar10_v3"
    lr_dim = 1e-4
    lr_pdl = 1e-5
    model_dir = Path(f"./Models/{version}/")

    # image size (3,32,32)
    # batch size must be an even number
    # shuffle must be True
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.cifar.CIFAR10("~/.torch/", download=True, transform=transform)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    dim = DeepInfoMaxLoss(alpha=0.5, beta=1.0, gamma=0.1).to(device)
    pdl = PriorDiscriminatorLoss().to(device)

    epoch_start = 1

    if args.resume:
        dim_models = list(model_dir.glob("dim_epoch_*.pt"))
        dim.load_state_dict(torch.load(dim_models[-1]))
        dim.to(device)
        pdl_models = list(model_dir.glob("pdl_epoch_*.pt"))
        pdl.load_state_dict(torch.load(pdl_models[-1]))
        pdl.to(device)

        epoch_start = int(str(dim_models[-1].stem)[10:])

    optimizer_dim = torch.optim.Adam(dim.parameters(), lr=lr_dim)
    optimizer_pdl = torch.optim.Adam(pdl.parameters(), lr=lr_pdl)

    dim.train()
    for epoch in range(epoch_start, num_epochs+1):
        Batch = tqdm(train_loader, total=len(train_dataset) // batch_size)
        for i, (data, target) in enumerate(Batch, 1):
            data = data.to(device)

            # train prior discriminator
            Y, M = dim.encoder(data)
            prior = torch.rand_like(Y)
            discriminator_loss = pdl(Y, prior, device)
            assert not torch.isnan(discriminator_loss)
            assert not torch.isinf(discriminator_loss)
            optimizer_pdl.zero_grad()
            discriminator_loss.backward()
            optimizer_pdl.step()

            # shuffle batch to pair each element with another
            Y, M = dim.encoder(data)
            encoder_loss = pdl.encoder_loss(Y, device)
            M_fake = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            loss = dim(Y, M, M_fake, encoder_loss)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            Batch.set_description(f"[{epoch:>3d}/{num_epochs:<3d}]DIMLoss: {loss.item():.3f}, PDLoss: {discriminator_loss.item():.3f}")
            optimizer_dim.zero_grad()
            loss.backward()
            optimizer_dim.step()

        # checkpoint and save models
        if epoch % save_interval == 0:
            file = model_dir / f"dim_epoch_{epoch:04}.pt"
            file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dim.state_dict(), str(file))
            file = model_dir / f"pdl_epoch_{epoch:04}.pt"
            torch.save(pdl.state_dict(), str(file))
