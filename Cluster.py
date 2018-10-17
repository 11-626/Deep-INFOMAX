#-----------------------------------------------------------------------------
# test clustering
#-----------------------------------------------------------------------------

import torch, torchvision, torch.nn.functional as F
from Model import DeepInfoMaxLoss

import numpy as np
from skimage.io import imsave

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    num_workers = 1
    label = 5
    model_path = "./Models/cifar10/checkpoint_epoch_100.pkl"

    #-- dataset and DataLoader
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    train_dataset = torchvision.datasets.cifar.CIFAR10("~/.torch/", download=True, transform=transform)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    dim = DeepInfoMaxLoss(alpha=0.5, beta=1.0, gamma=0.1).to(device)
    dim.load_state_dict( torch.load(model_path) )
    dim.eval()

    _t = -1
    for x, t in train_loader:
        #x, t = x.to(device), t.to(device)
        if t.item() == label:
            break
    key, f = dim.encoder(x.to(device))

    distance = []
    features = []
    truth = []
    image = []

    with torch.no_grad():
        for i, (data, target) in enumerate(train_loader, 1):
            if i < 100 :
                continue
            elif i > 200 :
                break

            data = data.to(device)
            Y, M = dim.encoder(data)

            dist = F.l1_loss(Y,key.detach()).item()
            true = target.item()
            pic  = data.squeeze(0).cpu()

            distance.append(dist)
            truth.append(true)
            image.append(pic)

    idx = sorted(range(len(distance)), key=distance.__getitem__)
    #for i in idx[:10]:
    #    print(distance[i], truth[i])
    #print(f"original label: {label}")

    img_origin = x.squeeze(0).cpu().numpy().transpose(1,2,0)
    top_row = np.concatenate([img_origin,] + [np.ones_like(img_origin) for i in range(10-1)], axis=1)

    middle_row = np.concatenate([image[i].numpy().transpose(1,2,0) for i in idx[:10]], axis=1)
    bottom_row = np.concatenate([image[i].numpy().transpose(1,2,0) for i in idx[-10:]], axis=1)

    _img = np.concatenate((top_row, middle_row, bottom_row), axis=0)
    imsave("./Logs/cifar10/sample.png", _img)
