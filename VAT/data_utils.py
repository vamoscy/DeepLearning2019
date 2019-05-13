import numpy as np
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torchvision import datasets
from torchvision import transforms



def image_loader(path, batch_size):
    transform = transforms.Compose(
        # [transforms.ToTensor()
            [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.Resize(130),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0.1),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
    )
    indices = []
    for i in range(1000):
        for j in range(32):
            indices.append(64*i + 28 + j)
    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform)
    data_loader_sup_train = DataLoader(
        sup_train_data,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=8,
        sampler=sampler.SubsetRandomSampler(indices)
    )
    data_loader_sup_val = DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    data_loader_unsup = DataLoader(
        unsup_data,
        batch_size= batch_size,
        shuffle=True,
        num_workers=8
    )
    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup
