import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

def get_mnist(train_size=3000, batch_size=1024, noise_rate=0.0):
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(".", train=False, download=True, transform=transform)

    # limit training set
    subset = Subset(full_train, range(train_size))

    # add label noise
    if noise_rate > 0:
        for _, label in subset:
            if random.random() < noise_rate:
                label = random.randint(0, 9)

    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=256, shuffle=False)

    return train_loader, test_loader
