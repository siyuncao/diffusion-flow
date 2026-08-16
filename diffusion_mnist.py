import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([ # Compose is added first in order to chain the latter (ToTensor, Normalize) together in sequence
    transforms.ToTensor(), # The MNIST file on disk stores 784 bytes per digit. 
                           # ToTensor() reads those bytes into a PyTorch tensor and divides by 255, mapping everything into [0, 1].
    transforms.Normalize((0.5,), (0.5,)) # maps [0, 1] → [-1, 1]
]))

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
