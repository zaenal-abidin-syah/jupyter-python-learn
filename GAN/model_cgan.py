import torch
from torch import nn
from jcopdl.layers import linear_block

class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embed_label = nn.Embedding(n_classes, n_classes)
        self.fc = nn.Sequential(
            linear_block(784 + n_classes, 512, activation='lrelu'),
            linear_block(512, 256, activation='lrelu'),
            linear_block(256, 128, activation='lrelu'),
            linear_block(128, 1, activation='sigmoid'),
        )
    def forward(self, x, y):
        x = self.flatten(x)
        y = self.embed_label(y)
        x = torch.cat([x, y], dim=1)
        return self.fc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, n_classes):
        super().__init__()
        self.embed_label = nn.Embedding(n_classes, n_classes)
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            linear_block(self.z_dim + n_classes, 128, activation='lrelu'),
            linear_block(128, 256, activation='lrelu', batch_norm=True),
            linear_block(256, 512, activation='lrelu', batch_norm=True),
            linear_block(512, 1024, activation='lrelu', batch_norm=True),
            linear_block(1024, 784, activation='tanh')
        )
    def forward(self, x, y):
        y = self.embed_label(y)
        x = torch.cat([x, y], dim=1)
        return self.fc(x)
    def generate(self, labels, device):
        z = torch.randn((len(labels), self.z_dim), device=device)
        return self.forward(z, labels)
