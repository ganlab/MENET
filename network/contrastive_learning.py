import torch
from torch import nn

class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size):
        super(Backbone, self).__init__()
        self.resnet = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=in_channels, kernel_size=kernel_size,  padding=kernel_size // 2, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )
        self.max_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
    def forward(self, x):
        identity = self.shortcut(x)
        x = self.resnet(x)
        x = x + identity
        x = self.max_pool(x)
        return x

class TraitSpecificEncoderForRepGeno(nn.Module):
    def __init__(self, snp_size, stride, out_dim):
        super(TraitSpecificEncoderForRepGeno, self).__init__()
        self.backbone = nn.Sequential(
            Backbone(in_channels=1, out_channels=8, kernel_size=5, stride=stride, pool_size=stride),
            Backbone(in_channels=8, out_channels=16, kernel_size=5, stride=stride, pool_size=stride),
            Backbone(in_channels=16, out_channels=32, kernel_size=5, stride=stride, pool_size=stride),
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(in_features=self.backbone(torch.ones([32, 1, snp_size])).shape[-1], out_features=out_dim),
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output


if __name__ == '__main__':
    pass