from torch import nn
from model.backbone_RepGeno.default import Embedding
from model.backbone_RepGeno.chr_windows import Chr_Embedding


class Net(nn.Module):
    def __init__(self, size, parameters):
        super(Net, self).__init__()
        p = parameters[0]
        windows_mechanism = parameters[1]
        window_chr = parameters[2]
        count = parameters[3]
        if windows_mechanism == 0:
            self.embedding = Embedding(size)
        elif windows_mechanism == 1:
            self.embedding = Chr_Embedding(window_chr, count)

        self.network = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p),
            nn.Linear(8192, 2048),
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p),
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.network(x)
        return x

if __name__ == '__main__':
    pass
