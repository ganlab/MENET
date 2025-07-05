import torch
from torch import nn
from utils.process_model.split_windows import get_windows_linear
from itertools import accumulate

class VELowerBranchSplitWindowsEmbedding(nn.Module):
    def __init__(self, windows:int, count:list):
        super(VELowerBranchSplitWindowsEmbedding, self).__init__()
        self.input_dim, self.output_dim = get_windows_linear(windows, count)
        self.basenet = [nn.Linear(s, o) for s, o in zip(self.input_dim, self.output_dim)]

    def forward(self, x):
        cumulative_sum = [0] + list(accumulate(self.input_dim))
        temp = []
        for i in range(len(self.basenet)):
           temp.append(self.basenet[i](x[:, :, cumulative_sum[i]:cumulative_sum[i+1]]))
        x = torch.cat(temp, dim=-1)
        return x


class VELowerBranch(nn.Module):
    def __init__(self, size, p, parames):
        super(VELowerBranch, self).__init__()
        wm = parames[0]
        windows = parames[1]
        count = parames[2]
        if wm == 0:
            self.embedding = nn.Linear(size, 4096)
        else:
            self.embedding = VELowerBranchSplitWindowsEmbedding(windows=windows, count=count)
        self.output = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p),
            nn.Linear(4096, 1024),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.output(x)
        return x

if __name__ == '__main__':
    pass