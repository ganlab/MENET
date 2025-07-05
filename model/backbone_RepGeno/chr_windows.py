from torch import nn
from itertools import accumulate
import torch
from utils.process_model.split_windows import get_windows_linear

class Chr_Embedding(nn.Module):
    def __init__(self, windows:int, count:list):
        super(Chr_Embedding, self).__init__()
        self.input_dim, self.output_dim = get_windows_linear(windows, count)
        self.basenet = [nn.Linear(s, o) for s, o in zip(self.input_dim, self.output_dim)]

    def forward(self, x):
        cumulative_sum = [0] + list(accumulate(self.input_dim))
        temp = []
        for i in range(len(self.basenet)):
           temp.append(self.basenet[i](x[:, :, cumulative_sum[i]:cumulative_sum[i+1]]))
        x = torch.cat(temp, dim=-1)
        return x


if __name__ == '__main__':
    pass