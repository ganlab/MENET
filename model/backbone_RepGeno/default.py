from torch import nn

class Embedding(nn.Module):
    def __init__(self, size):
        super(Embedding, self).__init__()
        self.basenet = nn.Sequential(
            nn.Linear(size, 8192),
        )
    def forward(self, x):
        x = self.basenet(x)
        return x


if __name__ == '__main__':
    pass
