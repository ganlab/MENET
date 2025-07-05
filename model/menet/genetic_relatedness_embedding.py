from torch import nn

class GeneticRelatednessEmbedding(nn.Module):
    def __init__(self, genetic_relatedness_size, p):
        super(GeneticRelatednessEmbedding, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(genetic_relatedness_size, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p),
            nn.Linear(4096, 1024),
        )

    def forward(self, x):
        x = self.net(x)
        return x

if __name__ == '__main__':
    pass