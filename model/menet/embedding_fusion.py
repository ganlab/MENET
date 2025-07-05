import torch
from torch import nn
from model.menet.variation_embedding_lower import VELowerBranch
from model.menet.variation_embedding_upper import VEUpperBranch
from model.menet.genetic_relatedness_embedding import GeneticRelatednessEmbedding


class Scores(nn.Module):
    def __init__(self, p):
        super(Scores, self).__init__()
        self.score = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        return self.score(x)


class Net(nn.Module):
    def __init__(self, ve_size, gr_size, config, params):
        super(Net, self).__init__()
        self.variation_lower = VELowerBranch(ve_size, config['p2'], params)
        self.variation_upper = VEUpperBranch(config['p1'])
        self.weight = nn.Parameter(torch.Tensor([0.5, 0.5]))

        self.gr = GeneticRelatednessEmbedding(gr_size, config['p3'])

        self.varition_output = Scores(config['p4'])

        self.gr_output = Scores(config['p4'])

        self.fusion = nn.Sequential(
            nn.BatchNorm1d(1024 * 2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(config["p4"]),
            nn.Linear(512, 1),
        )

    def forward(self, snp, sim):
        variation_lower_embedding = self.variation_lower(snp)
        variation_upper_embedding = self.variation_upper(snp)
        weight = torch.softmax(self.weight, dim=-1)
        variation_embedding = variation_lower_embedding * weight[0] + variation_upper_embedding * weight[1]

        gr_embedding = self.gr(sim)

        output1 = self.varition_output(variation_embedding)
        output2 = self.gr_output(gr_embedding)

        cat = torch.cat((variation_embedding, gr_embedding), dim=1)
        output3 = self.fusion(cat)

        return output1, output2, output3

if __name__ == '__main__':
    pass