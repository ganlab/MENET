import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(ResNet, self).__init__()
        self.resnet = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=in_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
        self.max_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.resnet(x)
        x = x + identity
        x = self.max_pool(x)
        return x


class VE(nn.Module):
    def __init__(self, snp_size, config):
        super(VE, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv1d(1, config["conv"][0],
                      kernel_size=config["conv"][1], padding=config["conv"][1]//2,
                      stride=config["conv"][2]),
            ResNet(config["conv"][0], config["res1"][0], config["res1"][1], config["res1"][2]),
            ResNet(config["res1"][0], config["res2"][0], config["res2"][1], config["res2"][2])
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(config["adaptive"])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear((snp_size+config["conv"][2]-(1 if config["conv"][1]%2else 0))//(config["conv"][2]*config["res1"][2]*config["res2"][2])*config["adaptive"],
                      config["embedding_dim"][0]),
            nn.BatchNorm1d(config["embedding_dim"][0]),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["embedding_dim"][0], config["embedding_dim"][1])
        )

    def forward(self, x):
        x = self.resnet(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.adaptive_pool(x)
        x = self.linear(x)
        return x


class RepGeno(nn.Module):
    def __init__(self, rep_size, config):
        super(RepGeno, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(rep_size, config["embedding_dim"][0]),
            nn.BatchNorm1d(config["embedding_dim"][0]),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["embedding_dim"][0], config["embedding_dim"][1]),
        )
    def forward(self, x):
        x = self.linear(x)
        return x


class CrossInformationFeatureFusion(nn.Module):
    def __init__(self, dim):
        super(CrossInformationFeatureFusion, self).__init__()
        self.field = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.BatchNorm1d(4 * dim),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(0.6),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        x = self.field(x)
        return x



class MultiChrInformationFeatureFusion(nn.Module):
    def __init__(self, windows, dim):
        super(MultiChrInformationFeatureFusion, self).__init__()
        self.windows = windows
        self.dim = dim
        self.f_ve = [CrossInformationFeatureFusion(dim) for _ in range(len(windows))]
        self.f_repGeno = [CrossInformationFeatureFusion(dim) for _ in range(len(windows))]
        self.transfer_ve = nn.Linear(dim * len(windows), dim)
        self.transfer_RepGeno = nn.Linear(dim * len(windows), dim)

    def forward(self, x, y):
        y_all = []
        x_all = []
        for i in range(len(self.windows)):
            flow_field = torch.cat([x[i], y], dim=1)
            x_all.append(self.f_ve[i](flow_field) + x[i])
            y_all.append(self.f_repGeno[i](flow_field) + y)
        x_all = torch.cat(x_all, dim=1)
        y_all = torch.cat(y_all, dim=1)
        x_f = self.transfer_ve(x_all)
        y_f = self.transfer_RepGeno(y_all)
        return x_f, y_f




class Fusion(nn.Module):
    def __init__(self, config, windows=None):
        super(Fusion, self).__init__()
        dim = config["VE_and_RepGeno_embedding_dim"]
        self.windows = windows
        if self.windows:
            self.f_ve_RepGeno = MultiChrInformationFeatureFusion(self.windows, dim)
        else:
            self.f_ve = CrossInformationFeatureFusion(dim)
            self.f_repGeno = CrossInformationFeatureFusion(dim)
        self.output = nn.Sequential(
            nn.Linear(2 * dim, config["embedding_dim"]),
            nn.BatchNorm1d(config["embedding_dim"]),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["embedding_dim"], 1),
        )

    def forward(self, x, y):
        if self.windows:
            x_f, y_f = self.f_ve_RepGeno(x, y)
        else:
            flow_field = torch.cat([x, y], dim=1)
            x_f = self.f_ve(flow_field) + x
            y_f = self.f_repGeno(flow_field) + y
        output = self.output(torch.cat((x_f, y_f), dim=1))
        return output


class MeNet(nn.Module):
    def __init__(self, snp_size, rep_size, config, windows):
        super(MeNet, self).__init__()
        self.windows = windows

        if self.windows:
            self.ve = []
            for i in range(len(windows)):
                self.ve.append(VE(windows[f'{i+1}'], config["VE"]))
            self.fusion = Fusion(config["output"], self.windows)
        else:
            self.ve = VE(snp_size, config["VE"])
            self.fusion = Fusion(config["output"])
        self.repGeno = RepGeno(rep_size, config["RepGeno"])


    def forward(self, x, y):
        y = self.repGeno(y)
        if self.windows:
            x = torch.split(x, list(self.windows.values()), dim=-1)
            x = torch.stack([self.ve[i](x[i]) for i in range(len(self.windows))], dim=0)
            print(x.shape)
        else:
            x = self.ve(x)
        output = self.fusion(x, y)
        return output