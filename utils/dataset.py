import torch
import random
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, data, flag):
        self.genotype = _extract_features(data)
        self.population_labels = _extract_population_labels(data, flag)
        self.phenotype_target = _extract_phenotype(data)

    def __getitem__(self, index):
        return self.genotype[index], self.population_labels[index], self.phenotype_target[index]

    def __len__(self):
        return len(self.genotype)

    @property
    def targets(self):
        return self.population_labels


def _build_label_index(labels):
    label_to_index = {}
    for index, label in enumerate(labels):
        label = int(label)
        if label not in label_to_index:
            label_to_index[label] = []
        label_to_index[label].append(index)
    return label_to_index


class TripletDataset(Dataset):
    def __init__(self, dataset, flag):
        self.dataset = dataset
        self.labels = dataset.targets
        self.label_to_index = _build_label_index(self.labels)
        self.flag = flag
    def __getitem__(self, index):
        anchor_genotype, anchor_label, anchor_phenotype = self.dataset[index]
        anchor_label = int(anchor_label)

        # positive sample
        positive_candidates = self.label_to_index[anchor_label]
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(positive_candidates)
        positive_genotype, _, positive_phenotype = self.dataset[positive_index]

        # negative sample
        if self.flag:
            negative_label = anchor_label
            all_labels = list(self.label_to_index.keys())
            while negative_label == anchor_label:
                negative_label = random.choice(all_labels)
            negative_index = random.choice(self.label_to_index[negative_label])
        else:
            negative_candidates = self.label_to_index[anchor_label]
            negative_index = index
            while negative_index == index:
                negative_index = random.choice(negative_candidates)
        negative_genotype, _, negative_phenotype = self.dataset[negative_index]

        return (
            anchor_genotype, anchor_phenotype,
            positive_genotype, positive_phenotype,
            negative_genotype, negative_phenotype
        )

    def __len__(self):
        return len(self.dataset)


def _extract_features(data):
    features = data.iloc[:, 1:].values
    features = torch.tensor(features, dtype=torch.float32)
    features = features.unsqueeze(1)  # [N, 1, D]
    return features


def _extract_population_labels(data, flag=0):
    if flag:   # Assign labels to the data based on subpopulation membership, taking maize data as an example.
        index_list = list(data.index)
        suffixes = [f"MG_{i}" for i in range(1518, 1537)] + [f"MG_{i}" for i in range(1538, 1549)]
        suffixes_to_label = {suffix: idx for idx, suffix in enumerate(suffixes)}
        labels = []
        for idx in index_list:
            suffix = idx.split("_X_")[1]
            if suffix not in suffixes_to_label:
                raise ValueError(f"Unrecognized group prefix '{suffix}' in index '{idx}'")
            labels.append(suffixes_to_label[suffix])
    else:
        labels = [0 for _ in list(data.index)]
    return torch.tensor(labels, dtype=torch.long)


def _extract_phenotype(data):
    phenotype = data.iloc[:, 0].values
    return torch.tensor(phenotype, dtype=torch.float32).unsqueeze(1)  # [N, 1]


def create_triplet_dataloader(config, data, flag=0, shuffle=False, drop_last=False):
    return DataLoader(TripletDataset(MyDataset(data, flag=flag), flag=flag), batch_size=config['batch_size'], shuffle=shuffle, drop_last=drop_last)



def prepare_tensors(phen_snp_df, phen_gr_df):
    snp = phen_snp_df.iloc[:, 1:]
    phen = phen_snp_df.iloc[:, 0]
    gr = phen_gr_df.iloc[:, 1:]
    snp_tensor = torch.tensor(snp.values, dtype=torch.float).reshape(snp.shape[0], 1, -1)
    phen_tensor = torch.tensor(phen.values, dtype=torch.float).unsqueeze(1)
    gr_tensor = torch.tensor(gr.values, dtype=torch.float).reshape(gr.shape[0], 1, -1)
    return snp_tensor, gr_tensor, phen_tensor

def create_dual_scale_dataloader(phen_snp, phen_gr, config, shuffle=False, drop_last=False):
    x1, x2, y = prepare_tensors(phen_snp, phen_gr)
    return DataLoader(TensorDataset(x1, x2, y),batch_size=config['batch_size'], shuffle=shuffle, drop_last=drop_last)



if __name__ == '__main__':
    pass
