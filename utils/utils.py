import random
import numpy as np
import torch
import pandas as pd
from collections import Counter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def read_train_test_index(path):
    with open(f"{path}/split/train_index.txt", "r") as f:
        train_ids = f.read().split('\n')
    with open(f"{path}/split/test_index.txt", "r") as f:
        test_ids = f.read().split('\n')
    with open(f"{path}/split/valid_index.txt", "r") as f:
        val_ids = f.read().split('\n')
    return train_ids, val_ids, test_ids


def split_data(y, x, path):
    train_ids, val_ids, test_ids = read_train_test_index(path)
    data = pd.merge(y, x, left_index=True, right_index=True)
    data_train = data[data.index.isin(train_ids)]
    data_test = data[data.index.isin(test_ids)]
    data_val = data[data.index.isin(val_ids)]
    return data, data_train, data_val, data_test

def get_phen_snp(config, phen_name):
    phen_df = pd.read_csv(f"{config['root_path']}/phenotype/{phen_name}.csv", index_col=0)
    # genotype = torch.load(f"{config['root_path']}/genotype/genotype.pt")
    genotype = pd.read_csv(f"{config['root_path']}/genotype/genotype.csv", index_col=0)
    return split_data(phen_df, genotype, config["root_path"])

def get_phen_gr(config, phen_name):
    phen_df = pd.read_csv(f"{config['root_path']}/phenotype/{phen_name}.csv", index_col=0)
    gr = torch.load(f"{config['gr_path']}/genetic_relatedness.pt")
    return split_data(phen_df, gr, config["root_path"])

def calculate_snp_number(genotype):
    chr_snp_list = list(genotype.columns)
    chr_list = [item.split('_')[0] for item in chr_snp_list]
    chr_counter = Counter(chr_list)
    chr_counter = {k: chr_counter[k] for k in sorted(chr_counter.keys(), key=int)}
    return chr_counter

def re_set_config(config, args):
    for key in vars(args).keys():
        if vars(args)[key] is not None:
            config[key] = vars(args)[key]
    return config

def windows_flag(config, data_df):
    if config['windows'] or data_df.shape[-1]>100000:
        chr_counter = calculate_snp_number(data_df.iloc[:, 1:])
    else:
        chr_counter = None
    return chr_counter


if __name__ == '__main__':
    pass
