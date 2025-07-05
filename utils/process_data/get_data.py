import pandas as pd
from utils.process_data.utils import *


def get_data_for_genetic_relatedness(phenotype_path,
             genotype_path = "data/gene/genotype.csv" ,
             train_val_ids_path = "data/splits/"):

    with open(train_val_ids_path + "train_ids.txt", 'r') as f:
        train_ids = f.read().split(",")
    with open(train_val_ids_path + "val_ids.txt", 'r') as f:
        val_ids = f.read().split(",")

    genotype = pd.read_csv(genotype_path, index_col=0)
    count = calculate_snp_number(genotype)
    count = list(count.values())
    phenotype = pd.read_csv(phenotype_path, index_col=0)
    data = pd.merge(phenotype, genotype, left_index=True, right_index=True)
    data_train, data_val = data[data.index.isin(train_ids)], data[data.index.isin(val_ids)]
    snp_train, phen_train = data_train.iloc[:, 1:], data_train.iloc[:, 0]
    snp_val, phen_val = data_val.iloc[:, 1:], data_val.iloc[:, 0]

    snp_train = torch.tensor(snp_train.values, dtype=torch.float)
    snp_val = torch.tensor(snp_val.values, dtype=torch.float)
    snp_train = snp_train.reshape(snp_train.shape[0], 1, -1)
    snp_val = snp_val.reshape(snp_val.shape[0], 1, -1)
    phen_train = torch.tensor(phen_train.values, dtype=torch.float).unsqueeze(1)
    phen_val = torch.tensor(phen_val.values, dtype=torch.float).unsqueeze(1)
    return snp_train, phen_train, snp_val, phen_val, count, data


def get_data_for_menet(phenotype_path,
                       batch_size,
                       genotype_path = "data/gene/genotype.csv",
                       genetic_relatedness_path = "saved/genetic_relatedness.csv",
                       train_val_ids_path = "data/splits/",
                       ):
    phenotype = pd.read_csv(phenotype_path, index_col=0)
    genotype = pd.read_csv(genotype_path, index_col=0)
    count = calculate_snp_number(genotype)
    count = list(count.values())
    genetic_relatedness = pd.read_csv(genetic_relatedness_path, index_col=0)


    with open(train_val_ids_path + "train_ids.txt", 'r') as f:
        train_ids = f.read().split(",")
    with open(train_val_ids_path + "val_ids.txt", 'r') as f:
        val_ids = f.read().split(",")
    with open(train_val_ids_path + "test_ids.txt", 'r') as f:
        test_ids = f.read().split(",")

    phenotype_genotype = pd.merge(phenotype, genotype, left_index=True, right_index=True)
    phenotype_genetic_relatedness = pd.merge(phenotype, genetic_relatedness, left_index=True, right_index=True)

    # split data
    snp_train_tensor, phen_train_tensor, snp_val_tensor, phen_val_tensor, snp_test_tensor, phen_test_tensor = split_train_val_test(
        phenotype_genotype, train_ids, val_ids, test_ids)
    genetic_relatedness_train_tensor, _, genetic_relatedness_val_tensor, _, genetic_relatedness_test_tensor, _ = split_train_val_test(
        phenotype_genetic_relatedness, train_ids, val_ids, test_ids)

    # train_val_test_dataloader
    train_dataset = [snp_train_tensor, genetic_relatedness_train_tensor,phen_train_tensor]
    val_dataset = [snp_val_tensor, genetic_relatedness_val_tensor,phen_val_tensor]
    test_dataset = [snp_test_tensor, genetic_relatedness_test_tensor,phen_test_tensor]
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(train_dataset, val_dataset, test_dataset, batch_size)

    # genotype and genetic relatedness
    genotype_genetic_relatedness = [phenotype_genotype.iloc[:, 1:], phenotype_genetic_relatedness.iloc[:, 1:]]
    return train_dataloader, val_dataloader, test_dataloader, genotype_genetic_relatedness, count






if __name__ == '__main__':
    pass
