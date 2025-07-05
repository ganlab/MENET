from collections import Counter
import torch
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd


def calculate_snp_number(genotype):
    chr_snp_list = list(genotype.columns)
    chr_list = [item.split('_')[0] for item in chr_snp_list]
    chr_counter = Counter(chr_list)
    return chr_counter


def split_train_val_test(data, train_ids, val_ids, test_ids):
    X, Y = data.iloc[:, 1:], data.iloc[:, 0]
    X_train, Y_train = X[X.index.isin(train_ids)], Y[Y.index.isin(train_ids)]
    X_val, Y_val = X[X.index.isin(val_ids)], Y[Y.index.isin(val_ids)]
    X_test, Y_test = X[X.index.isin(test_ids)], Y[Y.index.isin(test_ids)]
    # X : transform to tensor
    X_train = torch.tensor(X_train.values, dtype=torch.float)
    X_val = torch.tensor(X_val.values, dtype=torch.float)
    X_test = torch.tensor(X_test.values, dtype=torch.float)
    # X : reshape
    X_train = X_train.reshape(X_train.shape[0], 1, -1)
    X_val = X_val.reshape(X_val.shape[0], 1, -1)
    X_test = X_test.reshape(X_test.shape[0], 1, -1)

    # Y
    Y_train = torch.tensor(Y_train.values, dtype=torch.float).unsqueeze(1)
    Y_val = torch.tensor(Y_val.values, dtype=torch.float).unsqueeze(1)
    Y_test = torch.tensor(Y_test.values, dtype=torch.float).unsqueeze(1)
    return  X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_dataloader(train_dataset, val_dataset, test_dataset, batch_size = 128):
    train_dataloader = DataLoader(dataset=TensorDataset(train_dataset[0], train_dataset[1], train_dataset[2]),
                                  batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=TensorDataset(val_dataset[0], val_dataset[1], val_dataset[2]),
                                batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=TensorDataset(test_dataset[0], test_dataset[1], test_dataset[2]),
                                 batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

def job_Euclidean(feature, file_path):
    # create G
    features_tensor = torch.tensor(feature.values, dtype=torch.float32)
    norms = features_tensor.norm(p=2, dim=1, keepdim=True)
    features_tensor = features_tensor / norms
    distance_matrix = torch.cdist(features_tensor, features_tensor, p=2)

    max_distance = torch.max(distance_matrix).item()
    gr = 1 - distance_matrix / max_distance
    gr_cpu = gr.cpu().numpy()
    gr_df = pd.DataFrame(gr_cpu, index=feature.index, columns=feature.index)
    gr_df.index.name = "FID"
    gr_df.astype("float16")
    # save G
    gr_df.to_csv(file_path, index=True, header=True)
