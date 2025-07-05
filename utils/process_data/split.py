from sklearn.model_selection import train_test_split
import pandas as pd
import random


def split_data(gene):
    random.seed(42)
    indices = list(gene.index)

    # Split into train (60%) and test (40%)
    train_indices, test_indices = train_test_split(indices, train_size=0.6, random_state=42)

    # Split test indices into test (20%) and validation (20%)
    random.shuffle(test_indices)
    num_val = len(test_indices) // 2
    val_indices = test_indices[:num_val]
    test_indices = test_indices[num_val:]

    # Save  train indices, test and validation indices
    with open("../../data/splits/train_ids.txt", "w") as f:
        f.write(",".join(map(str, train_indices)))
    with open("../../data/splits/test_ids.txt", "w") as f:
        f.write(",".join(map(str, test_indices)))
    with open("../../data/splits/val_ids.txt", "w") as f:
        f.write(",".join(map(str, val_indices)))

if __name__ == '__main__':
    gene = pd.read_csv("data/gene/genotype.csv", index_col=0)
    split_data(gene)