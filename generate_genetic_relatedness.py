from utils.process_data.get_data import get_data_for_genetic_relatedness
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import json
import argparse
from model.backbone_RepGeno.backbone import Net
from torch import nn
from utils.process_model.train_model import repgeno_backbone_train
import random
import torch


seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
config = json.load(open('configs/RepGeno.json'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RegGeno of MeNet")
    parser.add_argument("--phenotype", type=str, default="default", help="The name of the phenotype")

    parser.add_argument("--windows_mechanism", type=int, default=0,
                        help="Windows mechanism 0: don't use windows mechanism, 1: windows mechanism by chromosome")
    parser.add_argument("--windows_chr", type=int, default=2, help="the number of chromosomes in a window")
    parser.add_argument("--device", type=str, default="cuda", help="the device")
    args = parser.parse_args()

    snp_train_tensor, phen_train_tensor, snp_val_tensor, phen_val_tensor, count, data = get_data_for_genetic_relatedness(
        phenotype_path=f"data/phen/{args.phenotype}.csv")
    train_dataloader = DataLoader(TensorDataset(snp_train_tensor, phen_train_tensor),
                                  batch_size=config[args.phenotype]['batch size'],
                                  shuffle=True,
                                  drop_last=True)
    val_dataloader = DataLoader(TensorDataset(snp_val_tensor, phen_val_tensor),
                                batch_size=config[args.phenotype]['batch size'],
                                shuffle=False)

    model = Net(snp_train_tensor.size()[-1],
                [config[args.phenotype]['p'], args.windows_mechanism, args.windows_chr, count])

    loss_fn = nn.L1Loss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    repgeno_backbone_train(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=loss_fn,
        num_epochs=config[args.phenotype]['saved'],
        device=args.device,
        data = data
    )

