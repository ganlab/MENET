import torch
from torch import nn
import numpy as np
import random
from model.menet.embedding_fusion import Net
from utils.process_data.get_data import get_data_for_menet
from utils.process_model.train_model import  menet_train
import json
import argparse

config = json.load(open('configs/MeNet.json'))

seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MeNet")
    parser.add_argument("--phenotype", type=str, default="default", help="The name of the phenotype")
    parser.add_argument("--windows_mechanism", type=int, default=0,
                        help="Windows mechanism 0: don't use windows_mechanism, 1: windows_mechanism by chromosome")
    parser.add_argument("--windows_chr", type=int, default=2, help="the number of chromosomes in a window")
    parser.add_argument("--device", type=str, default="cuda", help="the device")
    args = parser.parse_args()
    device = args.device

    train_dataloader, val_dataloader, test_dataloader, ggr, count = get_data_for_menet(f"data/phen/{args.phenotype}.csv",
                                                                                       batch_size=config[args.phenotype]['batch size'], )

    ve_size = ggr[0].shape[1]
    gr_size = ggr[1].shape[0]
    model = Net(ve_size, gr_size, config[args.phenotype], [args.windows_mechanism, args.windows_chr, count]).to(device)
    loss_fn = nn.L1Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    model = menet_train(model = model,
                        train_dataloader = train_dataloader,
                        val_dataloader = val_dataloader,
                        test_dataloader = test_dataloader,
                        criterion = loss_fn,
                        optimizer = optimizer,
                        num_epochs = config[args.phenotype]['saved'],
                        device = device,
                        scheduler = scheduler,
                        ggr = ggr)
    torch.save(model, f"saved/menet.pt")

