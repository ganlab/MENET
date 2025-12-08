import json
from utils.utils import *
from network.contrastive_learning import TraitSpecificEncoderForRepGeno
from utils.dataset import create_triplet_dataloader
from utils.loss import TripletLoss
from utils.train import train_trait_specific_encoder
from utils.relatedness import calculate_genetic_relatedness
import argparse


set_seed(42)
config = json.load(open('configs/contrastive_learning.json'))

def main(config):
    data, data_train, data_val, _ = get_phen_snp(config, f"{config['phen_name']}")

    train_loader = create_triplet_dataloader(config, data_train, flag=config["flag"], shuffle=True, drop_last=True)
    val_loader = create_triplet_dataloader(config, data_val)

    model = TraitSpecificEncoderForRepGeno(snp_size=data_train.shape[-1] - 1,
                                           stride=config['stride'],
                                           out_dim=config['out_dim'])
    criterion = TripletLoss(margin=config['margin'])
    best_model = train_trait_specific_encoder(config=config,
                                              model=model,
                                              train_loader=train_loader,
                                              val_loader=val_loader,
                                              criterion=criterion)
    calculate_genetic_relatedness(best_model, data, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--margin", type=float, default=None)
    parser.add_argument("-p", "--phen_name", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("-f", "--flag", type=int, default=0)
    args = parser.parse_args()
    config = re_set_config(config, args)
    main(config)
