from utils.utils import *
import json
from utils.dataset import create_dual_scale_dataloader, prepare_tensors
from utils.loss import L1Loss
from network.menet import MeNet
from utils.train import train_menet
import argparse

set_seed(42)
config = json.load(open('configs/MeNet.json'))

def main(config):
    phen_snp, phen_snp_train, phen_snp_val, phen_snp_test = get_phen_snp(config, f"{config['phen_name']}")
    phen_gr, phen_gr_train, phen_gr_val, phen_gr_test = get_phen_gr(config, f"{config['phen_name']}")
    train_dataloader = create_dual_scale_dataloader(phen_snp_train, phen_gr_train, config, shuffle=True, drop_last=True)
    val_dataloader = create_dual_scale_dataloader(phen_snp_val, phen_gr_val, config)
    test_dataloader = create_dual_scale_dataloader(phen_snp_test, phen_gr_test, config)
    tensor_for_ig = prepare_tensors(phen_snp, phen_gr)
    chr_counter = windows_flag(config, phen_snp)
    model = MeNet(phen_snp.shape[-1] - 1, phen_gr.shape[-1] - 1, config["param"], windows=chr_counter)
    criterion = L1Loss()
    train_menet(config, model, train_dataloader, val_dataloader, test_dataloader, criterion, tensor_for_ig,
                windows=chr_counter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phen_name", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("-w", "--windows", type=int, default=0)
    args = parser.parse_args()
    config = re_set_config(config, args)
    main(config)

