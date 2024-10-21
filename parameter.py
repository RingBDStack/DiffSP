import yaml
import torch
import argparse
from easydict import EasyDict as edict

def parameter_parser():
    parser = argparse.ArgumentParser()

    # === basic ===
    parser.add_argument('--device_id', type=str, default='0', help='device id')
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
    parser.add_argument('--attack', type=str, default='prbcd', help='attack')
    parser.add_argument('--is_train', action='store_true', help='If set, train is True')

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.device_id))
        print('Device: gpu-{}'.format(args.device_id))
    else:
        args.device = torch.device("cpu")
        print('Device: cpu')
    
    # === special ===
    def load_config(config_mode):
        config_file = './config/{}.yaml'.format(config_mode)
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

    config = load_config(args.dataset)
    for key, value in config.items():
        setattr(args, key, value)
    
    # === return ===
    return edict(vars(args))