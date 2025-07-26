import argparse
import yaml
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args(config: str):
    with open(os.path.join(os.path.dirname(__file__), config), 'r') as f:
        config = yaml.safe_load(f)

    args = argparse.Namespace(**config)
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            args.model[k[2:]] = v
    return args