import pandas as pd
import numpy as np
import os
import logging

import json
import argparse
from easydict import EasyDict as edict
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train', default=None, type=str, help="If get"
                    "parameters from pretrained model")
parser.add_argument('--resume', default=0, type=int, help="If resume from "
                    "previous run")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")

def run(args):
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
        if args.verbose is True:
            print(json.dumps(cfg, indent=4))

    predictions = pd.read_csv(cfg.pred_csv)
    
    y_pred = predictions['y_pred_Problem_in']

    y_true = predictions['Pneumothorax'].map({1.: 1 , 0.: 0, -1.: 0}).fillna(0)
    sens_features = predictions[cfg.sensitive_attribute]
    # print(sens_features)

    print(equalized_odds_difference(y_true, y_pred, sensitive_features=sens_features))

    print(equalized_odds_ratio(y_true, y_pred, sensitive_features=sens_features))

def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)


if __name__ == '__main__':
    main()