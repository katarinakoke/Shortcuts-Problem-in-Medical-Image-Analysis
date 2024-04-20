import pandas as pd
import numpy as np
import os
import logging

import json
import argparse
from easydict import EasyDict as edict
from sklearn import metrics

parser = argparse.ArgumentParser(description='Check performance of the model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
# parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
#                     help="Path to the saved models")
# parser.add_argument('--num_workers', default=8, type=int, help="Number of "
#                     "workers for each data loader")
# parser.add_argument('--device_ids', default='0,1,2,3', type=str,
#                     help="GPU indices ""comma separated, e.g. '0,1' ")
# parser.add_argument('--pre_train', default=None, type=str, help="If get"
#                     "parameters from pretrained model")
# parser.add_argument('--resume', default=0, type=int, help="If resume from "
#                     "previous run")
# parser.add_argument('--logtofile', default=False, type=bool, help="Save log "
#                     "in save_path/log.txt if set True")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")


def performance_metrics(y_pred, y_true, sens_attr):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def run(args):
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
        if args.verbose is True:
            print(json.dumps(cfg, indent=4))

    # if not os.path.exists(args.save_path):
    #     os.mkdir(args.save_path)
    # if args.logtofile is True:
    #     logging.basicConfig(filename=args.save_path + '/log.txt',
    #                         filemode="w", level=logging.INFO)
    # else:
    #     logging.basicConfig(level=logging.INFO)

    # if not args.resume:
    #     with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
    #         json.dump(cfg, f, indent=1)

    predictions = pd.read_csv(cfg.pred_csv, index_col=0)
    y_pred = predictions['y_pred_Problem_in']
    y_true = predictions['Pneumothorax']
    y_pred = y_pred[~y_true.isna()]
    y_true = y_true.dropna()

    auc = performance_metrics(y_pred, y_true, cfg.sensitive_attribute)
    print(auc)

def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)

if __name__ == '__main__':
    main()