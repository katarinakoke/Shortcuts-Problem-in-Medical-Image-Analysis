import pandas as pd
import numpy as np
import os
import logging

import json
import argparse
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train', default=None, type=str, help="If get"
                    "parameters from pretrained model")
parser.add_argument('--resume', default=0, type=int, help="If resume from "
                    "previous run")
parser.add_argument('--logtofile', default=False, type=bool, help="Save log "
                    "in save_path/log.txt if set True")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")


def fairness_metrics(y_pred, y_true, sens_attr):
    eps = 1e-5
    groups = np.unique(sens_attr).tolist()
   
    max_error = 0
    min_error = 1

    max_mean_y = 0
    min_mean_y = 1

    max_mean_y0 = 0  # conditioned on y = 0
    min_mean_y0 = 1
   
    max_mean_y1 = 0
    min_mean_y1 = 1

    for group in groups:
        yt = y_true[sens_attr == group].astype('int32')
        ypt = (y_pred[sens_attr == group]).astype('int32')
        err = -np.mean(yt * np.log(ypt+eps) + (1-yt)*np.log(1-ypt+eps))
        mean_y = np.mean(y_pred[sens_attr == group])
        neg = np.logical_and(sens_attr == group, y_true == 0)
        pos = np.logical_and(sens_attr == group, y_true == 1)
        mean_y0 = np.mean(y_pred[neg])
        mean_y1 = np.mean(y_pred[pos])
      
        if err > max_error:
            max_error = err
        if err < min_error:
            min_error = err

        if mean_y > max_mean_y:
            max_mean_y = mean_y
        if mean_y < min_mean_y:
            min_mean_y = mean_y

        if mean_y0 > max_mean_y0:
            max_mean_y0 = mean_y0
        if mean_y0 < min_mean_y0:
            min_mean_y0 = mean_y0

        if mean_y1 > max_mean_y1:
            max_mean_y1 = mean_y1
        if mean_y1 < min_mean_y1:
            min_mean_y1 = mean_y1
    
    eo = 0.5*(max_mean_y0 - min_mean_y0 + max_mean_y1 - min_mean_y1)
    dp = max_mean_y - min_mean_y
    err_parity = max_error - min_error
    return eo, dp, err_parity

def run(args):
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
        if args.verbose is True:
            print(json.dumps(cfg, indent=4))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if args.logtofile is True:
        logging.basicConfig(filename=args.save_path + '/log.txt',
                            filemode="w", level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    predictions = pd.read_csv(f'{cfg.pred_model, ".csv"}', index_col=0)
    
    y_pred = predictions['y_pred_Problem_in']
    y_true = predictions['Pneumothorax']

    fairness_metrics(y_pred, y_true, cfg.sensitive_attribute)

def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)


if __name__ == '__main__':
    main()