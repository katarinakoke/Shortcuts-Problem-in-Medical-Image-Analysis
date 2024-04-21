import pandas as pd
import numpy as np
import os
import logging

import json
import argparse
from easydict import EasyDict as edict
from sklearn import metrics
from sklearn.metrics import average_precision_score

parser = argparse.ArgumentParser(description='Check performance of the model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")


def area_under_the_curve(y_pred, y_true, sens_attr):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def average_precision(y_pred, y_true):
    ap_score = average_precision_score(y_true, y_pred)
    return ap_score

def run(args):
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
        if args.verbose is True:
            print(json.dumps(cfg, indent=4))

    predictions = pd.read_csv(cfg.pred_csv, index_col=0)
    y_pred = predictions['y_pred_Problem_in']
    y_true = predictions['Pneumothorax'].map({1.: 1 , 0.: 0, -1.: 0}).fillna(0)
    y_pred = y_pred[y_true.isna()]
    y_true = y_true.dropna()

    auc = area_under_the_curve(y_pred, y_true, cfg.sensitive_attribute)
    print(auc)

    ap_score = average_precision(y_pred, y_true)
    print(ap_score)

def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)

if __name__ == '__main__':
    main()