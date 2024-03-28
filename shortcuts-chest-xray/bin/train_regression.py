import sys
import os
import argparse
import logging
import json
import time
import subprocess
from shutil import copyfile

import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa

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

def get_loss(output, target, sensitive_logits, sensitive_target, index, device, cfg):
    
    target = target.to(device)
    sensitive_target = sensitive_target.to(device)

    if cfg.criterion == 'BCE':
        for num_class in cfg.num_classes:
            assert num_class == 1
        target = target[:, index].view(-1)
        pos_weight = torch.from_numpy(
            np.array(cfg.pos_weight,
                     dtype=np.float32)).to(device).type_as(target)
        if cfg.batch_weight:
            if target.sum() == 0:
                primary_loss = torch.tensor(0., requires_grad=True).to(device)
            else:
                weight = (target.size()[0] - target.sum()) / target.sum()
                primary_loss = F.binary_cross_entropy_with_logits(
                    output[index].view(-1), target, pos_weight=weight)
        else:
            primary_loss = F.binary_cross_entropy_with_logits(
                output[index].view(-1), target, pos_weight=pos_weight[index])

        primary_label = torch.sigmoid(output[index].view(-1)).ge(0.5).float()
        primary_acc = (target == primary_label).float().sum() / len(primary_label)
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))
    
  
    if cfg.sensitive_criterion == 'MAE':
        sensitive_target = sensitive_target.view(-1)
        sensitive_logits = sensitive_logits.view(-1)
            
        sensitive_loss = torch.nn.functional.l1_loss(sensitive_logits, sensitive_target)

    else:
        raise Exception('Unknown criterion : {}'.format(cfg.sensitive_criterion))
    
    return (primary_loss, primary_acc), (sensitive_loss)



def train_epoch(summary, summary_dev, cfg, args, model, dataloader,
                dataloader_dev, optimizer, summary_writer, best_dict,
                dev_header):
    torch.set_grad_enabled(True)
    model.train()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    
    dataiter = iter(dataloader)
    label_header = dataloader.dataset._label_header
    num_tasks = len(cfg.num_classes)

    time_now = time.time()

    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks) 
    sensitive_loss_sum = np.zeros(num_tasks)
    
    for step in range(steps):
        image, target, sensitive_target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        output, logit_maps, sensitive_logits, sensitive_feat = model(image)

        # different number of tasks
        total_loss = 0

        for t in range(num_tasks):
            (loss_t, acc_t), (sensitive_loss) = get_loss(output, target, sensitive_logits, sensitive_target, t, device, cfg)
            
            total_loss += loss_t + sensitive_loss
            
            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()

            sensitive_loss_sum[t] += sensitive_loss.item()

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        summary['step'] += 1

        if summary['step'] % cfg.log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg.log_every
            acc_sum /= cfg.log_every
            loss_str = ' '.join(map(lambda x: '{:.5f}'.format(x), loss_sum))
            acc_str = ' '.join(map(lambda x: '{:.3f}'.format(x), acc_sum))

            sensitive_loss_sum /= cfg.log_every
            sensitive_acc_sum /= cfg.log_every
            sensitive_loss_str = ' '.join(map(lambda x: '{:.5f}'.format(x), sensitive_loss_sum))

            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Loss : {}, '
                'Acc : {}, Sensitive_Loss : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], loss_str,
                        acc_str, sensitive_loss_str, time_spent))

            for t in range(num_tasks):
                summary_writer.add_scalar(
                    'train/loss_{}'.format(label_header[t]), loss_sum[t],
                    summary['step'])
                summary_writer.add_scalar(
                    'train/acc_{}'.format(label_header[t]), acc_sum[t],
                    summary['step'])
                
                summary_writer.add_scalar(
                    'train/sensitive_loss_{}'.format(label_header[t]), sensitive_loss_sum[t],
                    summary['step'])

            loss_sum = np.zeros(num_tasks)
            acc_sum = np.zeros(num_tasks)

            sensitive_loss_sum = np.zeros(num_tasks)

        if summary['step'] % cfg.test_every == 0:
            time_now = time.time()

            summary_dev, predlist, true_list, sensitive_predlist, sensitive_true_list = test_epoch(
                summary_dev, cfg, args, model, dataloader_dev)
            time_spent = time.time() - time_now

            auclist = []
            sensitive_auclist = []
            for i in range(len(cfg.num_classes)):
                # main classifier
                y_pred = predlist[i]
                y_true = true_list[i]
                fpr, tpr, thresholds = metrics.roc_curve(
                    y_true, y_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                auclist.append(auc)

                # sensitive classifier
                sensitive_y_pred = sensitive_predlist[i]
                sensitive_y_true = sensitive_true_list[i]
                sensitive_fpr, sensitive_tpr, sensitive_thresholds = metrics.roc_curve(
                    sensitive_y_true, sensitive_y_pred, pos_label=1)
                sensitive_auc = metrics.auc(sensitive_fpr, sensitive_tpr)
                sensitive_auclist.append(sensitive_auc)

            summary_dev['auc'] = np.array(auclist)
            summary_dev['sensitive_auc'] = np.array(sensitive_auclist)

            loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                        summary_dev['loss']))
            acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['acc']))
            auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['auc']))
            
            sensitive_loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                        summary_dev['sensitive_loss']))
            sensitive_acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['sensitive_acc']))
            sensitive_auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['sensitive_auc']))

            logging.info(
                '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {}, Sensitive_Loss : {}, Sensitive_Acc : {}, Sensitive_Auc : {}, Mean auc: {:.3f}, Run Time : {:.2f} sec'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    sensitive_loss_dev_str,
                    sensitive_acc_dev_str,
                    sensitive_auc_dev_str,
                    summary_dev['auc'].mean(),
                    time_spent))


            for t in range(len(cfg.num_classes)):
                summary_writer.add_scalar(
                    'dev/loss_{}'.format(dev_header[t]),
                    summary_dev['loss'][t], summary['step'])
                summary_writer.add_scalar(
                    'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
                    summary['step'])
                summary_writer.add_scalar(
                    'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
                    summary['step'])
                
                summary_writer.add_scalar(
                    'dev/sensitive_loss_{}'.format(dev_header[t]),
                    summary_dev['sensitive_loss'][t], summary['step'])
                summary_writer.add_scalar(
                    'dev/sensitive_acc_{}'.format(dev_header[t]), summary_dev['sensitive_acc'][t],
                    summary['step'])
                summary_writer.add_scalar(
                    'dev/sensitive_auc_{}'.format(dev_header[t]), summary_dev['sensitive_auc'][t],
                    summary['step'])

            save_best = False
            mean_acc = summary_dev['acc'][cfg.save_index].mean()
            if mean_acc >= best_dict['acc_dev_best']:
                best_dict['acc_dev_best'] = mean_acc
                if cfg.best_target == 'acc':
                    save_best = True

            mean_auc = summary_dev['auc'][cfg.save_index].mean()
            if mean_auc >= best_dict['auc_dev_best']:
                best_dict['auc_dev_best'] = mean_auc
                if cfg.best_target == 'auc':
                    save_best = True

            mean_loss = summary_dev['loss'][cfg.save_index].mean()
            if mean_loss <= best_dict['loss_dev_best']:
                best_dict['loss_dev_best'] = mean_loss
                if cfg.best_target == 'loss':
                    save_best = True

            if save_best:
                torch.save(
                    {'epoch': summary['epoch'],
                     'step': summary['step'],
                     'acc_dev_best': best_dict['acc_dev_best'],
                     'auc_dev_best': best_dict['auc_dev_best'],
                     'loss_dev_best': best_dict['loss_dev_best'],
                     'sensitive_acc_dev_best': best_dict['sensitive_acc_dev_best'],
                     'sensitive_auc_dev_best': best_dict['sensitive_auc_dev_best'],
                     'sensitive_loss_dev_best': best_dict['sensitive_loss_dev_best'],
                     'state_dict': model.module.state_dict()},
                    os.path.join(args.save_path, 'best{}.ckpt'.format(
                        best_dict['best_idx']))
                )
                best_dict['best_idx'] += 1
                if best_dict['best_idx'] > cfg.save_top_k:
                    best_dict['best_idx'] = 1
                
                logging.info(
                    '{}, Best, Step : {}, Loss : {}, Acc : {}, Auc : {}, Sensitive_Loss : {}, Sensitive_Acc : {}, Sensitive_Auc : {}, Best Auc : {:.3f}'.format(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['step'],
                        loss_dev_str,
                        acc_dev_str,
                        auc_dev_str,
                        sensitive_loss_dev_str,
                        sensitive_acc_dev_str,
                        sensitive_auc_dev_str,
                        best_dict['auc_dev_best']))

        model.train()
        torch.set_grad_enabled(True)
    summary['epoch'] += 1

    return summary, best_dict

                
def test_epoch(summary, cfg, args, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)

    sensitive_loss_sum = 0

    predlist = list(x for x in range(len(cfg.num_classes)))
    true_list = list(x for x in range(len(cfg.num_classes)))

    sensitive_predlist = list()
    sensitive_true_list = list()

    for step in range(steps):
        image, target, sensitive_target = next(dataiter)

        image = image.to(device)
        target = target.to(device)
        sensitive_target = sensitive_target.to(device)
        output, logit_map, sensitive_logits, sensitive_feat = model(image)

        # different number of tasks
        for t in range(len(cfg.num_classes)):

            (loss_t, acc_t), (sensitive_loss) = get_loss(output, target, sensitive_logits, sensitive_target, t, device, cfg)

            # AUC
            output_tensor = torch.sigmoid(
                output[t].view(-1)).cpu().detach().numpy()
            target_tensor = target[:, t].view(-1).cpu().detach().numpy()

            # sensitive_output_tensor = torch.sigmoid(
            #     sensitive_logits[t].view(-1)).cpu().detach().numpy()
            # sensitive_target_tensor = sensitive_target[t].view(-1).cpu().detach().numpy()

            if step == 0:
                predlist[t] = output_tensor
                true_list[t] = target_tensor
                # sensitive_predlist[t] = sensitive_output_tensor
                # sensitive_true_list[t] = sensitive_target_tensor

            else:
                predlist[t] = np.append(predlist[t], output_tensor)
                true_list[t] = np.append(true_list[t], target_tensor)
                # sensitive_predlist[t] = np.append(sensitive_predlist[t], sensitive_output_tensor)
                # sensitive_true_list[t] = np.append(sensitive_true_list[t], sensitive_target_tensor)

            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()


        
        (loss_t, acc_t), (sensitive_loss) = get_loss(output, target, sensitive_logits, sensitive_target, t, device, cfg)

        sensitive_predlist.extend(sensitive_logits.view(-1).cpu().detach().numpy())
        sensitive_true_list.extend(sensitive_target.view(-1).cpu().detach().numpy())

        sensitive_loss_sum += sensitive_loss.item()

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    summary['sensitive_loss'] = sensitive_loss_sum / steps

    return summary, predlist, true_list, sensitive_predlist, sensitive_true_list

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

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
            .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg)
    if args.verbose is True:
        from torchsummary import summary
        if cfg.fix_ratio:
            h, w = cfg.long_side, cfg.long_side
        else:
            h, w = cfg.height, cfg.width
        summary(model.to(device), (3, h, w))

    model = DataParallel(model, device_ids=device_ids).to(device).train()
    if args.pre_train is not None:
        if os.path.exists(args.pre_train):
            ckpt = torch.load(args.pre_train, map_location=device)
            model.module.load_state_dict(ckpt)
    optimizer = get_optimizer(model.parameters(), cfg)
    
    dataloader_train = DataLoader(
        ImageDataset(cfg.train_csv, cfg, mode='train'),
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=True, shuffle=True)
    dataloader_dev = DataLoader(
        ImageDataset(cfg.dev_csv, cfg, mode='dev'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)
    dev_header = dataloader_dev.dataset._label_header

    summary_train = {'epoch': 0, 'step': 0}
    summary_dev = {'loss': float('inf'), 'acc': 0.0, 'sensitive_loss': float('inf')}
    summary_writer = SummaryWriter(args.save_path)
    epoch_start = 0
    best_dict = {
        "acc_dev_best": 0.0,
        "auc_dev_best": 0.0, 
        "loss_dev_best": float('inf'),
        "sensitive_loss_dev_best": float('inf'),
        "fused_dev_best": 0.0,
        "best_idx": 1}

    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'train.ckpt')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['state_dict'])
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        best_dict['acc_dev_best'] = ckpt['acc_dev_best']
        best_dict['auc_dev_best'] = ckpt['auc_dev_best']
        best_dict['loss_dev_best'] = ckpt['loss_dev_best']
        best_dict['sensitive_loss_dev_best'] = ckpt['sensitive_loss_dev_best']
        epoch_start = ckpt['epoch']

    for epoch in range(epoch_start, cfg.epoch):
        lr = lr_schedule(cfg.lr, cfg.lr_factor, summary_train['epoch'],
                         cfg.lr_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        summary_train, best_dict = train_epoch(
            summary_train, summary_dev, cfg, args, model,
            dataloader_train, dataloader_dev, optimizer,
            summary_writer, best_dict, dev_header)

        time_now = time.time()
        summary_dev, predlist, true_list, sensitive_predlist, sensitive_true_list = test_epoch(
                summary_dev, cfg, args, model, dataloader_dev)
        time_spent = time.time() - time_now

        auclist = []
        sensitive_auclist = []

        for i in range(len(cfg.num_classes)):
            y_pred = predlist[i]
            y_true = true_list[i]

            sensitive_y_pred = sensitive_predlist[i]
            sensitive_y_true = sensitive_true_list[i]

            logging.info(time.strftime("%Y-%m-%d %H:%M:%S"))
            logging.info('y_pred: {}'.format(y_pred))
            logging.info('y_true: {}'.format(y_true))

            logging.info('sensitive_y_pred: {}'.format(sensitive_y_pred))
            logging.info('sensitive_y_true: {}'.format(sensitive_y_true))

            fpr, tpr, thresholds = metrics.roc_curve(
                y_true, y_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auclist.append(auc)

            sensitive_fpr, sensitive_tpr, sensitive_thresholds = metrics.roc_curve(
                sensitive_y_true, sensitive_y_pred, pos_label=1)
            sensitive_auc = metrics.auc(sensitive_fpr, sensitive_tpr)
            sensitive_auclist.append(sensitive_auc)

        summary_dev['auc'] = np.array(auclist)
        summary_dev['sensitive_auc'] = np.array(sensitive_auclist)

        loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                    summary_dev['loss']))
        acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['acc']))
        auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['auc']))
        sensitive_loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                    summary_dev['sensitive_loss']))
        sensitive_acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['sensitive_acc']))
        sensitive_auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['sensitive_auc']))
        
        logging.info(
                '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {}, Sensitive_Loss : {}, Sensitive_Acc : {}, Sensitive_Auc : {}, Mean auc: {:.3f}, Run Time : {:.2f} sec'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    sensitive_loss_dev_str,
                    sensitive_acc_dev_str,
                    sensitive_auc_dev_str,
                    summary_dev['auc'].mean(),
                    time_spent))

        for t in range(len(cfg.num_classes)):
            summary_writer.add_scalar(
                'dev/loss_{}'.format(dev_header[t]), summary_dev['loss'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/sensitive_loss_{}'.format(dev_header[t]), summary_dev['sensitive_loss'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/sensitive_acc_{}'.format(dev_header[t]), summary_dev['sensitive_acc'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/sensitive_auc_{}'.format(dev_header[t]), summary_dev['sensitive_auc'][t],
                summary_train['step'])

        save_best = False

        mean_acc = summary_dev['acc'][cfg.save_index].mean()
        if mean_acc >= best_dict['acc_dev_best']:
            best_dict['acc_dev_best'] = mean_acc
            if cfg.best_target == 'acc':
                save_best = True

        mean_auc = summary_dev['auc'][cfg.save_index].mean()
        if mean_auc >= best_dict['auc_dev_best']:
            best_dict['auc_dev_best'] = mean_auc
            if cfg.best_target == 'auc':
                save_best = True

        mean_loss = summary_dev['loss'][cfg.save_index].mean()
        if mean_loss <= best_dict['loss_dev_best']:
            best_dict['loss_dev_best'] = mean_loss
            if cfg.best_target == 'loss':
                save_best = True
        
        # mean_sensitive_acc = summary_dev['sensitive_acc'][cfg.save_index].mean()
        # if mean_sensitive_acc >= best_dict['sensitive_acc_dev_best']:
        #     best_dict['sensitive_acc_dev_best'] = mean_sensitive_acc
        #     if cfg.best_target == 'sensitive_acc':
        #         save_best = True

        # mean_sensitive_auc = summary_dev['sensitive_auc'][cfg.save_index].mean()
        # if mean_sensitive_auc >= best_dict['sensitive_auc_dev_best']:
        #     best_dict['sensitive_auc_dev_best'] = mean_sensitive_auc
        #     if cfg.best_target == 'sensitive_auc':
        #         save_best = True

        if save_best:
            torch.save(
                {'epoch': summary_train['epoch'],
                 'step': summary_train['step'],
                 'acc_dev_best': best_dict['acc_dev_best'],
                 'auc_dev_best': best_dict['auc_dev_best'],
                 'loss_dev_best': best_dict['loss_dev_best'],
                 'sensitive_acc_dev_best': best_dict['sensitive_acc_dev_best'],
                 'sensitive_auc_dev_best': best_dict['sensitive_auc_dev_best'],
                 'sensitive_loss_dev_best': best_dict['sensitive_loss_dev_best'],
                 'state_dict': model.module.state_dict()},
                os.path.join(args.save_path,
                             'best{}.ckpt'.format(best_dict['best_idx']))
            )

            best_dict['best_idx'] += 1
            if best_dict['best_idx'] > cfg.save_top_k:
                best_dict['best_idx'] = 1

            logging.info(
                '{}, Best, Step : {}, Loss : {}, Acc : {},'
                'Auc : {}, Best Auc : {:.3f}, Sensitive_Loss : {}, Sensitive_Acc : {}, Sensitive_Auc : {}' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    best_dict['auc_dev_best'],
                    sensitive_loss_dev_str,
                    sensitive_acc_dev_str,
                    sensitive_auc_dev_str))
            
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'acc_dev_best': best_dict['acc_dev_best'],
                    'auc_dev_best': best_dict['auc_dev_best'],
                    'loss_dev_best': best_dict['loss_dev_best'],
                    'sensitive_acc_dev_best': best_dict['sensitive_acc_dev_best'],
                    'sensitive_auc_dev_best': best_dict['sensitive_auc_dev_best'],
                    'sensitive_loss_dev_best': best_dict['sensitive_loss_dev_best'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))
    summary_writer.close()



def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)


if __name__ == '__main__':
    main()