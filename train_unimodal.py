import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset, dataset
from torchvision.ops import sigmoid_focal_loss

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score
from tqdm import tqdm
from tensorboardX import SummaryWriter

import os
import pprint
import argparse
import time
import json
import warnings

# dataset and models
from dataset import AnaxnetDataset
from model.anaxnet import AnaXnetGCN
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
# action
parser.add_argument('--load_config', type=str, help='Path to config.json file to load args from.')
parser.add_argument('--train', default=True, action='store_true', help='Train model.')
parser.add_argument('--evaluate_single_model', default=False, action='store_true', help='Evaluate a single model.')
parser.add_argument('--evaluate_ensemble', action='store_false', help='Evaluate an ensemble (given a checkpoints tracker of saved model checkpoints).')
parser.add_argument('--visualize', default=True, action='store_true', help='Visualize Grad-CAM.')
parser.add_argument('--plot_roc', action='store_true', help='Filename for metrics json file to plot ROC.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
# paths
parser.add_argument('--data_path', default='./CheXpert-v1.0-small/', help='Location of train/valid datasets directory or path to test csv file.')
parser.add_argument('--output_dir', help='Path to experiment output, config, checkpoints, etc.')
parser.add_argument('--restore', default='./mimic-densenet/best_checkpoints/checkpoint_9.pt', type=str, help='Path to a single model checkpoint to restore or folder of checkpoints to ensemble.')
# model architecture
parser.add_argument('--model', default='densenet121', help='What model architecture to use. (densenet121, resnet152, efficientnet-b[0-7])')
# data params
parser.add_argument('--mini_data', type=int, help='Truncate dataset to this number of examples.')
parser.add_argument('--resize', type=int, help='Size of minimum edge to which to resize images.')
# training params
parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained model and normalize data mean and std.')
parser.add_argument('--batch_size', type=int, default=16, help='Dataloaders batch size.')
parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--lr_warmup_steps', type=float, default=0, help='Linear warmup of the learning rate for lr_warmup_steps number of steps.')
parser.add_argument('--lr_decay_factor', type=float, default=0.97, help='Decay factor if exponential learning rate decay scheduler.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--log_interval', type=int, default=50, help='Interval of num batches to show loss statistics.')
parser.add_argument('--eval_interval', type=int, default=100, help='Interval of num epochs to evaluate, checkpoint, and save samples.')
parser.add_argument('--focal_alpha', type=float, default=0.25, help='Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore.')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
diseases = ["Lung opacity", "Pleural effusion", "Atelectasis", "Enlarged cardiac silhouette", "Pulmonary edema/hazy opacity", 
            "Pneumothorax", "Consolidation" , "Fluid overload/heart failure", "Pneumonia"]
# --------------------
# Distributed setup
# --------------------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


# --------------------
# Data IO
# --------------------

class InfiniteDataLoader(DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)  # type: ignore

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def get_loader(rank, root_folder,mode, batch_size=16,num_workers=0,shuffle=False,pin_memory=True, world_size=1):
    assert mode in ['train', 'valid', 'test']
    dataset= AnaxnetDataset(os.path.join(root_folder, 'new_{}.csv'.format(mode))) # 'sample_{}.csv'.format(mode)))  #

    if batch_size % world_size != 0:
        raise Exception("Batch size must be a multiple of the number of workers")

    batch_size = batch_size // world_size

    print(f"World size: {world_size}, setting effective batch size to {batch_size}. Should be batch size / num input gpus.")

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
        loader = InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True
            )
    else:
        loader = InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
    return loader, dataset


def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_checkpoint(checkpoint, optim_checkpoint, sched_checkpoint, args, max_records=10):
    """ save model and optimizer checkpoint along with csv tracker
    of last `max_records` best number of checkpoints as sorted by avg auc """
    # 1. save latest
    torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pt'))
    torch.save(optim_checkpoint, os.path.join(args.output_dir, 'optim_checkpoint_latest.pt'))
    if sched_checkpoint: torch.save(sched_checkpoint, os.path.join(args.output_dir, 'sched_checkpoint_latest.pt'))

    # 2. save the last `max_records` number of checkpoints as sorted by avg auc
    tracker_path = os.path.join(args.output_dir, 'checkpoints_tracker.csv')
    tracker_header = ' '.join(['CheckpointId', 'Step', 'Loss', 'AvgAUC'])

    # 2a. load checkpoint stats from file
    old_data = None             # init and overwrite from records
    file_id = 0                 # init and overwrite from records
    lowest_auc = float('-inf')  # init and overwrite from records
    if os.path.exists(tracker_path):
        old_data = np.atleast_2d(np.loadtxt(tracker_path, skiprows=1))
        file_id = len(old_data)
        if len(old_data) == max_records: # remove the lowest-roc record and add new checkpoint record under its file-id
            lowest_auc_idx = old_data[:,3].argmin()
            lowest_auc = old_data[lowest_auc_idx, 3]
            file_id = int(old_data[lowest_auc_idx, 0])
            old_data = np.delete(old_data, lowest_auc_idx, 0)

    # 2b. update tracking data and sort by descending avg auc
    data = np.atleast_2d([file_id, args.step, checkpoint['eval_loss'], checkpoint['avg_auc']])
    if old_data is not None: data = np.vstack([old_data, data])
    data = data[data.argsort(0)[:,3][::-1]]  # sort descending by AvgAUC column

    # 2c. save tracker and checkpoint if better than what is already saved
    if checkpoint['avg_auc'] > lowest_auc:
        np.savetxt(tracker_path, data, delimiter=' ', header=tracker_header)
        torch.save(checkpoint, os.path.join(args.output_dir, 'best_checkpoints', 'checkpoint_{}.pt'.format(file_id)))


# --------------------
# Evaluation metrics
# --------------------
# def compute_metrics(outputs, targets, losses):
    
#     anatomy = outputs.shape[1]
#     diseases = outputs.shape[2]
#     fpr1, tpr1, aucs1, precision1, recall1 = {}, {}, {}, {}, {}
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         for j in range(diseases):
#             fpr, tpr, aucs, precision, recall, acc = {}, {}, {}, {}, {}, {}
#             for i in range(anatomy):
#                 acc[i] = accuracy_score(targets[:,i,j], outputs[:,i,j])
#                 fpr[i], tpr[i], _ = roc_curve(targets[:,i,j], outputs[:,i,j])
#                 aucs[i] = auc(fpr[i], tpr[i])
#                 precision[i], recall[i], _ = precision_recall_curve(targets[:,i,j], outputs[:,i,j])
#                 fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()
#             fpr1[j], tpr1[j], aucs1[j], precision1[j], recall1[j] = fpr, tpr, aucs, precision, recall

#     metrics = {'fpr': fpr1,
#                'tpr': tpr1,
#                'aucs': aucs1,
#                'precision': precision1,
#                'recall': recall1,
#                'loss': dict(enumerate(losses.mean(0).tolist()))}

#     return metrics

def compute_metrics(outputs, targets, losses):
    
    anatomy = outputs.shape[1]
    diseases = outputs.shape[2]
    fpr1, tpr1, aucs1, precision1, recall1, accs1 = {}, {}, {}, {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(diseases):
            fpr, tpr, aucs, precision, recall, accs = {}, {}, {}, {}, {}, {}
            for i in range(anatomy):
                # print(outputs)
                #print(targets[:,i,j])
                preds = outputs[:,i,j].sigmoid().round()
                #accuracy_metric.update(targets[:,i,j], outputs[:,i,j])
                # print(accuracy_metric.compute())
                #accs[i] = accuracy_metric.compute().item()
                accs[i] = accuracy_score(targets[:,i,j], preds)
                fpr[i], tpr[i], _ = roc_curve(targets[:,i,j], outputs[:,i,j])
                aucs[i] = auc(fpr[i], tpr[i])
                precision[i], recall[i], _ = precision_recall_curve(targets[:,i,j], outputs[:,i,j])
                fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()
            fpr1[j], tpr1[j], aucs1[j], precision1[j], recall1[j], accs1[j] = fpr, tpr, aucs, precision, recall, accs

    metrics = {'fpr': fpr1,
               'tpr': tpr1,
               'aucs': aucs1,
               'accuracy': accs1,
               'precision': precision1,
               'recall': recall1,
               'loss': dict(enumerate(losses.mean(0).tolist()))}

    return metrics

# --------------------
# Train and evaluate
# --------------------

def train_epoch(rank, model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, epoch, args):
    model.train()
    with tqdm(total=len(train_dataloader), desc='Step at start {}; Training epoch {}/{}'.format(args.step, epoch+1, args.n_epochs)) as pbar:
        for image_features, target in train_dataloader:
            
            # print("image_features shape:", image_features.size())
            inp_data = image_features.to(rank)
            target_data = target.to(rank)
            args.step += 1
            
            anatomy, logits = model(inp_data)
            # print("out", torch.isnan(out).any() )
            # loss = loss_fn(out, target_data) #.sum(1).sum(1).mean(0)
            loss = loss_fn(logits, target_data, reduction='none', alpha=args.focal_alpha, gamma=args.focal_gamma)
            # print("loss1", torch.isnan(loss).any() )
            # print("img_loss", loss.size())
            loss = torch.mean(loss)
            # print("mean_loss", loss.size())
            # model.parameters.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            pbar.set_postfix(loss = '{:.4f}'.format(loss.item()))
            pbar.update()

            # record
            if (args.step % args.log_interval == 0) and (rank==0):
                writer.add_scalar('train_loss', loss.item(), args.step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.step)

            # # evaluate and save on eval_interval
            # if args.step % args.eval_interval == 0:
            #     with torch.no_grad():
            #         model.eval()

            #         eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
            #         avg_aucs = {name: np.nanmean(list(tests.values())) for (name, tests) in eval_metrics['aucs'].items()}
                    
            #         writer.add_scalar('eval_loss', np.sum(list(eval_metrics['loss'].values())), args.step)
            #         for k, v in eval_metrics['aucs'].items():
            #             # print(v.items())
            #             for key, val in v.items():
            #                 writer.add_scalar('eval_auc_anatomy_{}_disease_{}'.format(key, k), val, args.step)

            #         # save model
            #         save_checkpoint(checkpoint={'global_step': args.step,
            #                                     'eval_loss': np.sum(list(eval_metrics['loss'].values())),
            #                                     'avg_auc': np.nanmean(list(avg_aucs.values())),
            #                                     'state_dict': model.state_dict()},
            #                         optim_checkpoint=optimizer.state_dict(),
            #                         sched_checkpoint=scheduler.state_dict() if scheduler else None,
            #                         args=args)

            #         # switch back to train mode
            #         model.train()

@torch.no_grad()
def evaluate(rank, model, dataloader, loss_fn, args):
    model.eval()
    targets, outputs, losses = [], [], []
    count = 0
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for image_features, target in dataloader:
            inp_data = image_features.to(rank)
            target_data = target.to(rank)
            count +=1

            with torch.no_grad():
                anatomy, logits = model(inp_data)
            loss = loss_fn(logits, target_data)
            
            outputs += [logits.cpu()]
            targets += [target]
            losses  += [loss.cpu()]

            pbar.update()

            # if count == 10:
            #     break
        
    # coco_data = pd.DataFrame(
    #         {'image_id': imageIDs,
    #         'objects': objectss,
    #         'output': torch.cat(outputss).tolist(),
    #         'target': torch.cat(targets).tolist()
    #         })
    # save_file_name = './DensenetModel.csv'
    # coco_data.to_csv(save_file_name, sep='\t', index=False)
    # print(coco_data.head(5))
    return torch.cat(outputs), torch.cat(targets), torch.cat(losses)

def evaluate_single_model(rank, model, dataloader, loss_fn, args):
    outputs, targets, losses = evaluate(rank, model, dataloader, loss_fn, args)
    return compute_metrics(outputs, targets, losses)

def evaluate_ensemble(rank, model, dataloader, loss_fn, args):
    checkpoints = [c for c in os.listdir(args.restore) \
                        if c.startswith('checkpoint') and c.endswith('.pt')]
    print('Running ensemble prediction using {} checkpoints.'.format(len(checkpoints)))
    outputs, losses = [], []
    for checkpoint in checkpoints:
        # load weights
        model_checkpoint = torch.load(os.path.join(args.restore, checkpoint), map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
        model.load_state_dict(model_checkpoint['state_dict'])
        del model_checkpoint
        # evaluate
        outputs_, targets, losses_ = evaluate(model, dataloader, loss_fn, args)
        outputs += [outputs_]
        losses  += [losses_]

    # take mean over checkpoints
    outputs  = torch.stack(outputs, dim=2).mean(2)
    losses = torch.stack(losses, dim=2).mean(2)

    return compute_metrics(outputs, targets, losses)

def train_and_evaluate(rank, model, train_dataloader, valid_dataloader, test_dataloader, loss_fn, optimizer, scheduler, writer, args):
    for epoch in range(args.n_epochs):
        # train
        train_epoch(rank, model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, epoch, args)

        # evaluate
        print('Evaluating...', end='\r')
        eval_metrics = evaluate_single_model(rank, model, test_dataloader, loss_fn, args)
        
        # save model
        avg_aucs = {name: np.nanmean(list(tests.values())) for (name, tests) in eval_metrics['aucs'].items()}
        if rank == 0:
            save_checkpoint(checkpoint={'global_step': args.step,
                                        'eval_loss': np.sum(list(eval_metrics['loss'].values())),
                                        'avg_auc': np.nanmean(list(avg_aucs.values())),
                                        'state_dict': model.state_dict()},
                            optim_checkpoint=optimizer.state_dict(),
                            sched_checkpoint=scheduler.state_dict() if scheduler else None,
                            args=args)
        print('Evaluate metrics @ step {}:'.format(args.step))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        # print('Recall:\n', pprint.pformat(eval_metrics['recall']))
        # print('Precision:\n', pprint.pformat(eval_metrics['precision']))
        
        if rank == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                writer.add_scalar('eval_loss', np.sum(list(eval_metrics['loss'].values())), args.step)
                for k, v in eval_metrics['aucs'].items():
                    # print(v.items())
                    for key, val in v.items():
                        writer.add_scalar('eval_auc_anatomy_{}_disease_{}'.format(key, k), val, args.step)

            # save eval metrics
            save_json(eval_metrics, 'eval_results_step_{}'.format(args.step), args)

        dist.barrier()
        if scheduler and (epoch+1) >= args.lr_warmup_steps: scheduler.step()
        
# --------------------
# Visualization
# --------------------

def plot_roc(metrics, args, filename, labels=diseases):
    fig, axs = plt.subplots(2, len(labels), figsize=(24,12))

    for i, (fpr, tpr, aucs, precision, recall, label) in enumerate(zip(metrics['fpr'].values(), metrics['tpr'].values(),
                                                                       metrics['aucs'].values(), metrics['precision'].values(),
                                                                       metrics['recall'].values(), labels)):
        # top row -- ROC
        axs[0,i].plot(fpr, tpr, label='AUC = %0.2f' % aucs)
        axs[0,i].plot([0, 1], [0, 1], 'k--')  # diagonal margin
        axs[0,i].set_xlabel('False Positive Rate')
        # bottom row - Precision-Recall
        axs[1,i].step(recall, precision, where='post')
        axs[1,i].set_xlabel('Recall')
        # format
        axs[0,i].set_title(label)
        axs[0,i].legend(loc="lower right")

    plt.suptitle(filename)
    axs[0,0].set_ylabel('True Positive Rate')
    axs[1,0].set_ylabel('Precision')

    for ax in axs.flatten():
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'plots', filename + '.png'), pad_inches=0.)
    plt.close()

# --------------------
# Main
# --------------------
def main(rank, args, world_size):
    if world_size > 1:
        setup(rank, world_size)

    args.output_dir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/AnaXnet-Output-focal-25-1/'
    n_classes = len(diseases)

    # overwrite args from config
    if args.load_config: args.__dict__.update(load_json(args.load_config))

    # set up output folder
    # if not args.output_dir:
    #     if args.restore: raise RuntimeError('Must specify `output_dir` argument')
    #     args.output_dir: args.output_dir = os.path.join('results', time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    # make new folders if they don't exist
    writer = SummaryWriter(logdir=args.output_dir)  # creates output_dir
    if rank == 0:
        if not os.path.exists(os.path.join(args.output_dir, 'vis')): os.makedirs(os.path.join(args.output_dir, 'vis'))
        if not os.path.exists(os.path.join(args.output_dir, 'plots')): os.makedirs(os.path.join(args.output_dir, 'plots'))
        if not os.path.exists(os.path.join(args.output_dir, 'best_checkpoints')): os.makedirs(os.path.join(args.output_dir, 'best_checkpoints'))

        # save config
        if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)
        writer.add_text('config', str(args.__dict__))

    # device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    
    # load data Generators
    print("loading training data...")
    path = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/coco/updated/'  # "/home/agun/mimic/dataset/VG/"
    train_dataloader, train_dataset = get_loader(rank, root_folder=path,mode='train',num_workers=world_size,batch_size=args.batch_size, shuffle=True, world_size=world_size)
    # valid_dataloader, valid_dataset = get_loader(rank, root_folder=path,mode='valid',num_workers=world_size,batch_size=args.batch_size, shuffle=False, world_size=world_size)
    valid_dataloader = None
    test_dataloader, test_dataset = get_loader(rank, root_folder=path,mode='test',num_workers=world_size,batch_size=args.batch_size, shuffle=False, world_size=world_size)
    
    # datasets= AnaxnetDataset(os.path.join(path, 'new_train.csv'))
    # # print(len(datasets))
    # train_set, val_set = torch.utils.data.random_split(datasets, [50000, 14003])
    # train_dataloader = DataLoader(
    #     dataset=train_set,
    #     batch_size=16,
    #     num_workers=4,shuffle=False,pin_memory=True)
    # valid_dataloader = DataLoader(
    #     dataset=val_set,
    #     batch_size=16,
    #     num_workers=0,shuffle=False,pin_memory=True)
    # test_dataloader = valid_dataloader
    # # valid_dataset = AnaxnetDataset(os.path.join(path, 'new_train.csv'))
    # # test_dataset = AnaxnetDataset(os.path.join(path, 'new_train.csv'))

   

    print('Train data length: ', len(train_dataloader))
    # print('Valid data length: ', len(valid_dataloader))
    print('Test data length: ', len(test_dataloader))

    # load model
    model = AnaXnetGCN(
                num_classes = n_classes, 
                in_channel1=300, 
                in_channel2=1024)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids = [0,4])
    model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_decay_factor, verbose=True)
    scheduler = None
    
    print('Loaded {} (number of parameters: {:,}; weights trained to step {})'.format(
        model._get_name(), sum(p.numel() for p in model.parameters()), args.step))

    if args.restore and os.path.isfile(args.restore):  # restore from single file, else ensemble is handled by evaluate_ensemble
        print('Restoring model weights from {}'.format(args.restore))
        model_checkpoint = torch.load(args.restore, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
        model.load_state_dict(model_checkpoint['state_dict'])
        args.step = model_checkpoint['global_step']
        del model_checkpoint
        # if training, load optimizer and scheduler too
        if args.train:
            print('Restoring optimizer.')
            optim_checkpoint_path = os.path.join(os.path.dirname(args.restore), 'optim_' + os.path.basename(args.restore))
            optimizer.load_state_dict(torch.load(optim_checkpoint_path, map_location={'cuda:%d' % 0: 'cuda:%d' % rank}))
            if scheduler:
                print('Restoring scheduler.')
                sched_checkpoint_path = os.path.join(os.path.dirname(args.restore), 'sched_' + os.path.basename(args.restore))
                scheduler.load_state_dict(torch.load(sched_checkpoint_path, map_location={'cuda:%d' % 0: 'cuda:%d' % rank}))

    # load data
    if args.restore:
        # load pretrained flag from config -- in case forgotten e.g. in post-training evaluation
        # (images still need to be normalized if training started on an imagenet pretrained model)
        args.pretrained = load_json(os.path.join(args.output_dir, 'config.json'))['pretrained']

    # setup loss function for train and eval
    # loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(rank)

    loss_fn = sigmoid_focal_loss
    if args.train:
        train_and_evaluate(rank, model, train_dataloader, valid_dataloader, test_dataloader, loss_fn, optimizer, scheduler, writer, args)
    
    if args.evaluate_single_model:
        eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
        print('Evaluate metrics -- \n\t restore: {} \n\t step: {}:'.format(args.restore, args.step))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        save_json(eval_metrics, 'eval_results_step_{}'.format(args.step), args)
    
    if world_size > 1:
        cleanup()

if __name__ == '__main__':
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs")
    world_size = n_gpus

    if world_size > 1:
        mp.spawn(main, args=(args, world_size,), nprocs=world_size, join=True)
    else:
        rank = 0
        main(rank, args, world_size)
    