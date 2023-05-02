import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset, dataset
from transformers import AutoTokenizer

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
from dataset import AnaxnetDataset, BimodalDataset
from model.anaxnet import AnaXnetGCN
from model.radbert import RadBERT
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

def get_loader(rank, root_folder, mode, tokenizer, batch_size=16,num_workers=0,shuffle=False,pin_memory=True, world_size=1):
    assert mode in ['train', 'valid', 'test']
    dataset= BimodalDataset(os.path.join(root_folder, 'new_{}.csv'.format(mode)), tokenizer=tokenizer)  # 'sample_{}.csv'.format(mode)), tokenizer=tokenizer) # 

    if batch_size % world_size != 0:
        raise Exception("Batch size must be a multiple of the number of workers")

    batch_size = batch_size // world_size

    print(f"World size: {world_size}, setting effective batch size to {batch_size}. Should be batch size / num input gpus.")

    if world_size > 1:
        # print('here1')
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
        # print('here2')
        loader = InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True
            )
        # loader = DataLoader(
        #     dataset=dataset,
        #     batch_size=batch_size,
        #     num_workers=num_workers,
        #     sampler=sampler,
        #     pin_memory=True
        #     )
        # print('here3')
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

def train_epoch(rank, model, train_dataloader, epoch, args):
    model.train()
    with tqdm(total=len(train_dataloader), desc='Step at start {}; Training epoch {}/{}'.format(args.step, epoch+1, args.n_epochs)) as pbar:
        for image_features, report_tokens, target, image_id in train_dataloader:
            text_mask = report_tokens['attention_mask'].to(rank)
            # print('text_input_id0', report_tokens['input_ids'].size())
            text_input_id = report_tokens['input_ids'].to(rank)
            # text_input_id = report_tokens['input_ids'].squeeze(1).to(rank)
            # print('text_mask', text_mask.size())
            # print('text_input_id', text_input_id.size())
            # print("image_features shape:", image_features.size())
            # inp_data = image_features.to(rank)
            #print("text_input_id", text_input_id.size())

            embeddings = model(text_input_id, text_mask)
            embeddings_copy = embeddings.clone()
            for i in range(17):
                embeddings_copy = torch.cat([embeddings_copy, embeddings.clone()], dim=1)
            for i in range(embeddings_copy.size()[0]):
                fp = os.path.join(args.feature_dir, image_id[i]+'.json')
                data = load_json(fp)
                data['radbert_features'] = embeddings_copy[i,:,:].tolist()

                save_json(data, image_id[i], args)

            pbar.update()


def train_and_evaluate(rank, model, dataloader, args):
    for epoch in range(args.n_epochs):
        # train
        train_epoch(rank, model, dataloader, epoch, args)
        

        # dist.barrier()
        # if scheduler and (epoch+1) >= args.lr_warmup_steps: scheduler.step()

# --------------------
# Main
# --------------------
def main(rank, args, world_size):
    if world_size > 1:
        setup(rank, world_size)

    args.output_dir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/mm_feature_extraction_final_fulltext_txtfeats/'
    args.feature_dir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/mm_feature_extraction_final_fulltext/'
    n_classes = len(diseases)

    # overwrite args from config
    

    tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/RadBERT", use_fast=True)
    
    # load data Generators
    print("loading data...")
    path = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/coco/updated/'  # "/home/agun/mimic/dataset/VG/"
    dataloader, train_dataset = get_loader(rank, root_folder=path,mode='train', tokenizer=tokenizer, num_workers=world_size,batch_size=args.batch_size, shuffle=True, world_size=world_size)
   

    print('data length: ', len(dataloader))


    model = RadBERT("StanfordAIMI/RadBERT", n_classes, dropout=0.1)

    for param in model.base_model.parameters():
        param.requires_grad = False

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids = [0,4])
    model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)



    if args.train:
        train_and_evaluate(rank, model, dataloader, args)
    
   
    
    if world_size > 1:
        cleanup()

if __name__ == '__main__':
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs")
    world_size = n_gpus
    args.n_epochs = 1
    if world_size > 1:
        mp.spawn(main, args=(args, world_size,), nprocs=world_size, join=True)
    else:
        rank = 0
        main(rank, args, world_size)