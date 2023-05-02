import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, dataset
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score
from torcheval.metrics import MulticlassAccuracy
from torchvision.ops import sigmoid_focal_loss

from tqdm import tqdm
import os
import pprint
import argparse
import time
import json
import warnings
from collections import OrderedDict

from dataset import AnaxnetDataset
from model.anaxnet import AnaXnetGCN
from model.bimodal import BimodalModel
from model.fcn import FCN
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
# action
parser.add_argument('--model_path', type=str, default=None, help='Path to model directory')
parser.add_argument('--focal_alpha', type=float, default=0.25, help='Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore.')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples')


diseases = ["Lung opacity", "Pleural effusion", "Atelectasis", "Enlarged cardiac silhouette", "Pulmonary edema/hazy opacity", 
            "Pneumothorax", "Consolidation" , "Fluid overload/heart failure", "Pneumonia"]

organs = ["right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
        "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
        "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
        "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette", "trachea"]



# --------------------
# Data IO
# --------------------

def save_json(data, dest_path):
    with open(dest_path, 'w') as f:
        json.dump(data, f, indent=4)

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
    dataset= AnaxnetDataset(os.path.join(root_folder, 'new_{}.csv'.format(mode))) # 'sample_test.csv')) # 

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

def compute_metrics_max_pred(outputs, targets, losses):
    anatomy = outputs.shape[1]
    diseases = outputs.shape[2]
    fpr1, tpr1, aucs1, precision1, recall1, accs1 = {}, {}, {}, {}, {}, {}
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    outputs = torch.sigmoid(outputs)
    #outputs = torch.softmax(outputs, dim=1)
    print("output", outputs.size())
    max_pred_idx = outputs.argmax(dim=1)
    print("maxidx", max_pred_idx.unsqueeze(0).size())
    # print(targets)
    new_targets = torch.gather(targets, 1, max_pred_idx.unsqueeze(0)).squeeze(0)
    print("new_targets",new_targets.size())
    max_preds, _ = outputs.max(dim=1)
    print(max_preds)
    for j in range(diseases):
        preds = max_preds[:,j].round()
        # print("preds", preds.size())
        #print(targets[:,:,j])
        print("preds",preds)
        print(new_targets[:,j])
        accs1[j] = accuracy_score(new_targets[:,j], preds)
        #print(new_targets[:,j])
        # print("accs1", type(accs1[j]))
        fpr1[j], tpr1[j], _ = roc_curve(new_targets[:,j], preds)
        # print("fpr1", type(fpr1[j]))
        # print("tpr1", type(tpr1[j]))
        aucs1[j] = auc(fpr1[j], tpr1[j])
        # print("aucs1", type(aucs1[j]))
        precision1[j], recall1[j], _ = precision_recall_curve(new_targets[:,j], preds)
        # print("precision1", type(precision1[j]))
        # print("recall1", type(recall1[j]))
        fpr1[j], tpr1[j], precision1[j], recall1[j] = fpr1[j].tolist(), tpr1[j].tolist(), precision1[j].tolist(), recall1[j].tolist()
        #     #print(outputs[:,max_pred_idx,j].size())
            
        #     # print(max_pred_idx)
        #     fpr, tpr, aucs, precision, recall, accs = {}, {}, {}, {}, {}, {}
        #     preds = outputs[:, max_pred_idx, j].round()

            # for i in tqdm(max_pred_idx.tolist()):
            #     preds = outputs[:,i,j].round()

            #     accs[i] = accuracy_score(targets[:,i,j], preds)
            #     fpr[i], tpr[i], _ = roc_curve(targets[:,i,j], outputs[:,i,j])
            #     aucs[i] = auc(fpr[i], tpr[i])
            #     precision[i], recall[i], _ = precision_recall_curve(targets[:,i,j], outputs[:,i,j])
            #     fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()
            # fpr1[j], tpr1[j], aucs1[j], precision1[j], recall1[j], accs1[j] = fpr, tpr, aucs, precision, recall, accs

        # for j in range(diseases):
        #     print(outputs[:,:,j].size())
            
        #     # print(max_pred_idx)
        #     fpr, tpr, aucs, precision, recall, accs = {}, {}, {}, {}, {}, {}
        #     preds = outputs[:, max_pred_idx, j].round()

        #     for i in tqdm(max_pred_idx.tolist()):
        #         preds = outputs[:,i,j].round()

        #         accs[i] = accuracy_score(targets[:,i,j], preds)
        #         fpr[i], tpr[i], _ = roc_curve(targets[:,i,j], outputs[:,i,j])
        #         aucs[i] = auc(fpr[i], tpr[i])
        #         precision[i], recall[i], _ = precision_recall_curve(targets[:,i,j], outputs[:,i,j])
        #         fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()
        #     fpr1[j], tpr1[j], aucs1[j], precision1[j], recall1[j], accs1[j] = fpr, tpr, aucs, precision, recall, accs
    print("losses ",losses.size())
    metrics = {'fpr': fpr1,
               'tpr': tpr1,
               'aucs': aucs1,
               'accuracy': accs1,
               'precision': precision1,
               'recall': recall1,
               'loss': dict(enumerate(losses.mean(0).tolist()))}

    return metrics

def compute_metrics(outputs, targets, losses):
    anatomy = outputs.shape[1]
    diseases = outputs.shape[2]
    fpr1, tpr1, aucs1, precision1, recall1, accs1 = {}, {}, {}, {}, {}, {}
    accuracy_metric = MulticlassAccuracy()#average='macro', num_classes=diseases)
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
                # logits = model(inp_data)
                embedding_img, logits = model.img_model(inp_data)
            loss = loss_fn(logits, target_data, reduction='none', alpha=args.focal_alpha, gamma=args.focal_gamma)
            
            outputs += [logits.cpu()]
            targets += [target]
            losses  += [loss.cpu()]

            pbar.update()

    return compute_metrics(torch.cat(outputs), torch.cat(targets), torch.cat(losses))
     



if __name__=="__main__":
    args = parser.parse_args()
    rank = 1
    # model_path = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/AnaXnet-Output-focal-25-2/'
    n_classes = len(diseases)
    path = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/coco/updated/'
    test_dataloader, test_dataset = get_loader(0, root_folder=path,mode='test',num_workers=0,batch_size=16, world_size=1)
    print('Test data length: ', len(test_dataloader))

    model = BimodalModel(n_classes = n_classes)
    # model = FCN(num_classes = n_classes)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids = [0,4])
    model.to(rank)

    print('Restoring model weights from {}'.format(os.path.join(args.model_path, 'checkpoint_latest.pt')))
    model_checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint_latest.pt'))
    # state_dict = torch.load(model_path)
# create new OrderedDict that does not contain `module.`
# 
    # new_state_dict = OrderedDict()
    # for k, v in model_checkpoint['state_dict'].items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    model.load_state_dict(model_checkpoint['state_dict'])
    # model.load_state_dict(new_state_dict)

    step = model_checkpoint['global_step']
    del model_checkpoint

    print('Loaded {} (number of parameters: {:,}; weights trained to step {})'.format(
        model._get_name(), sum(p.numel() for p in model.parameters()), step))
    
    # loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(2)
    loss_fn = sigmoid_focal_loss

    eval_metrics = evaluate(rank, model, test_dataloader, loss_fn, args)
    #print(eval_metrics)
    output_path = os.path.join(args.model_path, 'eval_metrics_anaxnet_focal_25_1.json')
    save_json(eval_metrics, output_path)