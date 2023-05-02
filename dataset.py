import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np 
import pandas as pd
from transformers import AutoTokenizer

class AnaxnetDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, filepath):
        """
        :param data_folder: folder where data files are stored
        :param transform: image transform pipeline
        """

        # Open word2vec embedding file
        self.rootdir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/feature_extraction_final/' # '/home/agun/mimic/dataset/VG/FeatureData/' 
        self.data = pd.read_csv(filepath, sep='\t')

        # Total number of datapoints
        self.dataset_size = len(self.data)


    def __getitem__(self, i):
        jsonFile = self.data['image_id'][i] + '.json'
        filepath = os.path.join(self.rootdir, jsonFile)
        with open(filepath, 'r') as j:
            jsonData = json.load(j)
        # imageID = jsonData['image_id']
        # objects = jsonData['objects']
        image_features = torch.FloatTensor(np.array(jsonData['features']))
        feat_sum = image_features.sum(axis=1)
        missing_objs_idx = np.where(feat_sum==0)[0]
        target = np.array(jsonData['target'])
        if len(missing_objs_idx) > 0:
            target[missing_objs_idx,:] = 0
        target = torch.Tensor(target)

        # t = np.array(jsonData['target'])
        # if t.shape[0] < 18:
        #     feat_sum = image_features.sum(axis=1)
        #     # print("feat_sum", feat_sum.size())
        #     # missing_objs = df['features'].apply(lambda x: 1 if sum(x)==0 else 0).values
        #     missing_objs_idx = np.where(feat_sum==0)[0]
        #     # print("objs_idx", objs_idx)
        #     target = torch.Tensor(np.delete(t,missing_objs_idx, axis=0))
        #     # target = torch.Tensor(t[missing_objs_idx,:])
        # elif t.shape[0] > 18:
        #     target = torch.Tensor(np.zeros([18,9]))
        # else:
        #     target = torch.Tensor(t)
        target = torch.Tensor(np.array(jsonData['target']))
        
        #fix target being different
        # if target.size()[0] != 18:
        #     print(target.size())
 
        #     target = torch.Tensor(np.zeros([18,9]))
        # print(target.size())
        # print('imageID', imageID.size())
        # print('objects', len(objects))
        # print('image_features', image_features.size())
        return image_features, target # (imageID, objects, image_features), target

    def __len__(self):
        return self.dataset_size

    
class BimodalDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, filepath, tokenizer):
        """
        :param data_folder: folder where data files are stored
        :param transform: image transform pipeline
        """

        # Open word2vec embedding file
        self.rootdir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/mm_feature_extraction_final_fulltext_txtfeats/' # '/home/agun/mimic/dataset/VG/FeatureData/' 
        self.data = pd.read_csv(filepath, sep='\t')

        # Total number of datapoints
        self.dataset_size = len(self.data)

        self.tokenizer = tokenizer


    def __getitem__(self, i):
        jsonFile = self.data['image_id'][i] + '.json'
        filepath = os.path.join(self.rootdir, jsonFile)
        with open(filepath, 'r') as j:
            jsonData = json.load(j)
        # imageID = jsonData['image_id']
        # objects = jsonData['objects']
        image_features = torch.FloatTensor(np.array(jsonData['features']))
        feat_sum = image_features.sum(axis=1)
        missing_objs_idx = np.where(feat_sum==0)[0]
        target = np.array(jsonData['target'])
        if len(missing_objs_idx) > 0:
            target[missing_objs_idx,:] = 0
        target = torch.Tensor(target)
        # report_text = ' '.join(jsonData['report_text'])
        # report_tokens = self.tokenizer(report_text,padding='max_length', max_length = 512, 
        #                                 truncation=True,return_tensors="pt") 
        text_features = torch.FloatTensor(np.array(jsonData['radbert_features']))
        # t = np.array(jsonData['target'])
        # if t.shape[0] < 18:
        #     feat_sum = image_features.sum(axis=1)
        #     # print("feat_sum", feat_sum.size())
        #     # missing_objs = df['features'].apply(lambda x: 1 if sum(x)==0 else 0).values
        #     missing_objs_idx = np.where(feat_sum==0)[0]
        #     # print("objs_idx", objs_idx)
        #     target = torch.Tensor(np.delete(t,missing_objs_idx, axis=0))
        #     # target = torch.Tensor(t[missing_objs_idx,:])
        # elif t.shape[0] > 18:
        #     target = torch.Tensor(np.zeros([18,9]))
        # else:
        #     target = torch.Tensor(t)
        target = torch.Tensor(np.array(jsonData['target']))
        
        #fix target being different
        # if target.size()[0] != 18:
        #     print(target.size())
 
        #     target = torch.Tensor(np.zeros([18,9]))
        # print(target.size())
        # print('imageID', imageID.size())
        # print('objects', len(objects))
        # print('image_features', image_features.size())
        #print(report_tokens['input_ids'].size())
        # return image_features, report_tokens, target, jsonData['image_id']
        return image_features, text_features, target, jsonData['image_id'] # (imageID, objects, image_features), target

    def __len__(self):
        return self.dataset_size


