from torch.nn import Parameter
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from model.anaxnet import AnaXnetGCN
from model.radbert import RadBERT

# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

#     def __init__(self, in_features, out_features, bias=False):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(1, 1, out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj):
#         support = torch.matmul(input, self.weight) #18x1024
#         output = torch.matmul(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'


# class AnaXnetGCN(nn.Module):
#     def __init__(self, num_classes, anatomy_size=18, in_channel1=300, in_channel2=1024):
#         super(AnaXnetGCN, self).__init__()
#         anatomy_out = 1024
#         self.num_classes = num_classes

#         self.anatomy_gc1 = GraphConvolution(in_channel2, 512)
#         self.anatomy_gc2 = GraphConvolution(512, 1024)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim = -1)

#         # self.fc = nn.Linear((findings_out+anatomy_out), num_classes)
#         self.fc = nn.Sequential(
#             nn.LayerNorm(anatomy_out),
#             nn.AdaptiveAvgPool2d((anatomy_size, anatomy_out)),
#             nn.Linear(anatomy_out, num_classes, bias=False)
#         )

#         #anatomy adjacency matrix
#         anatomy_inp_name = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/coco/anatomy_matrix.csv' # "/home/agun/mimic/dataset/VG/anatomy_matrix.csv"
#         anatomy_inps = pd.read_csv(anatomy_inp_name, sep='\t')
#         self.anatomy_inp_tensor = Parameter(torch.FloatTensor(anatomy_inps.values))


#     def findings_gcn(self, feature, findings_inp):
#         findings_inp = findings_inp[0]
#         adj = self.findings_inp_tensor.detach() 
#         x = self.findings_gc1(findings_inp, adj)
#         x = self.relu(x)
#         x = self.findings_gc2(x, adj)

#         x = x.transpose(0, 1)
#         x = torch.matmul(feature, x)
#         return x

#     def anatomy_gcn(self, feature):
#         anatomy_inp = feature#anatomy_inp[0]
#         adj = self.anatomy_inp_tensor.detach() 
#         x = self.anatomy_gc1(anatomy_inp, adj)
#         x = self.relu(x)
#         x = self.anatomy_gc2(x, adj)
#         # print("x shape", x.size())
#         x = x.transpose(1, 2)
#         # print("x.T shape", x.size())
#         x = torch.matmul(feature, x)
#         x = self.softmax(x)
#         x = torch.matmul(x, anatomy_inp)
#         return x

#     def forward(self, feature):
#         anatomy = self.anatomy_gcn(feature)
#         anatomy = anatomy.add(feature)
#         # anatomy = torch.cat((anatomy, feature), 2)
#         # print("anatomy shape", anatomy.size())
#         logits = self.fc(anatomy)
#         return anatomy, logits

#     def get_config_optim(self, lr, lrp):
#         return [
#                 {'params': self.findings_gc1.parameters(), 'lr': lr},
#                 {'params': self.findings_gc2.parameters(), 'lr': lr},
#                 {'params': self.anatomy_gc1.parameters(), 'lr': lr},
#                 {'params': self.anatomy_gc2.parameters(), 'lr': lr},
#                 ]

class BimodalModel(nn.Module):
    def __init__(self, n_classes, embedding_size=1024, 
            input_size_img=1024, bottleneck_size_img=512,  
            input_size_text=768, bottleneck_size_text=384):
        super(BimodalModel, self).__init__()
        self.img_model = AnaXnetGCN(
                num_classes = n_classes, 
                input_size=input_size_img, 
                bottleneck_size=bottleneck_size_img,
                embedding_size=embedding_size)
        self.text_model = AnaXnetGCN(
                num_classes = n_classes, 
                input_size=input_size_img, 
                bottleneck_size=bottleneck_size_img,
                embedding_size=embedding_size)
        self.linear = nn.Linear(input_size_text, embedding_size)

    def forward(self, input_img, input_text):
        embedding_img, logits_img = self.img_model(input_img)
        text_features = self.linear(input_text)
        embedding_text, logits_text = self.text_model(text_features)

        return embedding_img, logits_img, embedding_text, logits_text


# class BimodalModel(nn.Module):
#     def __init__(self, n_classes, embedding_size):
#         super(BimodalModel, self).__init__()
#         self.img_model = AnaXnetGCN(
#                 num_classes = n_classes, 
#                 in_channel1=300, 
#                 in_channel2=embedding_size)
#         self.text_model = RadBERT("StanfordAIMI/RadBERT", n_classes, dropout=0.1)

#     def forward(self, input_img, text_input_ids, text_mask):
#         embedding_img, logits_img = self.img_model(input_img)
#         embedding_text, logits_text = self.text_model(text_input_ids, text_mask)

#         return embedding_img, logits_img, embedding_text, logits_text