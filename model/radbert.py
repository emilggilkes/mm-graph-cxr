import torch
from torch import nn
from transformers import AutoConfig, AutoModel, BertModel
import numpy as np
import copy

class RadBERT(nn.Module):
  def __init__(self, base_mod_name, NUM_CLASSES, dropout=0.1):
    super(RadBERT, self).__init__()
    self.config= AutoConfig.from_pretrained(base_mod_name)
    self.base_model = BertModel.from_pretrained(base_mod_name, config = self.config, add_pooling_layer=False)
    
    self.pooler = nn.Linear(768, 1024, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.tanh = nn.Tanh()
    self.classifier = nn.Linear(1024, NUM_CLASSES, bias=True)
    # self.pooler = nn.AdaptiveAvgPool1d(768)
    # self.relu = nn.ReLU()
    
    
  def forward(self, inputs, attention_mask):
    encoder_out = []
    for i in range(1):
      bert_out = self.base_model(inputs[:,i,:], attention_mask=attention_mask[:,i,:])
      # print(bert_out.keys())
      # pooler_out = self.pooler(bert_out['last_hidden_state'])
      mean_state = torch.mean(bert_out['last_hidden_state'], dim=1)
      # encoder_out.append(bert_out['last_hidden_state'])
      encoder_out.append(mean_state)
      # print(pooler_out.size())

    last_hidden_states = torch.permute(torch.stack(encoder_out, dim=0), (1,0,2))
    # pooler_out = self.pooler(last_hidden_states)
    #   last_hidden_states = torch.stack(encoder_out, dim=0)
    # out = self.tanh(pooler_out)
    # out = self.dropout(out)
    # logits = self.classifier(out)
    # return pooler_out, logits
    return last_hidden_states

# class RadBERT(nn.Module):
#   def __init__(self, base_mod_name, NUM_CLASSES, dropout=0.1):
#     super(RadBERT, self).__init__()
#     self.config= AutoConfig.from_pretrained(base_mod_name)
#     self.base_model = BertModel.from_pretrained(base_mod_name, config = self.config)
#     #self.dense = nn.Linear(768, 768, bias=False)
#     self.dropout = nn.Dropout(dropout)
#     self.classifier = nn.Linear(768, NUM_CLASSES, bias=True)
#     # self.relu = nn.ReLU()
    
#   def forward(self, inputs, attention_mask):
#     encoder_out = []
#     for i in range(18):
#       bert_out = self.base_model(inputs[:,i,:], attention_mask=attention_mask[:,i,:])
#       encoder_out.append(bert_out['pooler_output'])

#     pooler_out = torch.permute(torch.stack(encoder_out, dim=0), (1,0,2))

#     #out = self.dense(out)
#     out = self.dropout(pooler_out)
#     logits = self.classifier(out)
#     return pooler_out, logits
# class RadBERT(nn.Module):
#   def __init__(self, base_mod_name, NUM_CLASSES, dropout=0.1):
#     super(RadBERT, self).__init__()
#     self.config= AutoConfig.from_pretrained(base_mod_name)
#     bert = BertModel.from_pretrained(base_mod_name, config = self.config)

#     embedding_cpy = copy.deepcopy(bert.embeddings)
#     encoder_cpy = copy.deepcopy(bert.encoder)

#     self.base_model = nn.ModuleDict({'embeddings':embedding_cpy,'encoder':encoder_cpy})
    
#     self.dense = nn.Linear(768, 768, bias=True)
#     self.tanh = nn.Tanh()
#     self.dropout = nn.Dropout(dropout)
#     self.classifier = nn.Linear(768, NUM_CLASSES, bias=True)
#     # self.relu = nn.ReLU()
    
#   def forward(self, inputs, attention_mask):
#     encoder_out = []
#     for i in range(18):
#       bert_out = self.base_model(inputs[:,i,:], attention_mask=attention_mask[:,i,:])
#       print(bert_out.keys())
#       encoder_out.append(bert_out['encoder'])

#     out = torch.permute(torch.stack(encoder_out, dim=0), (1,0,2))

#     out = self.dense(out)
#     out = self.tanh(out)
#     out = self.dropout(out)
#     logits = self.classifier(out)
#     return out, logits