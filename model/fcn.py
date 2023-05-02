import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd

class FCN(nn.Module):
    def __init__(self, num_classes, anatomy_size=18, input_size=1024, bottleneck_size=512, embedding_size=1024):
        super(FCN, self).__init__()
        # anatomy_out = 1024
        self.num_classes = num_classes

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim = -1)

        # self.fc = nn.Linear((findings_out+anatomy_out), num_classes)
        self.fc = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.AdaptiveAvgPool2d((anatomy_size, embedding_size)),
            nn.Linear(embedding_size, num_classes, bias=False)
        )

    def forward(self, feature):
        logits = self.fc(feature)
        return logits