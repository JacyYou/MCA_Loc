import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.utils.data as data

from Encoder import *

script_path = os.path.split(os.path.realpath(__file__))[0] + "/"

##############  Hyperparameter  ##############
MCALoc_config = {
    'node_input_dim': 1024,
    'hidden_dim': 256,
    'layer': 2,
    'dropout': 0.2
}

##############  Model  ##############
class MCALoc(nn.Module):
    def __init__(self, node_input_dim, hidden_dim, num_layers, dropout, augment_eps=0.02):
        super(MCALoc, self).__init__()
        self.augment_eps = augment_eps
        self.task_list = ["Cell Junctions", "Cytoskeleton", "Cytosol", "Endoplasmic reticulum", "Golgi apparatus", "Mitochondrion",
                        "Nuclear speckles", "Nuclear membrane", "Nucleolus", "Nucleoplasm", "Plasma membrane", "Vesicles"]
        self.Encoder = ImprovedLightAttention(emb_dim=node_input_dim, final_dim=hidden_dim)
        self.FC_1 = self.clf_head = nn.Linear(256, len(self.task_list))

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, h_V, mask):
        # Data augmentation

        if self.training and self.augment_eps > 0:
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)
        pooled, attention = self.Encoder(h_V, mask)
        output = self.clf_head(pooled)

        return output


##############  DataLoader  ##############
class MCALoc_Dataset(data.Dataset):
    def __init__(self, ID_list, labels, outpath):
        super(MCALoc_Dataset, self).__init__()
        self.IDs = ID_list
        self.labels = {ID: labels[ID] for ID in self.IDs}  
        self.prottrans_path = outpath + "ProtTrans/{}.tensor"


    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        node_feat = self._featurize(idx)
        Label = self.labels[self.IDs[idx]]  
        return node_feat, Label

    def _featurize(self, idx):
        name = self.IDs[idx]
        with torch.no_grad():
            embedding = torch.load(self.prottrans_path.format(name))[0:850]

        return embedding

