# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:50:57 2021

@author: sswanson7
"""

import torch
import torch.nn as nn
from EmbedConv import EmbedConv
import numpy as np

class ImageLSTMNetwork(nn.Module):
    def __init__(self, 
                 entity_size=[1,28,28],
                 output_dim=1,
                 embed_type='conv',
                 embed_features=20,
                 activation=nn.Tanh()):

        super().__init__()
        self.entity_size = entity_size
        self.embed_features = embed_features

        if embed_type == 'conv': 
            self.embedding = EmbedConv(image_size=entity_size,
                                                    stride=1,
                                                    padding=1,
                                                    kernel_size=3,
                                                    channels=3,
                                                    p_dropout=0.1,
                                                    embed_features=embed_features
                                                    )
        else:
            self.embedding = nn.Embedding(entity_size[0], embed_features)

        self.recurrent = nn.LSTM(embed_features, embed_features, batch_first=True)

        self.post_layers = nn.Sequential(
                                nn.Linear(embed_features, embed_features*2),
                                activation,
                                nn.Linear(embed_features*2, output_dim)
                            )


    def forward(self, x, mask):
        # Embedding
        bs, set_size, d, h, w = x.shape
        x = x.view(bs*set_size, d, h, w)
        x = self.embedding(x).view(bs,set_size,-1)

        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x,
                        lengths = mask.sum(axis=1,dtype=torch.int),
                        batch_first=True,
                        enforce_sorted=False)
        packed_out, (hidden, cell) = self.recurrent(packed_x)
        padded_out, lengths_out = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = padded_out[range(bs),lengths_out-1,:]
        out = self.post_layers(out)
        return out


    def compute_loss(self, data, masks, output):
        pred = self(data, masks)
        loss = ((pred-output)**2).mean()
        return loss