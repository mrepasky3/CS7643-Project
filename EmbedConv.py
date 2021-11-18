# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:11:08 2021

@author: sswanson7
"""
from torch import nn

class EmbedConv(nn.Module):
    def __init__(self,
                 image_size=[2,28,28],
                 stride=1,
                 padding=1,
                 kernel_size=3,
                 channels=3,
                 p_dropout=0.1,
                 embed_features=10
                 ):
        super().__init__()
        
        d1, w1, h1 = image_size
        k = channels
        
        self.layer1 = nn.Sequential(
                            nn.Conv2d(d1, k, kernel_size=kernel_size, stride=stride, padding=padding),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, padding=1, stride=1, dilation=1),
                            nn.Dropout(p=p_dropout)
                        )
                            
        w2 = (w1-kernel_size+2*padding)/stride + 1
        h2 = (h1-kernel_size+2*padding)/stride + 1
        d2 = k
        
        self.layer2 = nn.Sequential(
                            nn.Conv2d(d2, k, kernel_size=kernel_size, stride=stride, padding=padding),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, padding=1, stride=1, dilation=1),
                            nn.Dropout(p=p_dropout)
                        )
        
        w3 = (w2-kernel_size+2*padding)/stride + 1
        h3 = (h2-kernel_size+2*padding)/stride + 1
        d3 = k
        
        self.layer3 = nn.Sequential(
                            nn.Conv2d(d3, k, kernel_size=kernel_size, stride=stride, padding=padding),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, padding=1, stride=1, dilation=1),
                            nn.Dropout(p=p_dropout)
                        )
        
        self.ff1 = nn.Linear(int(d3*w3*h3), embed_features*2)
        self.ff2 = nn.Linear(embed_features*2, embed_features)
        self.postFF = nn.Sequential( self.ff1, nn.ReLU(), self.ff2, nn.ReLU())
        
        # print('d1, w1, h1 ',d1,w1,h1)
        # print('d2, w2, h2 ',d2,w2,h2)
        # print('d3, w3, h3 ',d3,w3,h3)
    
    def forward(self, inpt):
        bs, d1, w1, h1 = inpt.shape
        # print('bs, d1, w1, h1 ',bs, d1,w1,h1)
        
        x = self.layer1(inpt)
        # bs,d2,w2,h2 = x.shape
        # print('bs, d2, w2, h2 ',bs,d2,w2,h2)
        x = self.layer2(x)
        x = self.layer3(x)
        # bs,d3,w3,h3 = x.shape
        # print('bs, d3, w3, h3 ',bs,d3,w3,h3)
        x = x.view(bs,-1)
        x = self.postFF(x)
        return x
     