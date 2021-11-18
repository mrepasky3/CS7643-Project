# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:24:32 2021

@author: sswanson7
"""
from torch import nn
from EmbedConv import EmbedConv

class ImageDeepSetNetwork(nn.Module):
    def __init__(self, 
                 entity_size=[1,28,28],
                 output_dim=1,
                 embed_type='conv',
                 embed_features=20,
                 activation=nn.Tanh(),
                 pool='max'
                 ):
                 
        super().__init__()
        self.entity_size = entity_size
        
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
            
        self.pool = pool
        # Just pool then feed-forward 
        self.post_ff = nn.Sequential(
                            nn.Linear(embed_features, embed_features*2),
                            activation,
                            nn.Linear(embed_features*2, output_dim)
                        )
        
    def forward(self, x, mask):
        # Embedding
        bs, set_size, d, h, w = x.shape
        x = x.view(bs*set_size, d, h, w)
        x = self.embedding(x).view(bs,set_size,-1)
        
        if self.pool == 'max':
            m = (1-mask.unsqueeze(dim=2))*-1e10 # Big negative if not actually there
            x = x + m
            # Max pool over entity dimension
            x = x.max(dim=1)[0]
        elif self.pool == 'mean' or self.pool == 'avg':
            m = mask.unsqueeze(dim=-1)
            x = x * m
            summed = x.sum(dim=1)
            denom = m.sum(dim=-2)
            x = summed/denom
            
        out = self.post_ff(x)
        return out
    
    
    def compute_loss(self, data, masks, output):
        pred = self(data, masks)
        loss = ((pred-output)**2).mean()
        return loss
    
if __name__ == '__main__':
    
    import torch
    torch.manual_seed(0)
    from dataset_prep import SetsMNIST
    from utils import process_dataset
    
    dataset = SetsMNIST()
    train_sets, train_labels, test_sets, test_labels = dataset.sum_task()
    
    max_size = 9
    
    data, masks, outs = process_dataset(train_sets, train_labels)
    
    ds = ImageDeepSetNetwork(pool='avg')
    
    d,m = data[:3],masks[:3]
    x = ds(d,m)
    # print(x)
    
    
    
    