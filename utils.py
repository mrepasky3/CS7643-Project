# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:06:23 2021

@author: sswanson7
"""
import torch
from random import shuffle

def process_dataset(sets, labels,max_size=9):
    ''' Sets up data set (forms them with equal sizes and appropriate masks)
    '''
    _,h,w = sets[0].shape
    d = 1
    data = torch.zeros(len(sets),max_size, d, h, w, dtype=torch.float32)
    masks = torch.zeros(len(sets),max_size, dtype=torch.float32)
    outs = torch.zeros(len(sets),1, dtype=torch.float32)
    
    for i in range(len(sets)):
        
        x = sets[i].unsqueeze(dim=1).float()/255
        n = x.shape[0]
        data[i, :n, :, :, :] = x
        masks[i, :n] = 1.
        outs[i] = labels[i]
        
    return data, masks, outs


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def batch_data(data, masks, outs, bs=100):
    ''' Simple batches from training data
    '''
    n = data.shape[0]
    inds = list(range(n))
    shuffle(inds)
    
    bd, bm, bo = [], [], []
    for binds in chunks(inds, bs):
        bd.append( data[binds] )
        bm.append( masks[binds] )
        bo.append( outs[binds] )
        
    return bd, bm, bo

