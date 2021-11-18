# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:14:33 2021

@author: sswanson7
"""
import torch
import torch.nn as nn
from EmbedConv import EmbedConv
import numpy as np

''' Self attention mechanism '''
class self_attention_block(nn.Module):
    ''' Self attention block. Made up of 4 layers (q, k, v, and post) and
        several matrix operations, and pooling operations
        
        Expects to recieve an input with shape batch_size, NE, features
            NE is number of entities in set
        Can use a mask to allow for varying NE in a batch
    '''
    def __init__(self, features=6, n_per_head=4, heads=4, pool='avg'):
        super().__init__()
        self.n_embd = n_per_head*heads

        self.heads, self.n_per_head = heads, n_per_head
        self.features = features

        self.q_embed_layer = nn.Linear(self.features, self.n_embd, bias=False)
        self.k_embed_layer = nn.Linear(self.features, self.n_embd, bias=False)
        self.v_layer = nn.Linear(self.features, self.n_embd, bias=False)
        self.post_mlp = nn.Linear(self.n_embd, self.features, bias=False)

        self._init_layer(self.q_embed_layer)
        self._init_layer(self.k_embed_layer)
        self._init_layer(self.v_layer)
        self._init_layer(self.post_mlp)

        self.pool = pool
        

    def _init_layer(self, layer):
        with torch.no_grad():
            if layer.bias is not None:
                layer.bias.uniform_(-0.001,.001)
            layer.weight.normal_(std=1/np.sqrt(self.n_embd))


    def forward(self, inp, mask=None):
        bs,NE,features = inp.shape

        # Make sure inp is zeroed out
        if mask is not None:
            inp = inp*(mask.unsqueeze(dim=2))

        # print('\nbs: ',bs)
        heads,n_embd = self.heads, self.n_embd
        # print(inp.shape) # [bs, NE, nembed]
        q = self.q_embed_layer(inp)
        k = self.k_embed_layer(inp)
        # print(qk_.shape)
        query = q.view(bs, NE, heads, n_embd // heads)
        key = k.view(bs, NE, heads, n_embd // heads)
        # qk.shape = [bs,NE,# heads, embeddings per head, 1 for k and 1 for q]
        value = self.v_layer(inp)
        value = value.view(bs, NE, heads, n_embd//heads)
        # flip around dimensions
        query = query.permute(0,2,1,3)
        key = key.permute(0,2,3,1)
        value = value.permute(0,2,1,3)
        logits = torch.matmul(query, key)

        # print('logits: ',logits.shape)
        # logits.shape = [bs, heads, NE, NE]
        logits /= np.sqrt(n_embd / heads)

        # Apply mask
        if mask is not None:
            mask_ = mask.unsqueeze(dim=1).unsqueeze(dim=1)
            logits = logits + 1.0e9*(mask_-1.)


        p = nn.Softmax(dim=-1)(logits)
        if mask is not None:
            p = p*mask.unsqueeze(dim=2).unsqueeze(dim=1) #*torch.ones(p.shape)

        # print('p: ',p.shape) # [bs, heads, NE, NE]
        # print('v: ',value.shape) # [bs, heads, NE, perhead]

        att_sum = torch.matmul(p, value)

        # print('attn_sum ',att_sum.shape) # [bs,  heads, NE, emb/head]
        att_sum = att_sum.permute(0,2,1,3) # [bs,  NE, heads, output-features (aka emb/head)]
        n_output_entities = att_sum.shape[1]
        att_sum = att_sum.contiguous() # Not really sure why or what this does,
                               # Added after getting errors on view and quick
                               # googling and it works
        att_sum = att_sum.view(bs,n_output_entities,n_embd)

        # [bs, output_entites (aka emb/head), n_embd]
        # print('attn_sum ',att_sum.shape) # [bs,  NE, nembed]
        ####### end self_attention
        post_mlp_val = self.post_mlp(att_sum)
        x = inp + post_mlp_val # skip connection
        # print(x.shape) # [bs, NE, features]
        if mask is not None:
            # need to get mean and std of active features
            N = mask.sum(dim=1,keepdim=True).unsqueeze(dim=-1)
            mu_correction = x.shape[1]/N
            mu = x.mean(dim=[1,2],keepdim=True)*mu_correction

            # add in mu's non-active to make rescaling easy
            x2 = x+(1-mask.unsqueeze(dim=-1))*mu
            # -1 for unbiased estimator of std
            std_correction = ((x.shape[2]*x.shape[1]-1)/(x.shape[2]*N-1))**(0.5)
            std = x2.std(dim=[1,2],keepdim=True,unbiased=True)*std_correction

        else:
            mu = x.mean(dim=[1,2],keepdim=True)
            std = x.std(dim=[1,2],keepdim=True,unbiased=True)

        x = (x-mu)/(std+1e-6)
        
        if mask is None:
            if self.pool == 'mean' or self.pool == 'avg':
                # print('pre-avg: ',x.shape)
                out = x.mean(dim=-2)
            else:
                out = x.max(dim=-2)[0].squeeze_(dim=1)
            # print(out.shape) # [bs, features]
        else:
            if self.pool == 'mean' or self.pool == 'avg':
                mask = mask.unsqueeze(dim=-1)
                x = x * mask
                summed = x.sum(dim=-2)
                denom = mask.sum(dim=-2)
                out = summed/denom
            else:
                mask = mask.unsqueeze(dim=-1)
                has_unmasked_entities = torch.sign(mask.sum(dim=-2, keepdim=True))
                offset = (mask-1)*1e9
                masked = (x+offset)*has_unmasked_entities
                out = masked.max(dim=-2)[0]
        # print(out.shape) [bs, features]
        return out
    

class ImageAttentionNetwork(nn.Module):
    def __init__(self,
                 heads=2, n_per_head=10,
                 entity_size=[1,28,28],
                 output_dim=1,
                 embed_type='conv',
                 embed_features=20,
                 activation=nn.Tanh()):
        
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
            
        # Self attention block
        self.sa_layer = self_attention_block(features=embed_features,
                                             n_per_head=n_per_head,
                                             heads=heads)
        # Info about entities for later
        self.activation = activation
        
        self.post_layers = nn.Sequential(
                                nn.Linear(embed_features, embed_features*2),
                                activation,
                                nn.Linear(embed_features*2, output_dim)
                            )
        

    def forward(self, x, mask=None, debug=False):
        
        bs, set_size, d, h, w = x.shape
        
        # Perform embeddings
        x = x.view(bs*set_size, d, h, w)
        x = self.embedding(x).view(bs,set_size,-1)
                
        x = self.sa_layer(x, mask=mask)
        
        out = self.post_layers(x)
        return out
    
    def compute_loss(self, data, masks, output):
        pred = self(data, masks)
        loss = ((pred-output)**2).mean()
        return loss