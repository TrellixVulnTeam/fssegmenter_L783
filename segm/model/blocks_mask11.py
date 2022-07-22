"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path
import math
import torch.nn.functional as F

from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.Mk = nn.Linear(dim, dim)
        self.Mv = nn.Linear(dim, dim)
        self.Mq = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.Mq.weight.data += torch.eye(self.Mq.weight.data.shape[0]).to(self.Mq.weight.data.device)
        self.Mk.weight.data += torch.eye(self.Mk.weight.data.shape[0]).to(self.Mk.weight.data.device)
        self.Mv.weight.data += torch.eye(self.Mv.weight.data.shape[0]).to(self.Mv.weight.data.device)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, fore_feature,mask=None):
        Fx,Fs = x,fore_feature
        B, N, C = Fx.shape
        q = self.Mq(Fs).reshape(B, 1, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = self.Mk(Fx).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = self.Mv(Fx).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn if mask is None else attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print(attn.shape)
        # print(v.shape)
        x = (attn.transpose(2, 3).expand(-1,-1,-1,v.shape[-1]) * v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale = dim ** -0.5
        self.mask_conv = nn.Sequential(nn.Conv2d(dim, int(dim/heads), 3, 1, 1),
                                       nn.BatchNorm2d(int(dim/heads)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(int(dim/heads), int(dim/heads), 3, 1, 1),
                                       nn.BatchNorm2d(int(dim/heads)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(int(dim/heads), 2, 3, 1, 1),
                                       nn.BatchNorm2d(2),
                                       nn.ReLU(inplace=True),
                                       )


    def forward(self, x, fore_feature, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), fore_feature,mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        masks = self.get_att_mask(x,fore_feature)
        return x,masks

    def get_att_mask(self,x,fore_feature):
        GS = int(math.sqrt(x.shape[1]))
        patches, cls_seg_feat = x,fore_feature
        patches = rearrange(patches,'b (h w) c -> b c h w',h=GS)
        masks = self.mask_conv(patches)
        return masks