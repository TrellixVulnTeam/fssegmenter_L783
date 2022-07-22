# backbone with vit

from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm
from torchvision.models import resnet
import segm.model.resnet as models
from torchvision.models import vgg

from segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_
from model.base.feature import extract_feat_vgg, extract_feat_res

import segm.utils.torch as ptu

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat
#
class Fssegmenter(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            patch_size=16,
            decoder=None,
            n_cls=1,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = timm.create_model(backbone,pretrained=True)

        for p in self.encoder.parameters():
            p.requires_grad = False




    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear',
                                 align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def forward(self, query_img, support_imgs, support_masks):
        b,s,c,h,w = support_imgs.shape
        query_feat = self.encoder.forward_features(query_img)[:,:-1]
        # supp_feat = self.encoder.forward_features(support_imgs.squeeze(1))[:,:-1]
        fore_feature = self.encoder.forward_features((support_imgs * support_masks.unsqueeze(2).expand(b, s, c, h, w)).squeeze(1))[:,:-1]
        fore_feature = fore_feature.mean(1).unsqueeze(1)

        #   Support Feature

        q_mask = self.decoder(query_feat, fore_feature, (h, w))
        # s_mask = self.decoder(supp_feat, fore_feature, (h, w))

        q_mask = F.interpolate(q_mask, size=(h, w), mode="bilinear")
        # s_mask = F.interpolate(s_mask, size=(h, w), mode="bilinear")

        q_mask = torch.sigmoid(q_mask.squeeze(1))
        # s_mask = torch.sigmoid(s_mask.squeeze(1))
        # s_mask = 0

        # return q_mask, s_mask
        return q_mask

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)

    def train_mode(self):
        self.train()
        # self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
