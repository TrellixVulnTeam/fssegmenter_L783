from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet
from torchvision.models import vgg

from segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_
from model.base.feature import extract_feat_vgg, extract_feat_res
from model.base.correlation import Correlation


import segm.utils.torch as ptu


# Decoder不再处理特征向量，而是处理二者之间的相关性
class Fssegmenter(nn.Module):
    def __init__(
            self,
            backbone,
            patch_size,
            decoder,
            n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.backbone_type = backbone
        self.decoder = decoder

        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]

        # self.mask_squeeze = nn.Conv2d(1,1,self.patch_size,self.patch_size,bias=False)

        # for p in self.encoder.parameters():
        #     p.requires_grad = False

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
        B, S, C, H, W = support_imgs.shape
        q_masks = torch.zeros((B,1,H,W)).to(f'cuda:{torch.cuda.current_device()}') # q_masks output
        s_masks = torch.zeros((B,1,H,W)).to(f'cuda:{torch.cuda.current_device()}') # s_masks output

        
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_imgs.squeeze(1), self.backbone, self.feat_ids,
                                               self.bottleneck_ids, self.lids)
            support_masked_feats = self.mask_feature(support_feats, support_masks.squeeze(1).clone())
            corr = Correlation.multilayer_correlation(query_feats, support_masked_feats, self.stack_ids)

        for i in range(1):
            corr_25 = corr[1][:, i]
            corr_25 = rearrange(corr_25, 'b hq wq hs ws -> b (hq wq) (hs ws)')
            q_mask, _ = self.decoder(corr_25, (H, W))
            q_mask = F.interpolate(q_mask, size=(H, W), mode="bilinear")

            q_masks += q_mask

        q_masks = torch.sigmoid(q_masks.squeeze(1))

        # return q_mask.squeeze(1) / S,torch.stack(s_masks,2).squeeze(1),torch.stack(fore_features,1),torch.stack(fore_features_decoder,1)
        return q_masks,s_masks

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
