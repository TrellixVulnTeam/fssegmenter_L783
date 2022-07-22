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

import segm.utils.torch as ptu


#
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
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_imgs.squeeze(1), self.backbone, self.feat_ids,
                                               self.bottleneck_ids, self.lids)
            support_masked_feats = self.mask_feature(support_feats, support_masks.squeeze(1).clone())

        query_feat_9 = query_feats[9]
        support_feat_9 = support_feats[9]
        support_masked_feat_9 = support_masked_feats[9]

        B, S, C, H, W = support_imgs.shape
        query_feat_9 = rearrange(query_feat_9, 'b c h w -> b (h w) c')
        support_feat_9 = rearrange(support_feat_9, 'b c h w -> b (h w) c')
        support_masked_feat_9 = rearrange(support_masked_feat_9, 'b c h w -> b (h w) c')

        fore_feature = support_masked_feat_9.mean(1)
        q_mask, fore_feature_decoder = self.decoder(query_feat_9, fore_feature, (H, W))
        s_mask, _ = self.decoder(support_feat_9, fore_feature, (H, W))

        q_mask = F.interpolate(q_mask, size=(H, W), mode="bilinear")
        s_mask = F.interpolate(s_mask, size=(H, W), mode="bilinear")

        q_mask = torch.sigmoid(q_mask.squeeze(1))
        s_mask = torch.sigmoid(s_mask.squeeze(1))

        # return q_mask.squeeze(1) / S,torch.stack(s_masks,2).squeeze(1),torch.stack(fore_features,1),torch.stack(fore_features_decoder,1)
        return q_mask, s_mask

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
