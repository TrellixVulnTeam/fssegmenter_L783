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
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
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

    def forward(self, query_img,support_imgs,support_masks):
        B,S,C,H,W = support_imgs.shape
        # masked support imgs
        support_imgs_masked = support_imgs * support_masks.unsqueeze(2).expand(support_imgs.shape)

        q_feature = self.encoder(query_img, return_features=True)
        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        q_feature = q_feature[:, num_extra_tokens:]
        s_masks = [] # s_masks output
        q_masks = torch.zeros((B,1,H,W)).to(f'cuda:{torch.cuda.current_device()}') # q_masks output
        fore_features = [] # fore_features output
        fore_features_decoder = [] # fore_features_decoder means fore_features after decoder


        for i in range(S):
            s_feature_masked = self.encoder(support_imgs_masked[:, i], return_features=True)[:, num_extra_tokens:]
            s_feature = self.encoder(support_imgs[:, i], return_features=True)[:, num_extra_tokens:]

            # compute fore_feature
            fore_feature = s_feature_masked.mean(1)
            fore_features.append(fore_feature)

            # compute query_mask
            q_mask,fore_feature_decoder = self.decoder(q_feature,fore_feature, (H, W))#TODO:There is a Bug
            q_mask = F.interpolate(q_mask, size=(H, W), mode="bilinear")
            q_masks += q_mask

            fore_features_decoder.append(fore_feature_decoder.squeeze(1))

            # compute support_mask_pre
            s_mask, _ = self.decoder(s_feature, fore_feature, (H, W))
            s_mask = F.interpolate(s_mask, size=(H, W), mode="bilinear")
            s_masks.append(s_mask)


        q_masks = torch.sigmoid(q_masks.squeeze(1) / S)
        s_masks = torch.sigmoid(torch.stack(s_masks,2))

        # return q_mask.squeeze(1) / S,torch.stack(s_masks,2).squeeze(1),torch.stack(fore_features,1),torch.stack(fore_features_decoder,1)
        return q_masks,s_masks.squeeze(1),torch.stack(fore_features,1),torch.stack(fore_features_decoder,1)

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
