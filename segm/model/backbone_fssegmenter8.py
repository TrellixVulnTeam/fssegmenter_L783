from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet
import segm.model.resnet as models
from segm.model.conv4d import CenterPivotConv4d

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
        self.backbone_type = backbone
        self.decoder = decoder

        self.zoom_factor = 8
        self.vgg = False
        self.shot = 1
        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        # self.sim_conv = nn.Sequential(  nn.Conv2d(1, 1, 3, 3),
        #                                 # nn.BatchNorm2d(1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(1, 1, 3, 1),
        #                                 # nn.BatchNorm2d(1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(1, 1, 3, 1),
        #                                 nn.ReLU(inplace=True)
        #                                 )
        self.conv_4d = nn.Sequential(CenterPivotConv4d(1, 1, (3, 3, 3, 3), (1, 1, 1, 1), (1, 1, 1, 1)),
                                     nn.ReLU(inplace=True),
                                     CenterPivotConv4d(1, 1, (3, 3, 3, 3), (1, 1, 1, 1), (1, 1, 1, 1)),
                                     nn.ReLU(inplace=True),
                                     )
        self.pool = nn.MaxPool2d(4, 4, 2)

        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        # print(backbone)
        # print(resnet)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        #
        for layer in [self.layer0, self.layer2, self.layer2, self.layer3, self.layer4]:
            for p in layer.parameters():
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
        # backbone from PFENet,get features with shape [1,256,60,60]
        # print(query_img.device)
        x_size = query_img.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(query_img)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        #   Support Feature
        supp_feat_list = []
        final_supp_list = []
        mask_list = []

        for i in range(self.shot):
            mask = (support_masks[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(support_imgs[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_feat_masked = Weighted_GAP(supp_feat, mask)
            supp_feat_list.append(supp_feat_masked)

        cosine_eps = 1e-7
        # B, S, C, h, w = support_imgs.shape
        query_feat = rearrange(query_feat, 'b c h w -> b (h w) c')
        supp_feat = rearrange(supp_feat, 'b c h w -> b (h w) c')
        fore_feature = rearrange(supp_feat_masked, 'b c h w -> b (h w) c')

        query_feat_norm = torch.norm(query_feat, 2, 2, True)
        supp_feat_norm = torch.norm(supp_feat, 2, 2, True)

        assert not (torch.isnan(query_feat)).any(), 'query_feat value error'
        assert not (torch.isnan(supp_feat)).any(), 'supp_feat value error'
        assert not (torch.isnan(fore_feature)).any(), 'fore_feature value error'
        assert not (torch.isnan(query_feat_norm)).any(), 'query_feat_norm value error'
        assert not (torch.isnan(supp_feat_norm)).any(), 'supp_feat_norm value error'

        similarity_q = torch.bmm(query_feat, supp_feat.transpose(1, 2)) / (
                    torch.bmm(query_feat_norm, supp_feat_norm.transpose(1, 2)) + cosine_eps)
        assert not (torch.isnan(similarity_q)).any(), 'similarity_q value error'
        # similarity_q = rearrange(similarity_q, 'b (hq wq) (hs ws) -> (b hq wq) 1 hs ws',hq=60,hs=60)
        similarity_q = rearrange(similarity_q, 'b (hq wq) (hs ws) -> b 1 hq wq hs ws', hq=60, hs=60)
        assert not (torch.isnan(similarity_q)).any(), 'similarity_q value error'
        similarity_q = self.conv_4d(similarity_q)
        assert not (torch.isnan(similarity_q)).any(), 'similarity_q value error'
        similarity_q = rearrange(similarity_q, 'b 1 hq wq hs ws -> b (hq wq) hs ws')
        assert not (torch.isnan(similarity_q)).any(), 'similarity_q value error'

        similarity_q = self.pool(similarity_q)
        assert not (torch.isnan(similarity_q)).any(), 'similarity_q value error'
        # similarity_q = rearrange(similarity_q, '(b hq wq) 1 hs ws -> b (hq wq) (hs ws)',b=x_size[0],hq=60)
        similarity_q = rearrange(similarity_q, 'b hq_wq hs ws -> b hq_wq (hs ws)')
        assert not (torch.isnan(similarity_q)).any(), 'similarity_q value error'
        query_feat = torch.cat([query_feat, similarity_q], 2)

        similarity_s = torch.bmm(supp_feat, supp_feat.transpose(1, 2)) / (
                    torch.bmm(supp_feat_norm, supp_feat_norm.transpose(1, 2)) + cosine_eps)
        assert not (torch.isnan(similarity_s)).any(), 'similarity_s value error'
        # similarity_s = rearrange(similarity_s, 'b (hq wq) (hs ws) -> (b hq wq) 1 hs ws', hq=60, hs=60)
        similarity_s = rearrange(similarity_s, 'b (hq wq) (hs ws) -> b 1 hq wq hs ws', hq=60, hs=60)
        assert not (torch.isnan(similarity_s)).any(), 'similarity_s value error'
        similarity_s = self.conv_4d(similarity_s)
        similarity_s = rearrange(similarity_s, 'b 1 hq wq hs ws -> b (hq wq) hs ws')
        similarity_s = self.pool(similarity_s)
        assert not (torch.isnan(similarity_s)).any(), 'similarity_s value error'
        similarity_s = rearrange(similarity_s, 'b hq_wq hs ws -> b hq_wq (hs ws)')
        assert not (torch.isnan(similarity_s)).any(), 'similarity_s value error'
        supp_feat = torch.cat([supp_feat, similarity_s], 2)

        assert not (torch.isnan(query_feat)).any(), 'query_feat value error'
        assert not (torch.isnan(supp_feat)).any(), 'supp_feat value error'
        assert not (torch.isnan(fore_feature)).any(), 'fore_feature value error'

        q_mask, masks_list_q = self.decoder(query_feat, fore_feature, (h, w))
        s_mask, _ = self.decoder(supp_feat, fore_feature, (h, w))

        assert not (torch.isnan(q_mask)).any(), 'q_mask value error'
        assert not (torch.isnan(s_mask)).any(), 's_mask value error'
        for mask in masks_list_q:
            assert not (torch.isnan(mask)).any(), 'mask value error'

        q_mask = F.interpolate(q_mask, size=(h, w), mode="bilinear", align_corners=True)
        s_mask = F.interpolate(s_mask, size=(h, w), mode="bilinear", align_corners=True)
        masks_list_q = [F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=True) for masks in
                        masks_list_q]

        q_mask = torch.nan_to_num(q_mask, -0.01)
        s_mask = torch.nan_to_num(s_mask, -0.01)
        masks_list_q = [torch.nan_to_num(mask, -0.01) for mask in masks_list_q]
        #
        # assert not (torch.isnan(q_mask)).any(), 'q_mask value error'
        # assert not (torch.isnan(s_mask)).any(), 's_mask value error'
        # for mask in masks_list_q:
        #     assert not (torch.isnan(mask)).any(), 'mask value error'

        q_mask = torch.sigmoid(q_mask.squeeze(1))
        s_mask = torch.sigmoid(s_mask.squeeze(1))
        masks_list_q = [torch.sigmoid(masks.squeeze(1)) for masks in masks_list_q]
        # q_mask = q_mask.squeeze(1)
        # s_mask = s_mask.squeeze(1)
        # masks_list_q = [masks.squeeze(1) for masks in masks_list_q]

        return q_mask, s_mask, masks_list_q

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
