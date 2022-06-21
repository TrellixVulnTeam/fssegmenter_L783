import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_
import segm.utils.torch as ptu

#
class Fssegmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        # self.mask_squeeze = nn.Conv2d(1,1,self.patch_size,self.patch_size,bias=False)

        # for p in self.mask_squeeze.parameters():
        #     p.requires_grad = False
        # nn.init.constant_(self.mask_squeeze.weight, 1 / self.patch_size**2)


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
        q_masks = torch.zeros((B,1,H,W)).to(ptu.device) # q_masks output
        # q_masks = torch.zeros((B,2,H,W))
        # fore_feature_difference = torch.zeros((B,self.decoder.d_model)).to(ptu.device)
        fore_features = [] # fore_features output
        fore_features_decoder = [] # fore_features_decoder means fore_features after decoder


        for i in range(S):
            s_feature_masked = self.encoder(support_imgs_masked[:, i], return_features=True)[:, num_extra_tokens:]
            # s_feature_masked = s_feature_masked[:, num_extra_tokens:]
            s_feature = self.encoder(support_imgs[:, i], return_features=True)[:, num_extra_tokens:]
            # s_feature = s_feature[:, num_extra_tokens:]

            # compute fore_feature
            fore_feature = s_feature_masked.mean(1)
            fore_features.append(fore_feature)

            # compute query_mask
            q_mask,fore_feature_decoder = self.decoder(q_feature,fore_feature, (H, W))#TODO:There is a Bug
            q_mask = F.interpolate(q_mask, size=(H, W), mode="bilinear")
            q_masks += q_mask

            fore_features_decoder.append(fore_feature_decoder.squeeze(1))

            # compute support_mask_pre
            s_feature = s_feature / s_feature.norm(dim=-1, keepdim=True)
            # print(s_feature.shape)
            fore_feature = fore_feature / fore_feature.norm(dim=-1, keepdim=True)

            s_mask = s_feature @ fore_feature.unsqueeze(2)
            # s_mask = self.decoder.mask_norm(s_mask)
            s_mask = rearrange(s_mask, "b (h w) n -> b n h w", h=int(H // self.patch_size))
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


