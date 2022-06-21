import torch
import torch.nn as nn
import torch.nn.functional as F

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
        support_imgs_masked = support_imgs * support_masks.unsqueeze(2).expand(support_imgs.shape)
        # H_ori, W_ori = query_img.size(2), query_img.size(3)
        # query_img = padding(query_img, self.patch_size)
        # H, W = query_img.size(2), query_img.size(3)

        q_feature = self.encoder(query_img, return_features=True)
        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        q_feature = q_feature[:, num_extra_tokens:]
        s_masks = []
        q_masks = torch.zeros((B,2,H,W)).to(ptu.device)

        for i in range(S):
            s_feature_masked = self.encoder(support_imgs_masked[:, i], return_features=True)
            s_feature_masked = s_feature_masked[:, num_extra_tokens:]
            s_feature = self.encoder(support_imgs[:, i], return_features=True)
            s_feature = s_feature[:, num_extra_tokens:]
            # squeezed_mask = self.mask_squeeze(support_masks[:,i].unsqueeze(1).float()).flatten(1)
            # fore_mask = (squeezed_mask / squeezed_mask.sum()).unsqueeze(-1)
            # fore_feature = (s_feature*fore_mask).sum(1)
            fore_feature = s_feature_masked.mean(1)


            s_mask = self.decoder(s_feature, fore_feature, (H, W))
            s_mask = F.interpolate(s_mask, size=(H, W), mode="bilinear")
            s_masks.append(s_mask)

            q_mask = self.decoder(q_feature,fore_feature, (H, W))
            q_mask = F.interpolate(q_mask, size=(H, W), mode="bilinear")
            q_masks += q_mask

        # masks = unpadding(masks, (H_ori, W_ori))

        return q_masks/S,torch.stack(s_masks,2)

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


