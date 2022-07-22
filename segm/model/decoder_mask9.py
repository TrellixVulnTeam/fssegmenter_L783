import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_

from segm.model.blocks_mask9 import Block, FeedForward
from segm.model.utils import init_weights


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = int(math.sqrt(x.shape[1]))

        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class MaskTransformer(nn.Module):
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, fore_feature,im_size,similarity):
        H, W = im_size
        GS = int(math.sqrt(x.shape[1]))

        # x = torch.cat((x,fore_feature), 1)
        x = self.proj_dec(x)    # [B, 3600, 192]
        fore_feature = self.proj_dec(fore_feature)    # [B, 1, 192]
        masks_0 = self.get_att_mask(x,fore_feature)
        masks_list = []
        for blk in self.blocks:
            # masks_0 = masks_0.squeeze(1)
            # masks_0 = masks_0.lt(0).int()
            masks_0 = masks_0.argmax(1)
            masks_0 = masks_0.float()
            masks_0[masks_0.bool()] = torch.tensor(-1e100).to(masks_0.device)
            masks_0 = rearrange(masks_0, 'b h w -> b (h w)')
            # masks_0 = torch.cat([masks_0, torch.zeros([masks_0.shape[0], 1]).to(masks_0.device)], 1)
            x,masks_1 = blk(x,fore_feature,similarity,masks_0.reshape(masks_0.shape[0],1,1,masks_0.shape[1]))
            masks_0 = masks_1
            masks_list.append(self.proj_mask(masks_1))

        masks = self.proj_mask(masks_1)

        return masks,masks_list[1:-1]

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_att_mask(self,x,fore_feature):
        GS = int(math.sqrt(x.shape[1]))
        patches, cls_seg_feat = x,fore_feature
        # patches = patches @ self.proj_patch
        # cls_seg_feat = cls_seg_feat @ self.proj_classes

        # patches = patches / patches.norm(dim=-1, keepdim=True)
        # cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        patches = patches @ cls_seg_feat.transpose(1, 2)

        patches = rearrange(patches,'b (h w) c -> b c h w',h=GS)
        masks = self.mask_conv(patches)
        # masks[:,0]=0
        # masks[:,1]=1
        return masks

    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.mask_conv = nn.Sequential(nn.Conv2d(1, 2, 3, 1, 1),
                                       nn.BatchNorm2d(2),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(2, 2, 3, 1, 1),
                                       nn.BatchNorm2d(2),
                                       nn.ReLU(inplace=True)
                                       )

        # self.cls_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)
        # self.proj_dec_q = nn.Linear(d_encoder, d_model)
        # self.proj_dec_s = nn.Linear(d_encoder, d_model)

        # self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        # self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_mask = nn.Conv2d(2,1,3,1,1)


        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        # trunc_normal_(self.cls_emb, std=0.02)