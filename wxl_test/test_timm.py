import torch
import timm
model = timm.create_model('vit_base_patch16_384',pretrained=True)
feat = model.forward_features(torch.randn(2,3,384,384))
print(feat.shape)
models = timm.list_models('vit*',pretrained=True)