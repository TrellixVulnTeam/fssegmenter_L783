# python train.py --backbone {vgg16, resnet50, resnet101}
#                 --fold {0, 1, 2, 3}
#                 --benchmark pascal
#                 --lr 1e-3
#                 --bsz 20
#                 --logpath "your_experiment_name"

# python train.py --backbone resnet50 --fold 0 --benchmark pascal --niter 50 --lr 1e-3 --bsz 20 --logpath hsnet/test
# python train.py --fold 0 --benchmark pascal --lr 1e-3 --bsz 20 --logpath logs/segm/test1 --model segm

from train import *
import os
from einops import rearrange

os.chdir('/home/prlab/wxl/fss-frame-hsnet')

# Arguments parsing
parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
parser.add_argument('--datapath', type=str, default=r'/home/prlab/wxl/dataset/dir/pcontext/VOCdevkit')
parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
parser.add_argument('--model', type=str, default='hsnet', choices=['hsnet', 'segm'])
parser.add_argument('--logpath', type=str, default='')
parser.add_argument('--bsz', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--niter', type=int, default=2000)
parser.add_argument('--nworker', type=int, default=8)
parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
args = parser.parse_args('')
Logger.initialize(args, training=True)

# Model initialization
if args.model == 'hsnet':
    model = HypercorrSqueezeNetwork(args.backbone, False)
elif args.model == 'segm':
    net_kwargs = {'image_size': (400, 400),
                  'patch_size': 16, 'd_model': 192, 'n_heads': 3, 'n_layers': 12,
                  'normalization': 'vit', 'distilled': False, 'backbone': 'vit_tiny_patch16_384',
                  'dropout': 0.0, 'drop_path_rate': 0.1,
                  'decoder': {'drop_path_rate': 0.0, 'dropout': 0.1, 'n_layers': 2,
                              'name': 'mask_transformer', 'n_cls': 1},
                  'n_cls': 1}
    model = create_segmenter(net_kwargs)
Logger.log_params(model)

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Logger.info('# available GPUs: %d' % torch.cuda.device_count())
model = nn.DataParallel(model)
model.to(device)

# Helper classes (for training) initialization
optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
Evaluator.initialize()

# Dataset initialization
FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

# Train HSNet
best_val_miou = float('-inf')
best_val_loss = float('inf')

'''
epoch = 0
dataloader = dataloader_trn
training = True
utils.fix_randseed(None) if training else utils.fix_randseed(0)
model.module.train_mode() if training else model.module.eval()
average_meter = AverageMeter(dataloader.dataset)
for idx, batch in enumerate(dataloader): break
batch = utils.to_cuda(batch)
query_img,support_imgs,support_masks = batch['query_img'], batch['support_imgs'], batch['support_masks']
self = model.module

# in model
B,S,C,H,W = support_imgs.shape
# masked support imgs
support_imgs_masked = support_imgs * support_masks.unsqueeze(2).expand(support_imgs.shape)

q_feature = self.encoder(query_img, return_features=True)
# remove CLS/DIST tokens for decoding
num_extra_tokens = 1 + self.encoder.distilled
q_feature = q_feature[:, num_extra_tokens:]
s_masks = [] # s_masks output
q_masks = torch.zeros((B,1,H,W)).to(f'cuda:{torch.cuda.current_device()}') # q_masks output
# q_masks = torch.zeros((B,2,H,W))
# fore_feature_difference = torch.zeros((B,self.decoder.d_model)).to(ptu.device)
fore_features = [] # fore_features output
fore_features_decoder = [] # fore_features_decoder means fore_features after decoder

# in shots
i = 0
s_feature_masked = self.encoder(support_imgs_masked[:, i], return_features=True)[:, num_extra_tokens:]
s_feature = self.encoder(support_imgs[:, i], return_features=True)[:, num_extra_tokens:]

# compute fore_feature
fore_feature = s_feature_masked.mean(1)
fore_features.append(fore_feature)

# in decoder
self = self.decoder
x, fore_feature,im_size = q_feature,fore_feature, (H, W)
H, W = im_size
GS = H // self.patch_size

x = self.proj_dec(x)
# cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
# x = torch.cat((x, cls_emb,fore_feature.unsqueeze(1)), 1)
x = torch.cat((x,fore_feature.unsqueeze(1)), 1)
for blk in self.blocks:break



x = blk(x)
patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls:]
fore_feature_decoder = cls_seg_feat
patches = patches @ self.proj_patch
cls_seg_feat = cls_seg_feat @ self.proj_classes

patches = patches / patches.norm(dim=-1, keepdim=True)
cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

masks = patches @ cls_seg_feat.transpose(1, 2)
# masks = self.mask_norm(masks)
masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
masks = self.proj_mask(masks)




'''

#
# for epoch in range(args.niter):
#
#     trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
#     with torch.no_grad():
#         val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)
#
#     # Save the best model
#     if val_miou > best_val_miou:
#         best_val_miou = val_miou
#         Logger.save_model_miou(model, epoch, val_miou)
#
#     Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
#     Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
#     Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
#     Logger.tbd_writer.flush()
# Logger.tbd_writer.close()
# Logger.info('==================== Finished Training ====================')