# python train.py --backbone {vgg16, resnet50, resnet101}
#                 --fold {0, 1, 2, 3}
#                 --benchmark pascal
#                 --lr 1e-3
#                 --bsz 20
#                 --logpath "your_experiment_name"
# ps aux|grep prlab|grep python
# fuser -v /dev/nvidia*
# sudo kill -9 281099
# cd /proc/281099
# cat status
# sudo kill -9 ppid

# python train.py --backbone resnet50 --fold 0 --benchmark pascal --lr 1e-3 --bsz 20 --logpath logs
# python train.py --backbone resnet50 --fold 0 --benchmark pascal --lr 5e-4 --bsz 8 --niter 50 --model segm --logpath segm/backbone_fssgmenter_5
# python train.py --backbone vit_base_patch16_384 --fold 0 --benchmark pascal --lr 5e-4 --bsz 20 --niter 50 --model segm --logpath segm/backbone_fssgmenter_6_2_q_loss
# python train.py --backbone resnet50 --fold 0 --benchmark pascal --lr 5e-4 --bsz 8 --niter 50 --model segm --logpath segm/backbone_fssgmenter_5_17_dim_256_head_4
from train import *
import os
from einops import rearrange
from model.base.correlation import Correlation


os.chdir('/home/prlab/wxl/fssegmenter')

# Arguments parsing
parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
parser.add_argument('--datapath', type=str, default=r'/home/prlab/wxl/dataset/dir/pcontext/VOCdevkit')
parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
parser.add_argument('--model', type=str, default='segm', choices=['hsnet', 'segm'])
parser.add_argument('--logpath', type=str, default='')
parser.add_argument('--bsz', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--niter', type=int, default=50)
parser.add_argument('--nworker', type=int, default=8)
parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101','vit_base_patch16_384'])
args = parser.parse_args('')
Logger.initialize(args, training=True)

# Model initialization
if args.model == 'hsnet':
    model = HypercorrSqueezeNetwork(args.backbone, False)
elif args.model == 'segm':
    net_kwargs = {'backbone': args.backbone,
                  'decoder': {'drop_path_rate': 0.0, 'dropout': 0.1, 'n_layers': 6,
                              'n_heads': 4, 'd_model': 256, 'd_ff': 4 * 256,
                              'd_encoder': 256, 'n_cls': 1, 'patch_size': 16,
                              },
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
FSSDataset.initialize(img_size=473, datapath=args.datapath, use_original_imgsize=False)
dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

# Train HSNet
best_val_miou = float('-inf')
best_val_loss = float('inf')

epoch = 0
dataloader = dataloader_trn
training = True
utils.fix_randseed(None) if training else utils.fix_randseed(0)
model.module.train_mode() if training else model.module.eval()
average_meter = AverageMeter(dataloader.dataset)
criterion = torch.nn.BCELoss()
for idx, batch in enumerate(dataloader): break
# for idx, batch in enumerate(dataloader_val): break
batch = utils.to_cuda(batch)
query_img,support_imgs,support_masks = batch['query_img'], batch['support_imgs'], batch['support_masks']
query_pred, support_pred, masks_list_q = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
# query_pred, support_pred = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
# query_pred = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])



# in model
self = model.module
decoder = self.decoder
for blk in decoder.blocks:break
attn = blk.attn

# encoder  = self.encoder
# feat = encoder.forward_features(query_img)
# feat.shape
from segm.model.backbone_fssegmenter5 import *
'''
B, S, C, H, W = support_imgs.shape
q_masks = torch.zeros((B,1,H,W)).to(f'cuda:{torch.cuda.current_device()}') # q_masks output
s_masks = torch.zeros((B,1,H,W)).to(f'cuda:{torch.cuda.current_device()}') # q_masks output


with torch.no_grad():
    query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
    support_feats = self.extract_feats(support_imgs.squeeze(1), self.backbone, self.feat_ids,
                                       self.bottleneck_ids, self.lids)
    support_masked_feats = self.mask_feature(support_feats, support_masks.squeeze(1).clone())
    corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)

corr_25 = corr[1][:,0]
corr_25 = rearrange(corr_25,'b hq wq hs ws -> b (hq wq) (hs ws)')

# in decoder
decoder = self.decoder
self = decoder
x ,im_size = corr_25,(H,W)

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

batch = torch.load('debug/batch.pth')
query_pred2 = torch.load('debug/query_pred.pth')
support_pred2 = torch.load('debug/support_pred.pth')
masks_list_q2 = torch.load('debug/masks_list_q.pth')
q_mask = torch.load('debug/q_mask0.pth')
s_mask = torch.load('debug/s_mask0.pth')
support_pred2 = torch.load('debug/support_pred0.pth')
query_feat = torch.load('debug/query_feat0.pth')
supp_feat = torch.load('debug/supp_feat0.pth')
fore_feature = torch.load('debug/fore_feature0.pth')

query_pred, support_pred,masks_list_q = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
print(query_pred.max())
print(query_pred.min())
print(support_pred.max())
print(support_pred.min())
for mask in masks_list_q:
    print(mask.max())
    print(mask.min())
