r""" Hypercorrelation Squeeze training (validation) code """
import argparse
import datetime
import os

import torch.optim as optim
import torch.nn as nn
import torch

from model.hsnet import HypercorrSqueezeNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from segm.model.factory import create_segmenter


def train(epoch, model, dataloader, optimizer, training):
    r""" Train HSNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    criterion = torch.nn.BCELoss()

    time_start = datetime.datetime.now()
    for idx, batch in enumerate(dataloader):
        try:
            # 1. Hypercorrelation Squeeze Networks forward pass
            batch = utils.to_cuda(batch)
            # logit_mask_q,logit_mask_s = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
            # pred_mask_q = logit_mask_q.argmax(dim=1)
            if True in torch.isnan(batch['query_img']) or True in torch.isnan(batch['support_imgs']):
                print('nan in imgs')
            query_pred, support_pred,masks_list_q = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
            # query_pred = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
            # raise Exception
            assert not (torch.isnan(query_pred)).any(), 'query_pred value error'
            assert not (torch.isnan(support_pred)).any(), 'support_pred value error'
            for mask in masks_list_q:
                assert not (torch.isnan(mask)).any(), 'mask value error'
            # 2. Compute loss & update model parameters
            # loss = model.module.compute_objective(logit_mask_q, batch['query_mask'])
            # loss = model.module.compute_objective(logit_mask_q, batch['query_mask']) + model.module.compute_objective(logit_mask_s, batch['support_masks'].squeeze(1))
            loss = criterion(query_pred, batch['query_mask']) + 0.3 * criterion(support_pred, batch['support_masks'].squeeze(1))
            for mask in masks_list_q:
                loss = loss+0.1*criterion(mask, batch['query_mask'])
            # loss = criterion(query_pred, batch['query_mask'])
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 3. Evaluate prediction
            # area_inter, area_union = Evaluator.classify_prediction(pred_mask_q, batch)
            area_inter, area_union = Evaluator.classify_prediction(query_pred.gt(0.5).int(), batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, time_start, write_batch_idx=50)
        except Exception as e:
            torch.save(batch,'debug/batch.pth')
            torch.save(model.state_dict(),'debug/model.pth')
            torch.save(masks_list_q,'debug/masks_list_q.pth')
            torch.save(query_pred,'debug/query_pred.pth')
            torch.save(support_pred,'debug/support_pred.pth')

            print(query_pred.max())
            print(query_pred.min())
            print(support_pred.max())
            print(support_pred.min())
            for mask in masks_list_q:
                print(mask.max())
                print(mask.min())
            raise ValueError

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default=r'/home/deep3/wxl/dataset/Pascal/VOCdevkit')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--model', type=str, default='segm', choices=['hsnet', 'segm'])
    parser.add_argument('--logpath', type=str, default='segm/backbone_fssgmenter_11_2_Tanh')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--niter', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vit_base_patch16_384','vgg16', 'resnet50', 'resnet101'])
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # For cuda debugging
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Model initialization
    if args.model == 'hsnet':
        model = HypercorrSqueezeNetwork(args.backbone, False)
    elif args.model == 'segm':
        # 'image_size': (400, 400),
        # 'n_heads': 3,
        # 'distilled': False,
        net_kwargs = {'backbone': args.backbone,
                      'decoder': {'drop_path_rate': 0.0, 'dropout': 0.1, 'n_layers': 6,
                                  'n_heads':3, 'd_model': 192, 'd_ff': 4 * 192,
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
    for epoch in range(args.niter):

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
