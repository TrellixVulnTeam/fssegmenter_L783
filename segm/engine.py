import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu

from commen.miou import mean_iou



def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
):
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    # L2cost = torch.nn.MSELoss()

    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    q_miou = 0
    s_miou = 0
    for batch in logger.log_every(data_loader, print_freq, header):
        query_img = batch['query_img'].to(ptu.device)
        query_mask = batch['query_mask'].to(ptu.device)
        support_imgs = batch['support_imgs'].to(ptu.device)
        support_masks = batch['support_masks'].to(ptu.device)

        with amp_autocast():
            # query_pred,support_pred = model.forward(query_img,support_imgs,support_masks)
            query_pred,support_pred,fore_features,fore_features_decoder = model.forward(query_img,support_imgs,support_masks)
            loss = criterion(query_pred, query_mask) + \
                   criterion(support_pred, support_masks)
                   # L2cost(fore_features_decoder,fore_features)
        q_miou += mean_iou(query_pred.gt(0.5), query_mask)
        s_miou += mean_iou(support_pred.gt(0.5), support_masks)
        index = num_updates-epoch * len(data_loader)
        if index%100 == 0 and index != 0:
            # print('query_pred.gt(0.5):',query_pred.gt(0.5).count_nonzero(),
            #       '\nquery_pred.lt(0.5):',query_pred.lt(0.5).count_nonzero(),
            #       '\nsupport_pred.gt(0.5):',support_pred.gt(0.5).count_nonzero(),
            #       '\nsupport_pred.lt(0.5):',support_pred.lt(0.5).count_nonzero())
            print('q_miou:',q_miou/index)
            print('s_miou:',s_miou/index)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    return logger,q_miou/len(data_loader),s_miou/len(data_loader)


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
