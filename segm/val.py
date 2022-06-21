import time
import torch

from commen.miou import mean_iou

def reg_time(cost_time):
    hours = int(cost_time / 3600)
    mins = int((cost_time % 3600) / 60)
    secs = int(cost_time % 60)
    hours = hours if hours > 9 else '0' + str(hours)
    mins = mins if mins > 9 else '0' + str(mins)
    secs = secs if secs > 9 else '0' + str(secs)
    regtime = f'{hours}:{mins}:{secs}'
    return regtime

def evaluate(model,val_loader):
    model.eval()
    print_fre = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('-' * 50)
    start_time = time.time()
    q_iou = 0
    s_iou = 0
    for i, batch in enumerate(val_loader):
        if i % print_fre == 0:
            cost_time = time.time() - start_time
            eta_time = int(cost_time / (i + 1) * len(val_loader) - cost_time)
            eta = reg_time(eta_time)
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}\tbatch[{i}/{len(val_loader)}]\teta {eta}')
        query_img = batch['query_img'].to(device)
        query_mask = batch['query_mask'].to(device)
        support_imgs = batch['support_imgs'].to(device)
        support_masks = batch['support_masks'].to(device)
        shot = support_masks.shape[1]
        # query_pred, support_pred = model.forward(query_img, support_imgs, support_masks)
        query_pred, support_pred, fore_features, fore_features_decoder = model.forward(query_img, support_imgs,
                                                                                       support_masks)
        # q_iou += mean_iou(query_pred.argmax(1), query_mask)
        q_iou += mean_iou(query_pred.gt(0.5), query_mask)
        for i in range(shot):
            s_iou += mean_iou(support_pred.gt(0.5)[:,i], support_masks[:,i])
    epoch_cost_time = reg_time(time.time() - start_time)
    q_miou = q_iou / len(val_loader)
    s_miou = s_iou / len(val_loader)/shot

    print(f"Val cost time is {epoch_cost_time},"
          f"Eval q_miou is {q_miou},"
          f"Eval s_miou is {s_miou}")
    return q_miou,s_miou