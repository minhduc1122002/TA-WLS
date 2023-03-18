import torch
from utils import data_utils

def loss_func(batch_pred, batch_gt, mask, total_frame, total_joint, missing_frame, missing_joint):
    batch_pred = batch_pred.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)
    mask = mask.contiguous().view(-1, 3)
    ones = torch.ones(mask.size()).cuda()
    return torch.mean(torch.norm( (ones - mask) * (batch_gt - batch_pred), 2, 1)) * (total_frame / missing_frame) * (total_joint / missing_joint)




