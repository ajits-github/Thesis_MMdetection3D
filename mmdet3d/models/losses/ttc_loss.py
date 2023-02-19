# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
# from .utils import weighted_loss



# @mmcv.jit(derivate=True, coderize=True)
# # @weighted_loss
# def smooth_l1_loss(pred, target, beta=1.0):
#     """Smooth L1 loss.

#     Args:
#         pred (torch.Tensor): The prediction.
#         target (torch.Tensor): The learning target of the prediction.
#         beta (float, optional): The threshold in the piecewise function.
#             Defaults to 1.0.

#     Returns:
#         torch.Tensor: Calculated loss
#     """
#     assert beta > 0
#     if target.numel() == 0:
#         return pred.sum() * 0

#     assert pred.size() == target.size()
#     diff = torch.abs(pred - target)
#     loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
#                        diff - 0.5 * beta)
#     return loss


# @mmcv.jit(derivate=True, coderize=True)
# @weighted_loss
# def l1_loss(pred, target):
#     """L1 loss.

#     Args:
#         pred (torch.Tensor): The prediction.
#         target (torch.Tensor): The learning target of the prediction.

#     Returns:
#         torch.Tensor: Calculated loss
#     """
#     if target.numel() == 0:
#         return pred.sum() * 0

#     assert pred.size() == target.size()
#     loss = torch.abs(pred - target)
#     return loss


# @LOSSES.register_module()
# class SmoothL1Loss(nn.Module):
#     """Smooth L1 loss.

#     Args:
#         beta (float, optional): The threshold in the piecewise function.
#             Defaults to 1.0.
#         reduction (str, optional): The method to reduce the loss.
#             Options are "none", "mean" and "sum". Defaults to "mean".
#         loss_weight (float, optional): The weight of loss.
#     """

#     def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
#         super(SmoothL1Loss, self).__init__()
#         self.beta = beta
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         """Forward function.

#         Args:
#             pred (torch.Tensor): The prediction.
#             target (torch.Tensor): The learning target of the prediction.
#             weight (torch.Tensor, optional): The weight of loss for each
#                 prediction. Defaults to None.
#             avg_factor (int, optional): Average factor that is used to average
#                 the loss. Defaults to None.
#             reduction_override (str, optional): The reduction method used to
#                 override the original reduction method of the loss.
#                 Defaults to None.
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         loss_bbox = self.loss_weight * smooth_l1_loss(
#             pred,
#             target,
#             weight,
#             beta=self.beta,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss_bbox


# @LOSSES.register_module()
# class L1Loss(nn.Module):
#     """L1 loss.

#     Args:
#         reduction (str, optional): The method to reduce the loss.
#             Options are "none", "mean" and "sum".
#         loss_weight (float, optional): The weight of loss.
#     """

#     def __init__(self, reduction='mean', loss_weight=1.0):
#         super(L1Loss, self).__init__()
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None):
#         """Forward function.

#         Args:
#             pred (torch.Tensor): The prediction.
#             target (torch.Tensor): The learning target of the prediction.
#             weight (torch.Tensor, optional): The weight of loss for each
#                 prediction. Defaults to None.
#             avg_factor (int, optional): Average factor that is used to average
#                 the loss. Defaults to None.
#             reduction_override (str, optional): The reduction method used to
#                 override the original reduction method of the loss.
#                 Defaults to None.
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         loss_bbox = self.loss_weight * l1_loss(
#             pred, target, weight, reduction=reduction, avg_factor=avg_factor)
#         return loss_bbox


# @mmcv.jit(derivate=True, coderize=True)
def ratio_of_depths(time_interval, ttc):
    """MiD Loss.

    Args:
        time_interval (float): The time interval between consequitive frames.
            Defaults to 2.0.
        tau(torch.Tensor): value to convert for ratio of depths

    Returns:
        torch.Tensor: Calculated ratio of depth
    """

    ratio_of_depth = 1 - time_interval/ttc
    return ratio_of_depth

# def check_values(preds, targets):
#     for idx, (pred, target) in enumerate(zip(preds, targets)):
#         if 0<= pred and pred<= 0.5:
#             preds[idx] = 0.6
#         if (0<= target and target<= 0.5) or torch.isnan(target):
#             targets[idx] = 0.6

#     return preds, targets

def check_values(tensors):
    for idx, tensor in enumerate(tensors):
        if 0<= tensor and tensor<= 0.5 or torch.isnan(tensor):
            tensors[idx] = 0.6

    return tensors

def apply_softmax(preds, scale):
    preds = preds.softmax(dim=0)    # apply softmax, try and see with tanh
    preds = preds - 0.5         # move the range to [-0.5, 0.5]
    preds = preds * 2*scale     # move the range to [-10, 10]
    return preds

def apply_min_max(preds, targets, clamp_value):
    pred_clone = preds.clone()
    target_clone = targets.clone()
    for idx, (pred, target) in enumerate(zip(preds, targets)):
        pred_clone[idx] = min(max(-10, pred), clamp_value)
        # preds[idx] = pred_new
        target_clone[idx] = min(max(-10, target), clamp_value)
        # targets[idx] = target_new

    return pred_clone, target_clone



@LOSSES.register_module()
class TTCLoss(nn.Module):
    def __init__(self, time_interval=0.5, scale=10) -> None:
        super().__init__()
        self.time_interval = time_interval
        self.scale = scale # the clamping value

    def forward(self, initial_preds, initial_targets, avg_factor=1.0):
        # print("................pred.size........",initial_preds.size())
        # print("................target.size........",initial_targets.size())
        # print("................initial_preds........",initial_preds)
        # print("................target........",target)
        assert initial_preds.size() == initial_targets.size()



        # self.preds, self.targets = check_values(initial_preds, initial_targets)
        self.targets = check_values(initial_targets)  # check for nans or 0<=pred<=0.5 in targets
        # print('.......pred...before softmax...', initial_preds)
        self.preds = apply_softmax(initial_preds, self.scale) # apply softamx for preds and rescale 
        # print('.......pred...after softmax...', self.preds)
        self.preds, self.targets = apply_min_max(self.preds, self.targets, self.scale) # clamp the values greater than scale for both preds and targets
        self.preds = check_values(self.preds) # check for nans or 0<=pred<=0.5 in preds


        # print('.......pred....before mid..', self.preds)
        # print('.......target...before mid...', self.targets)

        ratio_of_depths_pred = ratio_of_depths(self.time_interval, self.preds)
        ratio_of_depths_target = ratio_of_depths(self.time_interval, self.targets)
        # print('.........ratio_of_depths_pred......',ratio_of_depths_pred)
        # print('.........ratio_of_depths_target......',ratio_of_depths_target)

        log_preds = torch.log(ratio_of_depths_pred)
        log_targets = torch.log(ratio_of_depths_target)
        self.print_nans(log_preds)
        
        mid_loss = torch.abs(log_preds - log_targets)
        # print('.........mid_loss..before....',mid_loss)
        # exit()

        # eps = torch.finfo(torch.float32).eps
        # mid_loss = mid_loss.sum() / (avg_factor + eps)
        mid_loss = mid_loss.sum() / avg_factor
        # print('.........mid_loss.sum.....',mid_loss)

        
        # mid_loss = mid_loss * 10**4
        # print('.........mid_loss..after....',mid_loss, '\n\n')
        if torch.isnan(mid_loss):
            # print('.........mid_loss..after....',mid_loss)
            print(".........Nan FOund.......preds.....\n",self.preds)
            print(".........Nan FOund.......targets.....\n",self.targets)
            print(".........Nan FOund......log_preds......\n",log_preds)
            print(".........Nan FOund......log_targets......\n",log_targets)
            exit()

        # exit()
        # if ratio_of_depths_pred[0] > 0:
        # #   print('..............midloss[0].......', mid_loss[0])

        return mid_loss

    def print_nans(self, log_preds):
        for pred in log_preds:
            if torch.isnan(pred):
                print(".........Nan FOund.......pred.....\n",self.preds)
                print(".........Nan FOund......log_preds......\n",log_preds)
                break



