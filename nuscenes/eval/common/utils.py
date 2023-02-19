# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import List, Dict, Any
import torch
import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.utils.data_classes import Box
# from nuscenes.eval.common.loaders import ego_velocity
# from nuscenes.eval.detection.evaluate import nusc_
from nuscenes import NuScenes
from mmdet3d.models.losses.ttc_loss import TTCLoss

DetectionBox = Any  # Workaround as direct imports lead to cyclic dependencies.


def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))


def velocity_l2(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.velocity) - np.array(gt_box.velocity))


def yaw_diff(gt_box: EvalBox, eval_box: EvalBox, period: float = 2*np.pi) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
    yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))

    return abs(angle_diff(yaw_gt, yaw_est, period))


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def attr_acc(gt_box: DetectionBox, pred_box: DetectionBox) -> float:
    """
    Computes the classification accuracy for the attribute of this class (if any).
    If the GT class has no attributes or the annotation is missing attributes, we assign an accuracy of nan, which is
    ignored later on.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Attribute classification accuracy (0 or 1) or nan if GT annotation does not have any attributes.
    """
    if gt_box.attribute_name == '':
        # If the class does not have attributes or this particular sample is missing attributes, return nan, which is
        # ignored later. Note that about 0.4% of the sample_annotations have no attributes, although they should.
        acc = np.nan
    else:
        # Check that label is correct.
        acc = float(gt_box.attribute_name == pred_box.attribute_name)
    return acc


def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation.size)
    sr_size = np.array(sample_result.size)
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union

    return iou


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def boxes_to_sensor(boxes: List[EvalBox], pose_record: Dict, cs_record: Dict):
    """
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box = Box(box.translation, box.size, Quaternion(box.rotation))

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out


def cummean(x: np.array) -> np.array:
    """
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)

# New
def l1_distance_pred(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L1 distance between the ttc for gt and pred
    :param gt_ttc: ttc for GT annotation sample.
    :param pred_ttc: ttc for Predicted sample.
    :return: absolute L1 distance.
    """
    # print("...........gt_box...........in nuscenes.eval.common.utils.py....",gt_box)
    # print("...........pred_box...........in nuscenes.eval.common.utils.py....",pred_box)
    # exit()
    # gt_relative_translation = (gt_box.translation[0] - gt_box.ego_translation[0],
    #                     gt_box.translation[1] - gt_box.ego_translation[1],
    #                     gt_box.translation[2] - gt_box.ego_translation[2])
    
    # ego_velo = ego_velocity(nusc, gt_box.sample_token)
    # # ego_velo = ""

    # gt_relative_translation = gt_box.ego_translation
    # gt_relative_velocity = (gt_box.velocity[0] - ego_velo[0],
    #                     gt_box.velocity[1] - ego_velo[1])

    # pred_relative_translation = pred_box.ego_translation
    # pred_relative_velocity = (pred_box.velocity[0] - ego_velo[0],
    #                     pred_box.velocity[1] - ego_velo[1])
    

    # gt_time_to_coll = np.linalg.norm(np.array(gt_relative_translation)) / np.linalg.norm(np.array(gt_relative_velocity))
    # pred_time_to_coll = np.linalg.norm(np.array(pred_relative_translation)) / np.linalg.norm(np.array(pred_relative_velocity))

    # print(".........gt_box.................", gt_box)
    # print(".........pred_box.................", pred_box)
    ttc_pred_error = abs(gt_box.time_to_coll_calc - pred_box.time_to_coll_pred)
    # ttc_pred_error = abs(gt_box.time_to_coll_calc) - abs(pred_box.time_to_coll_pred)
    # ttc_calc_error = abs(gt_box.time_to_coll_calc) - abs(pred_box.time_to_coll_calc)
    # print("...........ttc_pred_error.............", ttc_pred_error)
    return ttc_pred_error

def l1_distance_calc(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L1 distance between the ttc for gt and pred
    :param gt_ttc: ttc for GT annotation sample.
    :param pred_ttc: ttc for Predicted sample.
    :return: absolute L1 distance.
    """
    # ttc_calc_error = abs(gt_box.time_to_coll_calc) - abs(pred_box.time_to_coll_calc)
    ttc_calc_error = abs(gt_box.time_to_coll_calc - pred_box.time_to_coll_calc)
    # print("...........ttc_calc_error.............", ttc_calc_error)
    return ttc_calc_error

def mid_loss_pred(gt_box: EvalBox, pred_box: EvalBox) -> float:
    ttc_loss = TTCLoss()
    # print("...........pred_box.time_to_coll_pred......1.......", pred_box.time_to_coll_pred)
    # print("...........gt_box.time_to_coll_calc.......1......", gt_box.time_to_coll_calc)
    mid_loss_pred = ttc_loss(torch.tensor([pred_box.time_to_coll_pred]), torch.tensor([gt_box.time_to_coll_calc]))
    mid_loss_pred = mid_loss_pred.cpu().detach().numpy()
    return mid_loss_pred

def mid_loss_calc(gt_box: EvalBox, pred_box: EvalBox) -> float:
    ttc_loss = TTCLoss()
    # print("...........pred_box.time_to_coll_pred......2.......", pred_box.time_to_coll_calc)
    # print("...........gt_box.time_to_coll_calc........2.....", gt_box.time_to_coll_calc)
    mid_loss_calc = ttc_loss(torch.tensor([pred_box.time_to_coll_calc]), torch.tensor([gt_box.time_to_coll_calc]))
    mid_loss_calc = mid_loss_calc.cpu().detach().numpy()
    return mid_loss_calc

## New 
##########################################  NEW   ######################################
# def ego_velocity(nusc:NuScenes, sample_token: str, max_time_diff: float = 1.5):
#   current = nusc.get('sample', sample_token)
#   has_prev = current['prev'] != ''
#   has_next = current['next'] != ''

#   # Cannot estimate velocity for a single sample.
#   if not has_prev and not has_next:
#     raise  Exception("The sample doesn't have previous and next")
#       # return np.array([np.nan, np.nan, np.nan])

#   if has_prev:
#       first = nusc.get('sample', current['prev'])
#   else:
#       first = current

#   if has_next:
#       last = nusc.get('sample', current['next'])
#   else:
#       last = current

  
#   sd_rec_firstsample = nusc.get('sample_data', first['data']['LIDAR_TOP'])
# #   cs_record_firstsample = nusc.get('calibrated_sensor',
# #                         sd_rec_firstsample['calibrated_sensor_token'])
#   # print(cs_record_firstsample)
#   pose_record_firstsample = nusc.get('ego_pose', sd_rec_firstsample['ego_pose_token'])

#   sd_rec_lastsample = nusc.get('sample_data', last['data']['LIDAR_TOP'])
# #   cs_record_lastsample = nusc.get('calibrated_sensor',
# #                         sd_rec_lastsample['calibrated_sensor_token'])
#   pose_record_lastsample = nusc.get('ego_pose', sd_rec_lastsample['ego_pose_token'])


#   pos_last = np.array(pose_record_lastsample['translation'])
#   pos_first = np.array(pose_record_firstsample['translation'])
#   pos_diff = pos_last - pos_first

#   time_last = 1e-6 * last['timestamp']
#   time_first = 1e-6 * first['timestamp']
#   time_diff = time_last - time_first

#   if has_next and has_prev:
#       # If doing centered difference, allow for up to double the max_time_diff.
#       max_time_diff *= 2

#   if time_diff > max_time_diff:
#       # If time_diff is too big, don't return an estimate.
#       return np.array([np.nan, np.nan, np.nan])
#   else:
#       return pos_diff / time_diff

##########################################  NEW   ######################################
## New

