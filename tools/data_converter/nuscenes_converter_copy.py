# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

import torch
import functools
from inspect import getfullargspec
import pickle
from os import path as osp
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import numba
# import mmdet3d
# from mmdet3d.core.bbox import points_cam2img
# from mmdet3d.datasets import NuScenesDataset

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

# ============================================Start===============================================

# Copyright (c) OpenMMLab. All rights reserved.
import platform

from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS as MMDET_DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')

def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """Check points is in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray, optional): Number of surfaces a polygon
            contains shape of (num_polygon). Defaults to None.

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    # num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons, ), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces,
                                            normal_vec, d, num_surfaces)



@numba.njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d,
                                     num_surfaces):
    """
    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        normal_vec (np.ndarray): Normal vector of polygon_surfaces.
        d (int): Directions of normal vector.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains
            shape of (num_polygon).

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0] +
                    points[i, 1] * normal_vec[j, k, 1] +
                    points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

def build_dataset(cfg, default_args=None):
    # from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    # elif cfg['type'] == 'CBGSDataset':
    #     dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    elif cfg['type'] in DATASETS._module_dict.keys():
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    else:
        dataset = build_from_cfg(cfg, MMDET_DATASETS, default_args)
    return dataset

def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 1.0, 0.5),
                           axis=1):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): Origin point relate to
            smallest point. Use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0)
            in lidar. Defaults to (0.5, 1.0, 0.5).
        axis (int, optional): Rotation axis. 1 for camera and 2 for lidar.
            Defaults to 1.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(lwh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above to surfaces that
    normal vectors all direct to internal.

    Args:
        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces

def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0)):
    """Check points in rotated bbox and return indices.

    Note:
        This function is for counterclockwise boxes.

    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation.
        z_axis (int, optional): Indicate which axis is height.
            Defaults to 2.
        origin (tuple[int], optional): Indicate the position of
            box center. Defaults to (0.5, 0.5, 0).

    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    # TODO: this function is different from PointCloud3D, be careful
    # when start to use nuscene, check the input
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices

def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        if ann['area'] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get('iscrowd', False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann['segmentation'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(
        bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann

def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1:y1 + h, x1:x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1:y1 + h, x1:x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks

def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_prefix,
                                info_path=None,
                                mask_anno_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                with_mask=False):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
    """
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)
    if dataset_class_name == 'KittiDataset':
        file_client_args = dict(backend='disk')
        dataset_cfg.update(
            test_mode=False,
            split='training',
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=with_mask,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    file_client_args=file_client_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=file_client_args)
            ])

    elif dataset_class_name == 'NuScenesDataset':
        dataset_cfg.update(
            use_valid_flag=True,
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    elif dataset_class_name == 'WaymoDataset':
        file_client_args = dict(backend='disk')
        dataset_cfg.update(
            test_mode=False,
            split='training',
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=6,
                    file_client_args=file_client_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=file_client_args)
            ])

    dataset = build_dataset(dataset_cfg)

    # if database_save_path is None:
    #     database_save_path = osp.join(data_path, f'{info_prefix}_gt_database')
    # if db_info_save_path is None:
    #     db_info_save_path = osp.join(data_path,
    #                                  f'{info_prefix}_dbinfos_train.pkl')
    if database_save_path is None:
        database_save_path = osp.join(args.out_dir, f'{info_prefix}_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(args.out_dir,
                                     f'{info_prefix}_dbinfos_train.pkl')

    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    if with_mask:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info['file_name']: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].tensor.numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
        names = annos['gt_names']
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        # point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)
        point_indices = points_in_rbbox(points, gt_boxes_3d)

        if with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in file2id.keys():
                print(f'skip image {img_path} for empty mask')
                continue
            img_id = file2id[img_path]
            kins_annIds = coco.getAnnIds(imgIds=img_id)
            kins_raw_info = coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos['img_shape'][:2]
            gt_masks = [
                _poly2mask(mask, h, w) for mask in kins_ann_info['masks']
            ]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f'{info_prefix}_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + '.png'
                mask_patch_path = abs_filepath + '.mask.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)



class ArrayConverter:

    SUPPORTED_NON_ARRAY_TYPES = (int, float, np.int8, np.int16, np.int32,
                                 np.int64, np.uint8, np.uint16, np.uint32,
                                 np.uint64, np.float16, np.float32, np.float64)

    def __init__(self, template_array=None):
        if template_array is not None:
            self.set_template(template_array)

    def set_template(self, array):
        """Set template array.

        Args:
            array (tuple | list | int | float | np.ndarray | torch.Tensor):
                Template array.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to
                to a NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range,
                or the contents of a list or tuple do not share the
                same data type, a TypeError is raised.
        """
        self.array_type = type(array)
        self.is_num = False
        self.device = 'cpu'

        if isinstance(array, np.ndarray):
            self.dtype = array.dtype
        elif isinstance(array, torch.Tensor):
            self.dtype = array.dtype
            self.device = array.device
        elif isinstance(array, (list, tuple)):
            try:
                array = np.array(array)
                if array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
                self.dtype = array.dtype
            except (ValueError, TypeError):
                print(f'The following list cannot be converted to'
                      f' a numpy array of supported dtype:\n{array}')
                raise
        elif isinstance(array, self.SUPPORTED_NON_ARRAY_TYPES):
            self.array_type = np.ndarray
            self.is_num = True
            self.dtype = np.dtype(type(array))
        else:
            raise TypeError(f'Template type {self.array_type}'
                            f' is not supported.')

    def convert(self, input_array, target_type=None, target_array=None):
        """Convert input array to target data type.

        Args:
            input_array (tuple | list | np.ndarray |
                torch.Tensor | int | float ):
                Input array. Defaults to None.
            target_type (<class 'np.ndarray'> | <class 'torch.Tensor'>,
                optional):
                Type to which input array is converted. Defaults to None.
            target_array (np.ndarray | torch.Tensor, optional):
                Template array to which input array is converted.
                Defaults to None.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to
                to a NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range,
                or the contents of a list or tuple do not share the
                same data type, a TypeError is raised.
        """
        if isinstance(input_array, (list, tuple)):
            try:
                input_array = np.array(input_array)
                if input_array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
            except (ValueError, TypeError):
                print(f'The input cannot be converted to'
                      f' a single-type numpy array:\n{input_array}')
                raise
        elif isinstance(input_array, self.SUPPORTED_NON_ARRAY_TYPES):
            input_array = np.array(input_array)
        array_type = type(input_array)
        assert target_type is not None or target_array is not None, \
            'must specify a target'
        if target_type is not None:
            assert target_type in (np.ndarray, torch.Tensor), \
                'invalid target type'
            if target_type == array_type:
                return input_array
            elif target_type == np.ndarray:
                # default dtype is float32
                converted_array = input_array.cpu().numpy().astype(np.float32)
            else:
                # default dtype is float32, device is 'cpu'
                converted_array = torch.tensor(
                    input_array, dtype=torch.float32)
        else:
            assert isinstance(target_array, (np.ndarray, torch.Tensor)), \
                'invalid target array type'
            if isinstance(target_array, array_type):
                return input_array
            elif isinstance(target_array, np.ndarray):
                converted_array = input_array.cpu().numpy().astype(
                    target_array.dtype)
            else:
                converted_array = target_array.new_tensor(input_array)
        return converted_array

    def recover(self, input_array):
        assert isinstance(input_array, (np.ndarray, torch.Tensor)), \
            'invalid input array type'
        if isinstance(input_array, self.array_type):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            converted_array = input_array.cpu().numpy().astype(self.dtype)
        else:
            converted_array = torch.tensor(
                input_array, dtype=self.dtype, device=self.device)
        if self.is_num:
            converted_array = converted_array.item()
        return converted_array

def array_converter(to_torch=True,
                    apply_to=tuple(),
                    template_arg_name_=None,
                    recover=True):
    """Wrapper function for data-type agnostic processing.

    First converts input arrays to PyTorch tensors or NumPy ndarrays
    for middle calculation, then convert output to original data-type if
    `recover=True`.

    Args:
        to_torch (Bool, optional): Whether convert to PyTorch tensors
            for middle calculation. Defaults to True.
        apply_to (tuple[str], optional): The arguments to which we apply
            data-type conversion. Defaults to an empty tuple.
        template_arg_name_ (str, optional): Argument serving as the template (
            return arrays should have the same dtype and device
            as the template). Defaults to None. If None, we will use the
            first argument in `apply_to` as the template argument.
        recover (Bool, optional): Whether or not recover the wrapped function
            outputs to the `template_arg_name_` type. Defaults to True.

    Raises:
        ValueError: When template_arg_name_ is not among all args, or
            when apply_to contains an arg which is not among all args,
            a ValueError will be raised. When the template argument or
            an argument to convert is a list or tuple, and cannot be
            converted to a NumPy array, a ValueError will be raised.
        TypeError: When the type of the template argument or
                an argument to convert does not belong to the above range,
                or the contents of such an list-or-tuple-type argument
                do not share the same data type, a TypeError is raised.

    Returns:
        (function): wrapped function.

    Example:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Use torch addition for a + b,
        >>> # and convert return values to the type of a
        >>> @array_converter(apply_to=('a', 'b'))
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> a = np.array([1.1])
        >>> b = np.array([2.2])
        >>> simple_add(a, b)
        >>>
        >>> # Use numpy addition for a + b,
        >>> # and convert return values to the type of b
        >>> @array_converter(to_torch=False, apply_to=('a', 'b'),
        >>>                  template_arg_name_='b')
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> simple_add()
        >>>
        >>> # Use torch funcs for floor(a) if flag=True else ceil(a),
        >>> # and return the torch tensor
        >>> @array_converter(apply_to=('a',), recover=False)
        >>> def floor_or_ceil(a, flag=True):
        >>>     return torch.floor(a) if flag else torch.ceil(a)
        >>>
        >>> floor_or_ceil(a, flag=False)
    """

    def array_converter_wrapper(func):
        """Outer wrapper for the function."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Inner wrapper for the arguments."""
            if len(apply_to) == 0:
                return func(*args, **kwargs)

            func_name = func.__name__

            arg_spec = getfullargspec(func)

            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)

            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}

            all_arg_names = arg_names + kwonly_arg_names

            # in case there are args in the form of *args
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []

            # template argument data type is used for all array-like arguments
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_

            if template_arg_name not in all_arg_names:
                raise ValueError(f'{template_arg_name} is not among the '
                                 f'argument list of function {func_name}')

            # inspect apply_to
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(f'{arg_to_apply} is not '
                                     f'an argument of {func_name}')

            new_args = []
            new_kwargs = {}

            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray

            # non-keyword arguments
            for i, arg_value in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(
                        converter.convert(
                            input_array=arg_value, target_type=target_type))
                else:
                    new_args.append(arg_value)

                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value

            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values

            # keyword arguments and non-keyword arguments using default value
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=kwargs[arg_name],
                            target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]

            # add nameless args provided by *args (if exists)
            new_args += nameless_args

            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data,
                                                         tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for k, v in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data

            if recover:
                return recursive_recover(return_values)
            else:
                return return_values

        return new_func

    return array_converter_wrapper


@array_converter(apply_to=('points', 'angles'))
def rotation_3d_in_axis(points,
                        angles,
                        axis=0,
                        return_mat=False,
                        clockwise=False):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple | float):
            Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 \
        and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(f'axis should in range '
                             f'[-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin]),
            torch.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new

@array_converter(apply_to=('points_3d', 'proj_mat'))
def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
        proj_mat (torch.Tensor | np.ndarray):
            Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        (torch.Tensor | np.ndarray): Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res

# ============================================End===============================================

# @DATASETS.register_module()
class NuScenesDataset():
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

# =====================================Original=================================================

def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        # info_path = osp.join(root_path,
        #                      '{}_infos_test.pkl'.format(info_prefix))
        info_path = osp.join(args.out_dir,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        # info_path = osp.join(root_path,
        #                      '{}_infos_train.pkl'.format(info_prefix))
        info_path = osp.join(args.out_dir,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        # info_val_path = osp.join(root_path,
        #                          '{}_infos_val.pkl'.format(info_prefix))
        info_val_path = osp.join(args.out_dir,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec

# =====================================Original End=================================================

# ===========================================Start================================================
def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        # info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        info_test_path = osp.join(args.out_dir, f'{info_prefix}_infos_test.pkl')
        export_2d_annotation(
            root_path, info_test_path, version=version)
        return

    # info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    # info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')

    info_train_path = osp.join(args.out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(args.out_dir, f'{info_prefix}_infos_val.pkl')
    export_2d_annotation(
        root_path, info_train_path, version=version)
    export_2d_annotation(
        root_path, info_val_path, version=version)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{out_dir}/{info_prefix}_infos_train.pkl')

import argparse
parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        print("train_version:", train_version)
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        print("train_version:", train_version)
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
# ============================================End===============================================