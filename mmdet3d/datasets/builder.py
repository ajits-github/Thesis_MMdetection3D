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


def build_dataset(cfg, default_args=None):
    print("..................build_dataset.........................")
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        print("...............1..................")
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
        
    elif cfg['type'] == 'ConcatDataset':
        print("...............2..................")
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        print("...............3..................")
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        print("...............4..................")
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        print("...............5..................")
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        print("...............6..................")
        dataset = _concat_dataset(cfg, default_args)
    elif cfg['type'] in DATASETS._module_dict.keys():
        # print("...............7..................")
        # print("...............DATASETS._module_dict.keys()............", DATASETS._module_dict.keys())
        # print("...............DATASETS......................", DATASETS)
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    else:
        print("...............8..................")
        dataset = build_from_cfg(cfg, MMDET_DATASETS, default_args)
    return dataset
