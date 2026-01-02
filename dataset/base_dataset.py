from external.dataset import get_transforms


def construct_dataset(cfg, mode, **kwargs):
    """
    This function constructs datasets.
    """
    # dataset arguments for the dataloader
    if mode == 'train':
        dataset_args = {
            'cameras': cfg['data']['cameras'],
            'back_context': cfg['data']['back_context'],
            'forward_context': cfg['data']['forward_context'],
            'data_transform': get_transforms('train', **kwargs),
            'depth_type': cfg['data']['depth_type'] if 'gt_depth' in cfg['data']['train_requirements'] else None,
            'with_pose': 'gt_pose' in cfg['data']['train_requirements'],
            'with_ego_pose': 'gt_ego_pose' in cfg['data']['train_requirements'],
            'with_mask': 'mask' in cfg['data']['train_requirements']
        }
        
    elif mode == 'val' or mode == 'eval':
        dataset_args = {
            'cameras': cfg['data']['cameras'],
            'back_context': cfg['data']['back_context'],
            'forward_context': cfg['data']['forward_context'],
            'data_transform': get_transforms('train', **kwargs), # for aligning inputs without any augmentations
            'depth_type': cfg['data']['depth_type'] if 'gt_depth' in cfg['data']['val_requirements'] else None,
            'with_pose': 'gt_pose' in cfg['data']['val_requirements'],
            'with_ego_pose': 'gt_ego_pose' in cfg['data']['val_requirements'],
            'with_mask': 'mask' in cfg['data']['val_requirements']            
        }
             
    # NuScenes dataset         
    if cfg['data']['dataset'] == 'nuscenes':
        from dataset.nuscenes_dataset import NuScenesdataset
        if mode == 'train':
            split =  'train'
        else:
            if cfg['model']['novel_view_mode'] == 'MF':
                split = 'eval_MF'
            elif cfg['model']['novel_view_mode'] == 'SF':
                split = 'eval_SF'
            else:
                raise ValueError('Unknown novel view mode: ' + cfg['model']['novel_view_mode'])
        dataset = NuScenesdataset(
            cfg['data']['data_path'], split,
            **dataset_args            
        )
    else:
        raise ValueError('Unknown dataset: ' + cfg['data']['dataset'])
    return dataset