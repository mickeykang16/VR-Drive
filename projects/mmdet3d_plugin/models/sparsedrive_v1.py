from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False
import yaml 
from einops import rearrange
from models.gaussian import depth2pc, pts2render, focal2fov, getProjectionMatrix, getWorld2View2, rotate_sh
from models.losses import MultiCamLoss, SingleCamLoss
from torch import Tensor
from lpips import LPIPS
from jaxtyping import Float, UInt8
from skimage.metrics import structural_similarity
from einops import reduce

from PIL import Image
from pathlib import Path
from einops import rearrange, repeat
from typing import Union
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from gaussian_network import *


import os
import yaml 
from collections import defaultdict
from torch.cuda.amp.autocast_mode import autocast

import torch
import cv2
import random



FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


__all__ = ["SparseDrive"]


_NUSC_CAM_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
_REL_CAM_DICT = {0: [1,2], 1: [0,3], 2: [0,4], 3: [1,5], 4: [2,5], 5: [3,4]}


def camera2ind(cameras):
    """
    This function transforms camera name list to indices 
    """    
    indices = []
    for cam in cameras:
        if cam in _NUSC_CAM_LIST:
            ind = _NUSC_CAM_LIST.index(cam)
        else:
            ind = None
        indices.append(ind)
    return indices


def get_relcam(cameras):
    """
    This function returns relative camera indices from given camera list
    """
    relcam_dict = defaultdict(list)
    indices = camera2ind(cameras)
    for ind in indices:
        relcam_dict[ind] = []
        relcam_cand = _REL_CAM_DICT[ind]
        for cand in relcam_cand:
            if cand in indices:
                relcam_dict[ind].append(cand)
    return relcam_dict        


def get_config(config, mode='train', weight_path='./weights_SF', novel_view_mode='SF'):
    """
    This function reads the configuration file and return as dictionary
    """
    with open(config, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

        cfg_name = os.path.splitext(os.path.basename(config))[0]
        print('Experiment: ', cfg_name)

        _log_path = os.path.join(cfg['data']['log_dir'], cfg_name)
        cfg['data']['log_path'] = _log_path
        cfg['data']['save_weights_root'] = os.path.join(_log_path, 'models')
        cfg['data']['num_cams'] = len(cfg['data']['cameras'])
        cfg['data']['rel_cam_list'] = get_relcam(cfg['data']['cameras'])

        cfg['model']['mode'] = mode
        cfg['model']['novel_view_mode'] = novel_view_mode

        cfg['load']['load_weights_dir'] = weight_path
            
        if mode == 'eval':
            cfg['ddp']['world_size'] = 1
            cfg['ddp']['gpus'] = [0]
            cfg['training']['batch_size'] = cfg['eval']['eval_batch_size']
            cfg['training']['depth_flip'] = False
    return cfg

@DETECTORS.register_module()
class V1SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        gaussian_branch=None
    ):
        super(V1SparseDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        
        
        if gaussian_branch is not None:
            self.gaussian_branch = build_from_cfg(gaussian_branch, PLUGIN_LAYERS)
        else:
            self.gaussian_branch = None
        
        # with open('./driving_forward.yaml', 'r') as stream:  
        #     self.gaussian_cfg = yaml.load(stream, Loader=yaml.FullLoader)
        self.gaussian_cfg = get_config('./driving_forward.yaml')
        
        self.max_depth = self.gaussian_cfg['training']['max_depth']
        self.min_depth = self.gaussian_cfg['training']['min_depth']
        self.focal_length_scale = self.gaussian_cfg['training']['focal_length_scale']
        self.frame_ids = self.gaussian_cfg['training']['frame_ids']
        self.aug_frame_ids= [1]
        
        self.gaussian_coeff = self.gaussian_cfg['loss']['gaussian_coeff']
        self.read_config(self.gaussian_cfg)
        
        # self.losses = self.init_losses(self.gaussian_cfg)            
        
        if depth_branch is not None:
            # self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
            self.depth_branch = self.set_depthnet(self.gaussian_cfg)
        else:
            self.depth_branch = None
            
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 
            
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)


    def is_ddp(self):
        return dist.is_available() and dist.is_initialized()
    
    def set_depthnet(self, cfg):
        if self.is_ddp():
            rank = dist.get_rank()
            return DepthNetwork(cfg).to(rank)
        else:
            return DepthNetwork(cfg).cuda()

    def set_lpip_loss(self):
        self.losses = self.init_losses(self.gaussian_cfg)

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            self.num_cams = num_cams
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        
        # import pdb; pdb.set_trace()
        if "metas" in signature(self.img_backbone.forward).parameters:
            feat_out, feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feat_out, feature_maps = self.img_backbone(img)
        
        metas['feat_out'] = torch.reshape(
                feat_out, (bs, num_cams) + feat_out.shape[1:]
            )
            
            
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        
            
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
            
            
        
        # if return_depth:
        #     return feature_maps, depths
        return feature_maps
    
    def compute_depth_maps(self, inputs):     
        """
        This function computes depth map for each viewpoint.
        """                  
        source_scale = 0
        for cam in range(self.num_cams):
            ref_K = inputs[('K', source_scale)][:, cam, ...]
            for scale in self.scales:
                disp = inputs[('cam', cam)][('disp', scale)]
                inputs[('cam', cam)][('depth', 0, scale)] = self.to_depth(disp, ref_K)
              
    def to_depth(self, disp_in, K_in):        
        """
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1/disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2)/self.focal_length_scale

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        
        
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def init_losses(self, cfg):
        loss_model = MultiCamLoss(cfg, self.gaussian_coeff)
        return loss_model


    def forward_train(self, img, **data):
        # 
        
        aug_use = random.random() < 0.5
        frame_ids = [random.choice(self.frame_ids[1:])]
        instance_loss = None
        # aug_use = True
        data.update(data.pop('forward'))
                
        feature_maps = self.extract_feat(img, False, data)
        output = dict()
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            self.num_cams = num_cams
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        
        for cam in range(num_cams):
            data[('cam', cam)] = {}
            
        data['extrinsics_inv'] = torch.inverse(data['extrinsics'])
        if self.depth_branch is not None:       
            depth_feats = self.depth_branch(feature_maps, data)
            if self.gaussian_branch is not None:
                for cam in range(num_cams):   
                    data[('cam', cam)].update({('cam_T_cam', 0, 1): data[('cam_T_cam', 0, 1)][:, cam, ...]})
                    data[('cam', cam)].update({('cam_T_cam', 0, -1): data[('cam_T_cam', 0, -1)][:, cam, ...]}) 
                    data[('cam', cam)].update(depth_feats[('cam', cam)])
                    
            self.compute_depth_maps(data)
        else:
            depths = None
     
        if self.gaussian_branch is not None:
            for cam in range(num_cams):
                self.get_gaussian_data(feature_maps, data, cam, frame_ids=frame_ids)
        
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        
        feature_og = []
        for feat in feature_maps:
            feature_og.append(feat.detach().clone())
        data['feature_map_og'] = feature_og
        data['projection_mat_og'] = data['projection_mat']
    
    
        if not aug_use:
            model_outs = self.head(feature_maps, data, original=True)
            output_original = self.head.loss(model_outs, data)
            output.update(output_original)
        
        # if self.gaussian_branch is not None:
        if not aug_use:
            gaussian_losses = 0
            depth_losses = 0
            for cam in range(self.num_cams):
                self.pred_gaussian_imgs(data, cam, frame_ids=frame_ids)
                cam_loss, loss_dict = self.losses(data, cam, frame_ids=frame_ids)
                gaussian_losses += cam_loss    
                
                if "gt_depth" in data:
                    for scale in self.scales:
                        pred_depth = data[('cam', cam)][('depth', 0, scale)] 
                        gt_depth = data["gt_depth"][scale][:, cam].unsqueeze(1)
                        depth_loss = self.depth_loss(pred_depth, gt_depth)
                        depth_losses += depth_loss  
                
            output["loss_gaussian"] = gaussian_losses
            output["loss_depth"] = depth_losses
            
        #### augmentated start
        
        
        if aug_use:
            
            
            data.update(data.pop('aug'))
            data.update(data.pop('forward'))
            frame_ids = self.aug_frame_ids
            
            for cam in range(num_cams):   
                data[('cam', cam)].update({('cam_T_cam', 0, 1): data[('cam_T_cam', 0, 1)][:, cam, ...]})
                data[('cam', cam)].update({('cam_T_cam', 0, -1): data[('cam_T_cam', 0, -1)][:, cam, ...]}) 
            
            with torch.no_grad():
                for cam in range(num_cams):
                    self.get_gaussian_data(feature_maps, data, cam, aug=True, frame_ids=frame_ids)
                for cam in range(self.num_cams):
                    self.pred_gaussian_imgs(data, cam, aug=True, frame_ids=frame_ids)        
        
            img_list = []
            for novel_frame_id in self.aug_frame_ids:
                for cam in range(num_cams):
                    data[('color_aug', novel_frame_id, 0)][:, cam, ...] = data[('color_aug', 0, 0)][:, cam, ...]
                    data[('color', novel_frame_id, 0)][:, cam, ...] = data[('color', 0, 0)][:, cam, ...]

                for cam in range(num_cams):
                    img_list.append(data[('cam', cam)][('gaussian_color', novel_frame_id, 0)].detach())
                    data[('color_aug', 0, 0)][:, cam, ...] = data[('cam', cam)][('gaussian_color', novel_frame_id, 0)].detach()
                    data[('color', 0, 0)][:, cam, ...] = data[('cam', cam)][('gaussian_color', novel_frame_id, 0)].detach()
            

            img_aug = torch.stack(img_list, dim=1)
            ## Normalization ##
            img_aug = ((255.0 * img_aug.clip(0,1)) - data['img_norm_cfg']['mean'].unsqueeze(-1).unsqueeze(-1)) \
                                / data['img_norm_cfg']['std'].unsqueeze(-1).unsqueeze(-1)

            #### augmentated using
            feature_maps_aug = self.extract_feat(img_aug, False, data)
            depth_feats_aug = self.depth_branch(feature_maps_aug, data)
            
            # import pdb; pdb.set_trace()
            data['extrinsics'] =  data['extrinsics'] @ torch.inverse(data[('cam_T_cam', 0, 1)]).type(torch.float32)
            data['extrinsics_inv'] = torch.inverse(data['extrinsics'])
            
            ## transforme from novel to origin
            for cam in range(num_cams):   
                data[('cam', cam)].update({('cam_T_cam', 0, 1): torch.inverse(data[('cam_T_cam', 0, 1)][:, cam, ...])})
                data[('cam', cam)].update({('cam_T_cam', 0, -1): torch.inverse(data[('cam_T_cam', 0, -1)][:, cam, ...])})
                data[('cam', cam)].update(depth_feats_aug[('cam', cam)])
            self.compute_depth_maps(data)
            
            
            for cam in range(num_cams):
                self.get_gaussian_data_novel(feature_maps_aug, data, cam, frame_ids=frame_ids)
            ###


            if self.use_deformable_func:
                feature_maps_aug = feature_maps_format(feature_maps_aug)
            
            head_out = self.head(feature_maps_aug, data, original=False)
            if len(head_out) == 2 :
                model_outs_aug, instance_loss = head_out
            else:
                model_outs_aug = head_out
            output_aug = self.head.loss(model_outs_aug, data)
            output.update(output_aug)
           
            gaussian_losses_aug = 0
            depth_losses_aug = 0
            for cam in range(self.num_cams):
                self.pred_gaussian_imgs(data, cam, aug=True, frame_ids=frame_ids)
                ###
                # import pdb; pdb.set_trace()
                # self.save_images=True
                # data['token'] = ['']
                # self.compute_reconstruction_metrics(data)
                ###
                cam_loss, loss_dict = self.losses(data, cam, frame_ids=frame_ids)
                gaussian_losses_aug += cam_loss    
               
                if "gt_depth" in data:
                    for scale in self.scales:
                        pred_depth = data[('cam', cam)][('depth', 0, scale)] 
                        gt_depth = data["gt_depth_aug"][scale][:, cam].unsqueeze(1)
                        depth_loss = self.depth_loss(pred_depth, gt_depth)
                        depth_losses_aug += depth_loss  
            output["loss_depth_aug"] = 1.0 * depth_losses_aug
            output["loss_gaussian_aug"] = gaussian_losses_aug

            if instance_loss is not None:
                output["loss_instance"] = 0.05*instance_loss
                
            
        return output

    def depth_loss(self, pred, gt):
        loss = 0.0
       
        pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
        gt = gt.reshape(-1)
        fg_mask = torch.logical_and(
            gt > 0.0, torch.logical_not(torch.isnan(pred))
        )
        gt = gt[fg_mask]
        pred = pred[fg_mask]
        pred = torch.clip(pred, 0.0, self.max_depth)
        with autocast(enabled=False):
            error = torch.abs(pred - gt).mean()
            _loss = error * 0.05
        loss = loss + _loss
        return loss

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        # import pdb; pdb.set_trace()
        data.update(data.pop('forward'))
        
        # import pdb; pdb.set_trace()
        # if hasattr(img, 'data'):
        #     img = img.data[0]
        # if self.gaussian_branch is not None:
        #     feature_maps, _ = self.extract_feat(img, True, data)
        frame_ids = [1]
        
        
        feature_maps = self.extract_feat(img, False, data)
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            self.num_cams = num_cams
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        
        for cam in range(num_cams):
            data[('cam', cam)] = {}
            
        data['extrinsics_inv'] = torch.inverse(data['extrinsics'])
        if self.depth_branch is not None:       
            depth_feats = self.depth_branch(feature_maps, data)
            if self.gaussian_branch is not None:
                for cam in range(num_cams):   
                    data[('cam', cam)].update({('cam_T_cam', 0, 1): data[('cam_T_cam', 0, 1)][:, cam, ...]})
                    data[('cam', cam)].update({('cam_T_cam', 0, -1): data[('cam_T_cam', 0, -1)][:, cam, ...]}) 
                    data[('cam', cam)].update(depth_feats[('cam', cam)])
                    
            self.compute_depth_maps(data)
        else:
            depths = None
     
        if self.gaussian_branch is not None:
            for cam in range(num_cams):
                self.get_gaussian_data(feature_maps, data, cam, frame_ids=frame_ids)
        
        
        
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        
        model_outs = self.head(feature_maps, data, original=True)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        # output = []
        
        
               
        if self.gaussian_branch is not None:
            for cam in range(self.num_cams):
                self.pred_gaussian_imgs(data, cam, frame_ids=frame_ids)
                
            image_results = self.compute_reconstruction_metrics(data)
            
            
            # output[0]['psnr'], output[0]['ssim'] = image_results[0], image_results[1]
            # for output_ in output:
            #     output_['psnr'], output_['ssim'] = 
        
        return output, image_results

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)


    def get_gaussian_data(self, feature_maps, inputs, cam, aug=False, frame_ids=None):
        """
        This function computes gaussian data for each viewpoint.
        """
        bs, _, height, width = inputs[('color', 0, 0)][:, cam, ...].shape
        zfar = self.max_depth
        znear = 0.01

        if not aug:
            frame_id = 0
            
            inputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...]
            inputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...]
            
            
            inputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(inputs[('cam', cam)][('depth', frame_id, 0)], inputs[('cam', cam)][('e2c_extr', frame_id, 0)], inputs[('K', 0)][:, cam, ...])
            valid = inputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
            
            inputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
            
            
            
            # rot_maps, scale_maps, opacity_maps, sh_maps = \
            #         self.gaussian_branch(feature_maps, inputs[('cam', cam)][('depth', frame_id, 0)], cam)
            # rot_maps, scale_maps, opacity_maps, sh_maps = \
            #         self.gaussian_branch(inputs[('color_aug', 0, 0)][:, cam, ...], feature_maps, inputs[('cam', cam)][('depth', frame_id, 0)], cam)
            
            rot_maps, scale_maps, opacity_maps, sh_maps = \
                    self.gaussian_branch(inputs[('color_aug', 0, 0)][:, cam, ...], inputs[('cam', cam)][('img_feat', frame_id, 0)], inputs[('cam', cam)][('depth', frame_id, 0)], cam)
            
            
                
            c2w_rotations = rearrange(inputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3], "k i j -> k () () () i j")
            sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])
            inputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps
            inputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps
            inputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps
            inputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_maps

        if aug:
            frame_ids = self.aug_frame_ids

        # novel view
        for frame_id in frame_ids:
              
            inputs[('cam', cam)][('e2c_extr', frame_id, 0)] = \
                torch.matmul(inputs[('cam', cam)][('cam_T_cam', 0, frame_id)].float(), inputs['extrinsics_inv'][:, cam, ...])
            inputs[('cam', cam)][('c2e_extr', frame_id, 0)] = \
                torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(inputs[('cam', cam)][('cam_T_cam', 0, frame_id)]).float())
            

            

            FovX_list = []
            FovY_list = []
            world_view_transform_list = []
            full_proj_transform_list = []
            camera_center_list = []
            
            for i in range(bs):
                intr = inputs[('K', 0)][:, cam, ...][i,:]
                extr = inputs['extrinsics_inv'][:, cam, ...][i,:]
                T_i = inputs[('cam', cam)][('cam_T_cam', 0, frame_id)][i,:]
                FovX = focal2fov(intr[0, 0], width)
                FovY = focal2fov(intr[1, 1], height)
                projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=intr, h=height, w=width).transpose(0, 1).cuda()
                world_view_transform = torch.matmul(T_i.float(), torch.tensor(extr).cuda()).transpose(0, 1)
                # full_proj_transform: (E^T K^T) = (K E)^T
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.float().inverse()[3, :3] 
                FovX_list.append(FovX)
                FovY_list.append(FovY)
                world_view_transform_list.append(world_view_transform.unsqueeze(0))
                full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
                camera_center_list.append(camera_center.unsqueeze(0))
            inputs[('cam', cam)][('FovX', frame_id, 0)] = torch.tensor(FovX_list).cuda()
            inputs[('cam', cam)][('FovY', frame_id, 0)] = torch.tensor(FovY_list).cuda()
            inputs[('cam', cam)][('world_view_transform', frame_id, 0)] = torch.cat(world_view_transform_list, dim=0)
            inputs[('cam', cam)][('full_proj_transform', frame_id, 0)] = torch.cat(full_proj_transform_list, dim=0)
            inputs[('cam', cam)][('camera_center', frame_id, 0)] = torch.cat(camera_center_list, dim=0)


    def get_gaussian_data_novel(self, feature_maps, inputs, cam, frame_ids=None):
        """
        This function computes gaussian data for each viewpoint.
        """
        bs, _, height, width = inputs[('color', 0, 0)][:, cam, ...].shape
        zfar = self.max_depth
        znear = 0.01

        frame_id = 0
        
        inputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...]
        inputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...]
        
        
        inputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(inputs[('cam', cam)][('depth', frame_id, 0)], inputs[('cam', cam)][('e2c_extr', frame_id, 0)], inputs[('K', 0)][:, cam, ...])
        valid = inputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
        
        inputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
        
        rot_maps, scale_maps, opacity_maps, sh_maps = \
                self.gaussian_branch(inputs[('color_aug', 0, 0)][:, cam, ...], inputs[('cam', cam)][('img_feat', frame_id, 0)], inputs[('cam', cam)][('depth', frame_id, 0)], cam)
        
        
            
        c2w_rotations = rearrange(inputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3], "k i j -> k () () () i j")
        sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])
        inputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps
        inputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps
        inputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps
        inputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_maps

        frame_ids = self.aug_frame_ids

        # novel view
        for frame_id in frame_ids:
              
            inputs[('cam', cam)][('e2c_extr', frame_id, 0)] = \
                torch.matmul(inputs[('cam', cam)][('cam_T_cam', 0, frame_id)].float(), inputs['extrinsics_inv'][:, cam, ...])
            inputs[('cam', cam)][('c2e_extr', frame_id, 0)] = \
                torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(inputs[('cam', cam)][('cam_T_cam', 0, frame_id)]).float())
            
            FovX_list = []
            FovY_list = []
            world_view_transform_list = []
            full_proj_transform_list = []
            camera_center_list = []
            
            for i in range(bs):
                intr = inputs[('K', 0)][:, cam, ...][i,:]
                extr = inputs['extrinsics_inv'][:, cam, ...][i,:]
                T_i = inputs[('cam', cam)][('cam_T_cam', 0, frame_id)][i,:]
                FovX = focal2fov(intr[0, 0], width)
                FovY = focal2fov(intr[1, 1], height)
                projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=intr, h=height, w=width).transpose(0, 1).cuda()
                world_view_transform = torch.matmul(T_i.float(), torch.tensor(extr).cuda()).transpose(0, 1)
                # full_proj_transform: (E^T K^T) = (K E)^T
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.float().inverse()[3, :3] 
                FovX_list.append(FovX)
                FovY_list.append(FovY)
                world_view_transform_list.append(world_view_transform.unsqueeze(0))
                full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
                camera_center_list.append(camera_center.unsqueeze(0))
            inputs[('cam', cam)][('FovX', frame_id, 0)] = torch.tensor(FovX_list).cuda()
            inputs[('cam', cam)][('FovY', frame_id, 0)] = torch.tensor(FovY_list).cuda()
            inputs[('cam', cam)][('world_view_transform', frame_id, 0)] = torch.cat(world_view_transform_list, dim=0)
            inputs[('cam', cam)][('full_proj_transform', frame_id, 0)] = torch.cat(full_proj_transform_list, dim=0)
            inputs[('cam', cam)][('camera_center', frame_id, 0)] = torch.cat(camera_center_list, dim=0)

    def pred_gaussian_imgs(self, metas, cam, aug=False, frame_ids=None):
        if aug:
            frame_ids = self.aug_frame_ids
     
        
        for novel_frame_id in frame_ids:
            metas[('cam', cam)][('gaussian_color', novel_frame_id, 0)] = \
                pts2render(inputs=metas, 
                            cam_num=self.num_cams, 
                            novel_cam=cam,
                            novel_frame_id=novel_frame_id, 
                            bg_color=[1.0, 1.0, 1.0],
                            mode='SF')
        return metas
    
    
    @torch.no_grad()
    def compute_reconstruction_metrics(self, inputs):
        """
        This function computes reconstruction metrics.
        """
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0
        frame_id = 1
       
        for cam in range(self.num_cams):
            rgb_gt = inputs[('color', frame_id, 0)][:, cam, ...]
            image = inputs[('cam', cam)][('gaussian_color', frame_id, 0)]
            psnr += self.compute_psnr(rgb_gt, image).mean()
            ssim += self.compute_ssim(rgb_gt, image).mean()
            
            image = inputs['img_norm_cfg']['mean'].unsqueeze(-1).unsqueeze(-1) + image * inputs['img_norm_cfg']['std'].unsqueeze(-1).unsqueeze(-1)
            rgb_gt = inputs['img_norm_cfg']['mean'].unsqueeze(-1).unsqueeze(-1) + rgb_gt * inputs['img_norm_cfg']['std'].unsqueeze(-1).unsqueeze(-1)
            # import pdb; pdb.set_trace()
            if self.save_images:
                assert self.eval_batch_size == 1
                if self.novel_view_mode == 'SF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    # self.save_image(inputs[('color', 0, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_0_gt.png")
                elif self.novel_view_mode == 'MF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    # self.save_image(inputs[('color', -1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_prev_gt.png")
                    # self.save_image(inputs[('color', 1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_next_gt.png")
        psnr /= self.num_cams
        ssim /= self.num_cams
        
        return psnr, ssim
    
    
    
    
    
    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ground_truth = ground_truth.clip(min=0, max=1)
        predicted = predicted.clip(min=0, max=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * mse.log10()
    
    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        value = self.lpips.forward(ground_truth, predicted, normalize=True)
        return value[:, 0, 0, 0]
    
    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ssim = [
            structural_similarity(
                gt.detach().cpu().numpy(),
                hat.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)

    def save_image(
        self,
        image: FloatImage,
        path: Union[Path, str],
    ) -> None:
        """Save an image. Assumed to be in range 0-1."""

        # Create the parent directory if it doesn't already exist.
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Save the image.
        Image.fromarray(self.prep_image(image)).save(path)
        
        
    def prep_image(self, image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
        # Handle batched images.
        if image.ndim == 4:
            image = rearrange(image, "b c h w -> c h (b w)")

        # Handle single-channel images.
        if image.ndim == 2:
            image = rearrange(image, "h w -> () h w")

        # Ensure that there are 3 or 4 channels.
        channel, _, _ = image.shape
        if channel == 1:
            image = repeat(image, "() h w -> c h w", c=3)
        assert image.shape[0] in (3, 4)

        # ((255.0 * novel.clip(0,1)) - inputs['img_norm_cfg']['mean'].unsqueeze(-1).unsqueeze(-1)) / inputs['img_norm_cfg']['std'].unsqueeze(-1).unsqueeze(-1)
        # image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
        image = image.detach().type(torch.uint8)
        
        return rearrange(image, "c h w -> h w c").cpu().numpy()
