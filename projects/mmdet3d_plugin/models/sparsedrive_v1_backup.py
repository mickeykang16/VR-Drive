from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS, GAUSSIAN_LAYERS
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

from tqdm import tqdm

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


__all__ = ["SparseDrive"]


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
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        
        if gaussian_branch is not None:
            self.gaussian_branch = build_from_cfg(gaussian_branch, GAUSSIAN_LAYERS)
        else:
            self.gaussian_branch = None
        
        with open('./driving_forward.yaml', 'r') as stream:  
            self.gaussian_cfg = yaml.load(stream, Loader=yaml.FullLoader)
        
        self.max_depth = self.gaussian_cfg['training']['max_depth']
        self.min_depth = self.gaussian_cfg['training']['min_depth']
        self.focal_length_scale = self.gaussian_cfg['training']['focal_length_scale']
        self.frame_ids = self.gaussian_cfg['training']['frame_ids']
        self.gaussian_coeff = self.gaussian_cfg['loss']['gaussian_coeff']
        self.losses = self.init_losses(self.gaussian_cfg)            
        
            
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 

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
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
            
            
        for cam in range(num_cams):
            metas[('cam', cam)] = {}
        # if return_depth and self.depth_branch is not None:
        if self.depth_branch is not None:
            # import pdb; pdb.set_trace()
            depths, depth_feats = self.depth_branch(feature_maps, metas, metas.get("focal"), \
                                                    self.max_depth, self.min_depth, self.focal_length_scale)
            if self.gaussian_branch is not None:
                for cam in range(num_cams):   
                    metas[('cam', cam)].update({('cam_T_cam', 0, 1): metas[('cam_T_cam', 0, 1)][:, cam, ...]})
                    metas[('cam', cam)].update({('cam_T_cam', 0, -1): metas[('cam_T_cam', 0, -1)][:, cam, ...]}) 
                    metas[('cam', cam)].update(depth_feats[('cam', cam)])
        else:
            depths = None
        if self.gaussian_branch is not None:
            # gaussian = self.gaussian_branch(feature_maps, depths[0])
            metas['extrinsics_inv'] = torch.inverse(metas['extrinsics'])
            for cam in range(num_cams):
                self.get_gaussian_data(feature_maps, metas, cam)
            # for cam in range(self.num_cams):
            #     # self.pred_cam_imgs(inputs, outputs, cam)
            #     # if self.gaussian:
            #     self.pred_gaussian_imgs(metas, cam)
            #     cam_loss, loss_dict = self.losses(metas, cam)
            #     losses += cam_loss  
        else:
            gaussian = None
        
        
        
            
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

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
        # import pdb; pdb.set_trace()
        data.update(data.pop('forward'))
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        if self.gaussian_branch is not None:
            gaussian_losses = 0
            for cam in range(self.num_cams):
                self.pred_gaussian_imgs(data, cam)
                cam_loss, loss_dict = self.losses(data, cam)
                gaussian_losses += cam_loss
            output["loss_gaussian"] = gaussian_losses
        else:
            gaussian = None
            
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        data.update(data.pop('forward'))
        # import pdb; pdb.set_trace()
        # if self.gaussian_branch is not None:
        #     feature_maps, _ = self.extract_feat(img, True, data)
        feature_maps = self.extract_feat(img, False, data)
        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        
        
        if self.gaussian_branch is not None:
            for cam in range(self.num_cams):
                self.pred_gaussian_imgs(data, cam)
                
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


    def get_gaussian_data(self, feature_maps, inputs, cam):
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
                self.gaussian_branch(feature_maps, inputs[('cam', cam)][('depth', frame_id, 0)], cam)
            
        c2w_rotations = rearrange(inputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3], "k i j -> k () () () i j")
        sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])
        inputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps
        inputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps
        inputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps
        inputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_maps

        
        # novel view
        for frame_id in self.frame_ids[1:]:
              
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

    def pred_gaussian_imgs(self, metas, cam):
        
        for novel_frame_id in self.frame_ids[1:]:
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
            self.save_images=False
            image = inputs['img_norm_cfg']['mean'].unsqueeze(-1).unsqueeze(-1) + image * inputs['img_norm_cfg']['std'].unsqueeze(-1).unsqueeze(-1)
            rgb_gt = inputs['img_norm_cfg']['mean'].unsqueeze(-1).unsqueeze(-1) + rgb_gt * inputs['img_norm_cfg']['std'].unsqueeze(-1).unsqueeze(-1)
            if self.save_images:
                assert self.eval_batch_size == 1
                if self.novel_view_mode == 'SF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', 0, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_0_gt.png")
                elif self.novel_view_mode == 'MF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', -1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_prev_gt.png")
                    self.save_image(inputs[('color', 1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_next_gt.png")
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
