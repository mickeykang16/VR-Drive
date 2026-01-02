import torch
from pytorch3d.transforms import matrix_to_euler_angles 

from .loss_util import compute_photometric_loss, compute_masked_loss
from .single_cam_loss import SingleCamLoss

from lpips import LPIPS
import torch.nn as nn
import torch.distributed as dist

class MultiCamLoss(nn.Module):
    """
    Class for multi-camera(spatio & temporal) loss calculation
    """
    def __init__(self, cfg, gaussian_coeff):
        super(MultiCamLoss, self).__init__()

        
        if self.is_ddp():
            rank = dist.get_rank()
            self.lpips = LPIPS(net="vgg").to(rank)
        else:
            self.lpips = LPIPS(net="vgg").cuda()
        
        self.gaussian_coeff = gaussian_coeff
        self.frame_ids = cfg['training']['frame_ids']
        self.l1loss = nn.L1Loss()
        
    def is_ddp(self):
        return dist.is_available() and dist.is_initialized()
    
    def compute_spatio_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None):
        """
        This function computes spatial loss.
        """        
        # self occlusion mask * overlap region mask
        spatio_mask = ref_mask * target_view[('overlap_mask', 0, scale)]
        loss_args = {
            'pred': target_view[('overlap', 0, scale)],
            'target': inputs['color',0, 0][:,cam, ...]         
        }        
        spatio_loss = compute_photometric_loss(**loss_args)
        
        target_view[('overlap_mask', 0, scale)] = spatio_mask         
        return compute_masked_loss(spatio_loss, spatio_mask) 

    def compute_spatio_tempo_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes spatio-temporal loss.
        """
        spatio_tempo_losses = []
        spatio_tempo_masks = []
        for frame_id in self.frame_ids[1:]:

            pred_mask = ref_mask * target_view[('overlap_mask', frame_id, scale)]
            pred_mask = pred_mask * reproj_loss_mask 
            
            loss_args = {
                'pred': target_view[('overlap', frame_id, scale)],
                'target': inputs['color',0, 0][:,cam, ...]
            } 
            
            spatio_tempo_losses.append(compute_photometric_loss(**loss_args))
            spatio_tempo_masks.append(pred_mask)
        
        # concatenate losses and masks
        spatio_tempo_losses = torch.cat(spatio_tempo_losses, 1)
        spatio_tempo_masks = torch.cat(spatio_tempo_masks, 1)    

        # for the loss, take minimum value between reprojection loss and identity loss(moving object)
        # for the mask, take maximum value between reprojection mask and overlap mask to apply losses on all the True values of masks.
        spatio_tempo_loss, _ = torch.min(spatio_tempo_losses, dim=1, keepdim=True)
        spatio_tempo_mask, _ = torch.max(spatio_tempo_masks.float(), dim=1, keepdim=True)
     
        return compute_masked_loss(spatio_tempo_loss, spatio_tempo_mask) 
    
    def compute_pose_con_loss(self, inputs, outputs, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes pose consistency loss in "Full surround monodepth from multiple cameras"
        """        
        ref_output = outputs[('cam', 0)]
        ref_ext = inputs['extrinsics'][:, 0, ...]
        ref_ext_inv = inputs['extrinsics_inv'][:, 0, ...]
   
        cur_output = outputs[('cam', cam)]
        cur_ext = inputs['extrinsics'][:, cam, ...]
        cur_ext_inv = inputs['extrinsics_inv'][:, cam, ...] 
        
        trans_loss = 0.
        angle_loss = 0.
     
        for frame_id in self.frame_ids[1:]:
            ref_T = ref_output[('cam_T_cam', 0, frame_id)]
            cur_T = cur_output[('cam_T_cam', 0, frame_id)]    

            cur_T_aligned = ref_ext_inv@cur_ext@cur_T@cur_ext_inv@ref_ext

            ref_ang = matrix_to_euler_angles(ref_T[:,:3,:3], 'XYZ')
            cur_ang = matrix_to_euler_angles(cur_T_aligned[:,:3,:3], 'XYZ')

            ang_diff = torch.norm(ref_ang - cur_ang, p=2, dim=1).mean()
            t_diff = torch.norm(ref_T[:,:3,3] - cur_T_aligned[:,:3,3], p=2, dim=1).mean()

            trans_loss += t_diff
            angle_loss += ang_diff
        
        pose_loss = (trans_loss + 10 * angle_loss) / len(self.frame_ids[1:])
        return pose_loss

    def compute_gaussian_loss(self, inputs, target_view, cam=0, scale=0, frame_ids=[-1, 1]):
        """
        This function computes gaussian loss.
        """
        # self occlusion mask * overlap region mask
        
        gaussian_loss = 0.0 
        for frame_id in frame_ids:
            # import pdb; pdb.set_trace()
            pred = target_view[('gaussian_color', frame_id, scale)]
            gt = inputs['color', frame_id, 0][:,cam, ...]
            
            # pred = ((inputs['img_norm_cfg']['mean'].unsqueeze(-1).unsqueeze(-1) + pred * inputs['img_norm_cfg']['std'].unsqueeze(-1).unsqueeze(-1))/255.0).clip(0,1)
            # gt = ((inputs['img_norm_cfg']['mean'].unsqueeze(-1).unsqueeze(-1) + gt * inputs['img_norm_cfg']['std'].unsqueeze(-1).unsqueeze(-1))/255.0).clip(0,1)
            
            lpips_loss = self.lpips(pred, gt, normalize=True).mean()
            l2_loss = ((pred - gt)**2).mean()
            # l2_loss = self.l1loss(pred, gt)
            
            gaussian_loss += 1 * l2_loss + 0.05 * lpips_loss
            # gaussian_loss += 1 * l2_loss
        return gaussian_loss / 2


    def forward(self, inputs, cam, frame_ids):        
        loss_dict = {}
        cam_loss = 0. # loss across the multi-scale
        target_view = inputs[('cam', cam)]
        scale = 0
        kargs = {
            'cam': cam,
            'scale': scale,
            'ref_mask': inputs['mask'][:,cam,...]
        }
                        
        
        
        # kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
        gaussian_loss = self.compute_gaussian_loss(inputs, target_view, cam, scale, frame_ids)
        
            
        # cam_loss += reprojection_loss
        # cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)            
        # cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss
        cam_loss += self.gaussian_coeff * gaussian_loss                            
        
        # cam_loss += self.pose_loss_coeff* pose_loss
        
        ##########################
        # for logger
        ##########################
        loss_dict['gaussian_loss'] = gaussian_loss.item()
            
            # log statistics
            # self.get_logs(loss_dict, target_view, cam)                        
        # import pdb; pdb.set_trace()
        return cam_loss, loss_dict