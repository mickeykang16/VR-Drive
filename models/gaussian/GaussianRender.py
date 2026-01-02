
import torch
from .gaussian_renderer import render
from einops import rearrange

left_cam_dict = {2:0, 0:1, 4:2, 1:3, 5:4, 3:5}
right_cam_dict = {0:2, 1:0, 2:4, 3:1, 4:5, 5:3}

def get_adj_cams(cam):
    adj_cams = [cam]
    adj_cams.append(left_cam_dict[cam])
    adj_cams.append(right_cam_dict[cam])
    return adj_cams

def pts2render(inputs, cam_num, novel_cam, novel_frame_id, bg_color, mode='MF'):
    bs, _, height, width = inputs[('color', 0, 0)][:, novel_cam, ...].shape
    render_novel_list = []
    for i in range(bs):
        xyz_i_valid = []
        # rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        sh_i_valid = []
        frame_id = 0
        for cam in range(cam_num):
            valid_i = inputs[('cam', cam)][('pts_valid', frame_id, 0)][i, :]
            xyz_i = inputs[('cam', cam)][('xyz', frame_id, 0)][i, :, :]
            # rgb_i = inputs[('color', frame_id, 0)][:, cam, ...][i, :, :, :].permute(1, 2, 0).view(-1, 3) # HWC
            
            rot_i = inputs[('cam', cam)][('rot_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = inputs[('cam', cam)][('scale_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity_i = inputs[('cam', cam)][('opacity_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 1)
            sh_i = rearrange(inputs[('cam', cam)][('sh_maps', frame_id, 0)][i, :, :, :], "p srf r xyz d_sh -> (p srf r) d_sh xyz").contiguous()

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            # rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))
            sh_i_valid.append(sh_i[valid_i])


        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        # pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        # pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)
        sh_i = torch.concat(sh_i_valid, dim=0)

        novel_FovX_i = inputs[('cam', novel_cam)][('FovX', novel_frame_id, 0)][i]
        novel_FovY_i = inputs[('cam', novel_cam)][('FovY', novel_frame_id, 0)][i]
        novel_world_view_transform_i = inputs[('cam', novel_cam)][('world_view_transform', novel_frame_id, 0)][i]
        novel_function_proj_transform_i = inputs[('cam', novel_cam)][('full_proj_transform', novel_frame_id, 0)][i]
        novel_camera_center_i = inputs[('cam', novel_cam)][('camera_center', novel_frame_id, 0)][i]

        render_novel_i = render(novel_FovX=novel_FovX_i.float(),
                                novel_FovY=novel_FovY_i.float(),
                                novel_height=height,
                                novel_width=width,
                                novel_world_view_transform=novel_world_view_transform_i.float(),
                                novel_full_proj_transform=novel_function_proj_transform_i.float(),
                                novel_camera_center=novel_camera_center_i.float(),
                                pts_xyz=pts_xyz_i.float(), 
                                pts_rgb=None, 
                                rotations=rot_i.float(), 
                                scales=scale_i.float(), 
                                opacity=opacity_i.float(), 
                                shs=sh_i.float(), 
                                bg_color=bg_color)
        render_novel_list.append(render_novel_i.unsqueeze(0))

    
    novel = torch.concat(render_novel_list, dim=0)
    
    # novel = ((255.0 * novel.clip(0,1)) - inputs['img_norm_cfg']['mean'].unsqueeze(-1).unsqueeze(-1)) / inputs['img_norm_cfg']['std'].unsqueeze(-1).unsqueeze(-1)

    return novel
