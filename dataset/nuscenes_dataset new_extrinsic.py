import os

import numpy as np
import PIL.Image as pil

import torch
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .data_util import img_loader, mask_loader_scene, align_dataset, transform_mask_sample
from scipy.spatial.transform import Rotation as R
import pickle5 as pickle
from joblib import dump, load

from copy import deepcopy
import time


def is_numpy(data):
    """Checks if data is a numpy array."""
    return isinstance(data, np.ndarray)

def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor

def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list)

def stack_sample(sample):
    """Stack a sample from multiple sensors"""
    # If there is only one sensor don't do anything
    if len(sample) == 1:
        return sample[0]
    # import pdb; pdb.set_trace()
    # Otherwise, stack sample
    stacked_sample = {}
    for key in sample[0]:
        # Global keys (do not stack)
        if key in ['idx', 'dataset_idx', 'sensor_name', 'filename', 'token']:
            stacked_sample[key] = sample[0][key]
        else:
            # Stack torch tensors
            if is_tensor(sample[0][key]):
                stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
            # Stack numpy arrays
            elif is_numpy(sample[0][key]):
                stacked_sample[key] = np.stack([s[key] for s in sample], 0)
            # Stack list
            elif is_list(sample[0][key]):
                stacked_sample[key] = []
                # Stack list of torch tensors
                if is_tensor(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            torch.stack([s[key][i] for s in sample], 0))
                # Stack list of numpy arrays
                if is_numpy(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            np.stack([s[key][i] for s in sample], 0))

    # Return stacked sample
    return stacked_sample


def inverse_T(T):
    assert T.shape == (4, 4)
    R = T[:3, :3]
    R_inv = np.linalg.inv(R)
    t = T[:-1, -1]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:-1, -1] = -R_inv @ t
    return T_inv

class NuScenesdataset(Dataset):
    """
    Loaders for NuScenes dataset
    """
    def __init__(self, path, split,
                 cameras=None,
                 back_context=0,
                 forward_context=0,
                 data_transform=None,
                 depth_type=None,
                 scale_range=2,
                 with_pose=None,
                 with_ego_pose=None,
                 with_mask=None,
                 ):        
        super().__init__()
        version = 'v1.0-trainval'
        self.path = path
        # import pdb; pdb.set_trace()
        self.split = split
        self.dataset_idx = 0

        self.cameras = cameras
        self.scales = np.arange(scale_range+2) 
        self.num_cameras = len(cameras)

        self.bwd = back_context
        self.fwd = forward_context
        
        self.has_context = back_context + forward_context > 0
        self.data_transform = data_transform

        self.with_depth = depth_type is not None
        self.with_pose = with_pose
        self.with_ego_pose = with_ego_pose

        self.loader = img_loader

        self.with_mask = with_mask
        cur_path = os.path.dirname(os.path.realpath(__file__))        
        self.mask_path = os.path.join(cur_path, 'nuscenes_mask')
        self.mask_loader = mask_loader_scene

        nusc_filename = 'nusc.pkl'
        if os.path.isfile(nusc_filename):
            start = time.time()
            
            with open('nusc.pkl', 'rb') as f:
                self.dataset = pickle.load(f)
            
            # self.dataset = load(nusc_filename)
            
            end = time.time()

            print(f"Loaded pre-built nuScenes Object, which took: {end - start:.4f}s")
        else:
            self.dataset = NuScenes(version=version, dataroot=self.path, verbose=True)
            with open('nusc.pkl', 'wb') as f:
                pickle.dump(self.dataset, f)
            # dump(self.dataset, nusc_filename, compress=0)
        # list of scenes for training and validation of model
        with open('dataset/nuscenes/{}.txt'.format(self.split), 'r') as f:
            self.filenames = f.readlines()

    def get_current(self, key, cam_sample):
        """
        This function returns samples for current contexts
        """        
        # get current timestamp rgb sample
        if key == 'rgb':
            rgb_path = cam_sample['filename']
            return self.loader(os.path.join(self.path, rgb_path))
        # get current timestamp camera intrinsics
        elif key == 'intrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return np.array(cam_param['camera_intrinsic'], dtype=np.float32)
        # get current timestamp camera extrinsics
        elif key == 'extrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return self.get_tranformation_mat(cam_param)
        else:
            raise ValueError('Unknown key: ' +key)

    def get_context(self, key, cam_sample):
        """
        This function returns samples for backward and forward contexts
        """
        bwd_context, fwd_context = [], []
        if self.bwd != 0:
            if self.split == 'eval_SF': # validation
                bwd_sample = cam_sample
            else:
                bwd_sample = self.dataset.get('sample_data', cam_sample['prev'])
            bwd_context = [self.get_current(key, bwd_sample)]

        if self.fwd != 0:
            # fwd_sample = self.dataset.get('sample_data', cam_sample['next'])
            fwd_sample = cam_sample
            fwd_context = [self.get_current(key, fwd_sample)]
        return bwd_context + fwd_context
    
    def get_cam_T_cam(self, cam_sample):
        # cam 0 to world
        # cam 0 to ego 0
        cam_to_ego = self.dataset.get(
                'calibrated_sensor', cam_sample['calibrated_sensor_token'])
        cam_to_ego_rotation = Quaternion(cam_to_ego['rotation'])
        cam_to_ego_translation = np.array(cam_to_ego['translation'])[:, None]
        cam_to_ego = np.vstack([
            np.hstack((cam_to_ego_rotation.rotation_matrix,
                       cam_to_ego_translation)),
            np.array([0, 0, 0, 1])
            ])
        
        # ego 0 to world
        world_to_ego = self.dataset.get(
                'ego_pose', cam_sample['ego_pose_token'])
        world_to_ego_rotation = Quaternion(world_to_ego['rotation']).inverse
        world_to_ego_translation = - np.array(world_to_ego['translation'])[:, None]
        world_to_ego = np.vstack([
            np.hstack((world_to_ego_rotation.rotation_matrix,
                       world_to_ego_rotation.rotation_matrix @ world_to_ego_translation)),
            np.array([0, 0, 0, 1])
            ])
        ego_to_world = np.linalg.inv(world_to_ego)

        cam_T_cam = []

        # cam_T_cam, 0, -1
        if self.bwd != 0:
            
            if (self.split == 'eval_SF') or (cam_sample['prev'] == ''): # validation
                bwd_sample = cam_sample
            else:
                bwd_sample = self.dataset.get('sample_data', cam_sample['prev'])

            # world to ego -1
            world_to_ego_bwd = self.dataset.get(
                    'ego_pose', bwd_sample['ego_pose_token'])
            world_to_ego_bwd_rotation = Quaternion(world_to_ego_bwd['rotation']).inverse
            world_to_ego_bwd_translation = - np.array(world_to_ego_bwd['translation'])[:, None]
            world_to_ego_bwd = np.vstack([
                np.hstack((world_to_ego_bwd_rotation.rotation_matrix,
                           world_to_ego_bwd_rotation.rotation_matrix @ world_to_ego_bwd_translation)),
                np.array([0, 0, 0, 1])
                ])
            
            # ego -1 to cam -1
            cam_to_ego_bwd = self.dataset.get(
                    'calibrated_sensor', bwd_sample['calibrated_sensor_token'])
            cam_to_ego_bwd_rotation = Quaternion(cam_to_ego_bwd['rotation'])
            cam_to_ego_bwd_translation = np.array(cam_to_ego_bwd['translation'])[:, None]
            cam_to_ego_bwd = np.vstack([
                np.hstack((cam_to_ego_bwd_rotation.rotation_matrix,
                           cam_to_ego_bwd_translation)),
                np.array([0, 0, 0, 1])
                ])
            ego_to_cam_bwd = np.linalg.inv(cam_to_ego_bwd)

            cam_T_cam_bwd = ego_to_cam_bwd @ world_to_ego_bwd @ ego_to_world @ cam_to_ego

            cam_T_cam.append(cam_T_cam_bwd)

        # cam_T_cam, 0, 1
        if self.fwd != 0:
            if cam_sample['next'] == '':
                fwd_sample = cam_sample
            else:
                fwd_sample = self.dataset.get('sample_data', cam_sample['next'])
            
            # world to ego 1
            world_to_ego_fwd = self.dataset.get(
                    'ego_pose', fwd_sample['ego_pose_token'])
            world_to_ego_fwd_rotation = Quaternion(world_to_ego_fwd['rotation']).inverse
            world_to_ego_fwd_translation = - np.array(world_to_ego_fwd['translation'])[:, None]
            world_to_ego_fwd = np.vstack([
                np.hstack((world_to_ego_fwd_rotation.rotation_matrix,
                           world_to_ego_fwd_rotation.rotation_matrix @ world_to_ego_fwd_translation)),
                np.array([0, 0, 0, 1])
                ])
            
            # ego 1 to cam 1
            cam_to_ego_fwd = self.dataset.get(
                    'calibrated_sensor', fwd_sample['calibrated_sensor_token'])
            cam_to_ego_fwd_rotation = Quaternion(cam_to_ego_fwd['rotation'])
            cam_to_ego_fwd_translation = np.array(cam_to_ego_fwd['translation'])[:, None]
            cam_to_ego_fwd = np.vstack([
                np.hstack((cam_to_ego_fwd_rotation.rotation_matrix,
                           cam_to_ego_fwd_translation)),
                np.array([0, 0, 0, 1])
                ])
            ego_to_cam_fwd = np.linalg.inv(cam_to_ego_fwd)
            # import pdb; pdb.set_trace()
            cam_T_cam_fwd = ego_to_cam_fwd @ world_to_ego_fwd @ ego_to_world @ cam_to_ego

            cam_T_cam.append(cam_T_cam_fwd)

        return cam_T_cam

    def get_cam_T_cam_aug(self, cam_sample, rotation, translation, metas):
        # cam 0 to world
        # cam 0 to ego 0
        # import pdb; pdb.set_trace()
        
        cam_to_ego = self.dataset.get(
                'calibrated_sensor', cam_sample['calibrated_sensor_token'])
        cam_to_ego_rotation = Quaternion(cam_to_ego['rotation'])
        cam_to_ego_translation = np.array(cam_to_ego['translation'])[:, None]
        cam_to_ego = np.vstack([
            np.hstack((cam_to_ego_rotation.rotation_matrix,
                       cam_to_ego_translation)),
            np.array([0, 0, 0, 1])
            ])
        
        # ego 0 to world
        world_to_ego = self.dataset.get(
                'ego_pose', cam_sample['ego_pose_token'])
        world_to_ego_rotation = Quaternion(world_to_ego['rotation']).inverse
        world_to_ego_translation = - np.array(world_to_ego['translation'])[:, None]
        world_to_ego = np.vstack([
            np.hstack((world_to_ego_rotation.rotation_matrix,
                       world_to_ego_rotation.rotation_matrix @ world_to_ego_translation)),
            np.array([0, 0, 0, 1])
            ])
        ego_to_world = np.linalg.inv(world_to_ego)

        cam_T_cam = []

        # cam_T_cam, 0, -1
        if self.bwd != 0:
            # if self.split == 'eval_SF': # validation
            bwd_sample = cam_sample
            # else:
            #     bwd_sample = self.dataset.get('sample_data', cam_sample['prev'])
            # import pdb; pdb.set_trace()
            # world to ego -1
            world_to_ego_bwd = self.dataset.get(
                    'ego_pose', bwd_sample['ego_pose_token'])
            world_to_ego_bwd_rotation = Quaternion(world_to_ego_bwd['rotation']).inverse
            world_to_ego_bwd_translation = - np.array(world_to_ego_bwd['translation'])[:, None]
            world_to_ego_bwd = np.vstack([
                np.hstack((world_to_ego_bwd_rotation.rotation_matrix,
                           world_to_ego_bwd_rotation.rotation_matrix @ world_to_ego_bwd_translation)),
                np.array([0, 0, 0, 1])
                ])
            
            # ego -1 to cam -1
            cam_to_ego_bwd = self.dataset.get(
                    'calibrated_sensor', bwd_sample['calibrated_sensor_token'])
            cam_to_ego_bwd_rotation = Quaternion(cam_to_ego_bwd['rotation'])
            cam_to_ego_bwd_translation = np.array(cam_to_ego_bwd['translation'])[:, None]
                        
            cam_to_ego_bwd = np.vstack([
                np.hstack((cam_to_ego_bwd_rotation.rotation_matrix,
                           cam_to_ego_bwd_translation)),
                np.array([0, 0, 0, 1])
                ])
            ego_to_cam_bwd = np.linalg.inv(cam_to_ego_bwd)

            cam_T_cam_bwd = ego_to_cam_bwd @ world_to_ego_bwd @ ego_to_world @ cam_to_ego

            cam_T_cam.append(cam_T_cam_bwd)

        # cam_T_cam, 0, 1
        if self.fwd != 0:
            # fwd_sample = self.dataset.get('sample_data', cam_sample['next'])
            fwd_sample = cam_sample
            # import pdb; pdb.set_trace()
            # world to ego 1
            world_to_ego_fwd = self.dataset.get(
                    'ego_pose', fwd_sample['ego_pose_token'])
            world_to_ego_fwd_rotation = Quaternion(world_to_ego_fwd['rotation']).inverse
            world_to_ego_fwd_translation = - np.array(world_to_ego_fwd['translation'])[:, None]
            world_to_ego_fwd = np.vstack([
                np.hstack((world_to_ego_fwd_rotation.rotation_matrix,
                           world_to_ego_fwd_rotation.rotation_matrix @ world_to_ego_fwd_translation)),
                np.array([0, 0, 0, 1])
                ])
            
            # ego 1 to cam 1
            cam_to_ego_fwd = self.dataset.get(
                    'calibrated_sensor', fwd_sample['calibrated_sensor_token'])
            cam_to_ego_fwd_rotation = Quaternion(cam_to_ego_fwd['rotation'])
            cam_to_ego_fwd_translation = np.array(cam_to_ego_fwd['translation'])[:, None]
            
            # import pdb; pdb.set_trace()
            cam_name = fwd_sample.get('channel', None)
            shift = np.expand_dims(np.copy(translation), axis=-1)[[1, 0, 2], :]
            front_back_depth = shift[0, 0]
            # if cam_name == 'CAM_FRONT':
            
            #     shift[0, 0] = front_back_depth
            # elif cam_name == 'CAM_FRONT_LEFT' or cam_name == 'CAM_FRONT_RIGHT':
            #     shift[0, 0] = front_back_depth / 2.0
            # elif cam_name == 'CAM_BACK_LEFT' or cam_name ==  'CAM_BACK_RIGHT':
            #     shift[0, 0] = -front_back_depth / 2.0
            #     # import pdb; pdb.set_trace()
            # elif cam_name == 'CAM_BACK':
            #     shift[0, 0] = -front_back_depth
            # else:
            #     raise NotImplementedError
            # # import pdb; pdb.set_trace()          
            # cam_to_ego_fwd = np.vstack([
            #     np.hstack((cam_to_ego_fwd_rotation.rotation_matrix @ rotation,
            #                cam_to_ego_fwd_translation + shift)),
            #     np.array([0, 0, 0, 1])
            #     ])
            
            # ego_to_cam_fwd = np.linalg.inv(cam_to_ego_fwd)
            # cam_T_cam_fwd = ego_to_cam_fwd @ world_to_ego_fwd @ ego_to_world @ cam_to_ego
            # import pdb; pdb.set_trace()
            cam_idx = self.cameras.index(cam_name)
            cam_T_cam_fwd = inverse_T(metas['camaug2egos'][cam_idx]) @ metas['cam2egos'][cam_idx]
            cam_T_cam.append(cam_T_cam_fwd)

        return cam_T_cam, rotation, shift

    def generate_depth_map(self, sample, sensor, cam_sample):
        """
        This function returns depth map for nuscenes dataset,
        result of depth map is saved in nuscenes/samples/DEPTH_MAP
        """        
        # generate depth filename
        filename = '{}/{}.npz'.format(
                        os.path.join(os.path.dirname(self.path), 'samples'),
                        'DEPTH_MAP/{}/{}'.format(sensor, cam_sample['filename']))
                        
        # load and return if exists
        if os.path.exists(filename):
            # while True:
            try:
                return np.load(filename, allow_pickle=True)['depth']
            except:
                print('Broken file!: ', filename)
            #     import pdb; pdb.set_trace()
            #     pass
        if True:
            lidar_sample = self.dataset.get(
                'sample_data', sample['data']['LIDAR_TOP'])

            # lidar points                
            lidar_file = os.path.join(
                self.path, lidar_sample['filename'])
            lidar_points = np.fromfile(lidar_file, dtype=np.float32)
            lidar_points = lidar_points.reshape(-1, 5)[:, :3]

            # lidar -> world
            lidar_pose = self.dataset.get(
                'ego_pose', lidar_sample['ego_pose_token'])
            lidar_rotation= Quaternion(lidar_pose['rotation'])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1])
            ])

            # lidar -> ego
            sensor_sample = self.dataset.get(
                'calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            lidar_to_ego_rotation = Quaternion(
                sensor_sample['rotation']).rotation_matrix
            lidar_to_ego_translation = np.array(
                sensor_sample['translation']).reshape(1, 3)

            ego_lidar_points = np.dot(
                lidar_points[:, :3], lidar_to_ego_rotation.T)
            ego_lidar_points += lidar_to_ego_translation

            homo_ego_lidar_points = np.concatenate(
                (ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1)


            # world -> ego
            ego_pose = self.dataset.get(
                    'ego_pose', cam_sample['ego_pose_token'])
            ego_rotation = Quaternion(ego_pose['rotation']).inverse
            ego_translation = - np.array(ego_pose['translation'])[:, None]
            world_to_ego = np.vstack([
                    np.hstack((ego_rotation.rotation_matrix,
                               ego_rotation.rotation_matrix @ ego_translation)),
                    np.array([0, 0, 0, 1])
                    ])

            # Ego -> sensor
            sensor_sample = self.dataset.get(
                'calibrated_sensor', cam_sample['calibrated_sensor_token'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(
                sensor_sample['translation'])[:, None]
            sensor_to_ego = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, 
                           sensor_translation)),
                np.array([0, 0, 0, 1])
               ])
            ego_to_sensor = np.linalg.inv(sensor_to_ego)
            
            # lidar -> sensor
            lidar_to_sensor = ego_to_sensor @ world_to_ego @ lidar_to_world
            homo_ego_lidar_points = torch.from_numpy(homo_ego_lidar_points).float()
            cam_lidar_points = np.matmul(lidar_to_sensor, homo_ego_lidar_points.T).T

            # depth > 0
            depth_mask = cam_lidar_points[:, 2] > 0
            cam_lidar_points = cam_lidar_points[depth_mask]

            # sensor -> image
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = sensor_sample['camera_intrinsic']
            pixel_points = np.matmul(intrinsics, cam_lidar_points.T).T
            pixel_points[:, :2] /= pixel_points[:, 2:3]
            
            # load image for pixel range
            image_filename = os.path.join(
                self.path, cam_sample['filename'])
            img = pil.open(image_filename)
            h, w, _ = np.array(img).shape
            
            # mask points in pixel range
            pixel_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] <= w-1)\
                        & (pixel_points[:,1] >= 0) & (pixel_points[:,1] <= h-1)
            valid_points = pixel_points[pixel_mask].round().int()
            valid_depth = cam_lidar_points[:, 2][pixel_mask]
        
            depth = np.zeros([h, w])
            depth[valid_points[:, 1], valid_points[:,0]] = valid_depth
        
            # save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
            return depth

    def get_tranformation_mat(self, pose):
        """
        This function transforms pose information in accordance with DDAD dataset format
        """
        extrinsics = Quaternion(pose['rotation']).transformation_matrix
        extrinsics[:3, 3] = np.array(pose['translation'])
        return extrinsics.astype(np.float32)

    def __len__(self):
        return len(self.filenames)
    
    # def __getitem__(self, frame_idx, idx):
    #     # get nuscenes dataset sample
    #     # frame_idx = self.filenames[idx].strip().split()[0]
    #     sample_nusc = self.dataset.get('sample', frame_idx)
    #     # import pdb; pdb.set_trace()
    #     sample = []
    #     contexts = []
    #     if self.bwd:
    #         contexts.append(-1)
    #     if self.fwd:
    #         contexts.append(1)

    #     # loop over all cameras            
    #     for cam in self.cameras:
    #         cam_sample = self.dataset.get(
    #             'sample_data', sample_nusc['data'][cam])

    #         data = {
    #             'idx': idx,
    #             'token': frame_idx,
    #             'sensor_name': cam,
    #             'contexts': contexts,
    #             'filename': cam_sample['filename'],
    #             'rgb': self.get_current('rgb', cam_sample), # (H, W, 3)
    #             'intrinsics': self.get_current('intrinsics', cam_sample) # (3, 3)
    #         }

    #         # if depth is returned            
    #         if self.with_depth:
    #             data.update({
    #                 'depth': self.generate_depth_map(sample_nusc, cam, cam_sample) # (H, W)
    #             })
    #         # if pose is returned
    #         if self.with_pose:
    #             data.update({
    #                 'extrinsics':self.get_current('extrinsics', cam_sample) # (4, 4)
    #             })
            
    #         # if ego_pose is returned
    #         if self.with_ego_pose:
    #             data.update({
    #                 'ego_pose': self.get_cam_T_cam(cam_sample)
    #             })
    #         # if mask is returned
    #         if self.with_mask:
    #             data.update({
    #                 'mask': self.mask_loader(self.mask_path, '', cam) # (H, W)
    #             })        
    #         # if context is returned
    #         if self.has_context:
    #             data.update({
    #                 'rgb_context': self.get_context('rgb', cam_sample) # []
    #             })

    #         sample.append(data)

    #     # import pdb; pdb.set_trace()
    #     # apply same data transformations for all sensors
    #     if self.data_transform:
    #         sample = [self.data_transform(smp) for smp in sample]
    #         sample = [transform_mask_sample(smp, self.data_transform) for smp in sample]

    #     # sample_ = sample.copy()
    #     # stack and align dataset for our trainer
    #     sample = stack_sample(sample)
    #     sample = align_dataset(sample, self.scales, contexts)
    #     # import pdb; pdb.set_trace()
    #     return sample
    
    def preprocess(self, data):
        # add file_path
        
        forward_dict = data.get('forward_ref', {})
        save_dict = {'token': data['token'],
            'lidar2ego_rotation': data['lidar2ego_rotation'],
            'lidar2ego_translation': data['lidar2ego_translation'],
            'cam2egos':data['cam2egos'],
            'camaug2egos':data['camaug2egos'],
            # 'img' :data['img']
            }
        forward_dict.update(save_dict)
        
        sample = data['token']
        file_list = []
        sample_nusc = self.dataset.get('sample', sample)
        
        if self.bwd != 0:
            for cam in self.cameras:
                cam_sample = self.dataset.get(
                    'sample_data', sample_nusc['data'][cam])
                if (self.split == 'eval_SF') or (cam_sample['prev'] == ''): # validation
                    bwd_sample = cam_sample
                else:
                    bwd_sample = self.dataset.get('sample_data', cam_sample['prev'])
                file_list.append(os.path.join(self.path, bwd_sample['filename']))
                
        if self.fwd != 0:
            for cam in self.cameras:
                cam_sample = self.dataset.get(
                    'sample_data', sample_nusc['data'][cam])
                
                # if (self.split == 'eval_SF') or (cam_sample['next'] == ''):
                if cam_sample['next'] == '':
                    fwd_sample = cam_sample
                else:
                    fwd_sample = self.dataset.get('sample_data', cam_sample['next'])

                file_list.append(os.path.join(self.path, fwd_sample['filename']))
        
        mask = []
        for cam in self.cameras: mask.append(torch.tensor(np.asarray(self.mask_loader(self.mask_path, '', cam))))
        
        # aug_rotation, aug_translation = self.cam_pos_augmentor()
        forward_dict['aug_rotation'] = data['aug_rotation']
        forward_dict['aug_translation'] = data['aug_translation']
        
        data['img_filename'] = data['img_filename'] + file_list
        # import pdb; pdb.set_trace()
        data['forward_ref'] = forward_dict
        data['mask'] = mask
        return data
    
    def postprocess(self, out):
        
        # dict_keys(['idx', 'token', 'sensor_name', 'filename', 'depth', 'extrinsics', 'mask', ('K', 0), ('inv_K', 0), ('color', 0, 0), ('color_aug', 0, 0), ('K', 1), ('inv_K', 1), ('color', 0, 1), ('color_aug', 
        # 0, 1), ('K', 2), ('inv_K', 2), ('color', 0, 2), ('color_aug', 0, 2), ('K', 3), ('inv_K', 3), ('color', 0, 3), ('color_aug', 0, 3), ('color', -1, 0), ('color_aug', -1, 0), ('cam_T_cam', 0, -1), ('color',
        # 1, 0), ('color_aug', 1, 0), ('cam_T_cam', 0, 1)])
        # import pdb; pdb.set_trace()
        
        img_orig = out.pop('img_orig', None)
        context_img = out.pop('context_img', None)
        mask = out.pop('mask', None)
        # mask = torch.stack(mask, axis=1).unsqueeze(1)
        forward_ref = out.pop('forward_ref', None)
        intrinsic = out.pop('cam_intrinsic', None)
        # gt_depth_aug = out.pop('gt_depth_aug', None)
        gt_depth_aug = out.get('gt_depth_aug', None)
        
        if 'img' not in out: return out
        
        img_orig = img_orig.transpose([0, 2, 3, 1])
        if context_img is not None:
            # context_img = np.array(deepcopy(context_img).permute([0, 2, 3, 1]))
            context_img = img_orig[6:]
        
        # rgb = out['img']
        
        # rgb = np.array(deepcopy(out['img'].data).permute([0, 2, 3, 1])) # tensor(6, 3, H, W) -> array(6, H, W ,3)
        rgb = img_orig[:6]
        
        token = forward_ref['token']
        
        # pad_intrinsic = []
        # for i in intrinsic:
        #     eye = np.eye(4)
        #     eye[:3, :3] = i
        #     pad_intrinsic.append(eye)
        # pad_intrinsic = np.stack(pad_intrinsic, axis=0)
        
        # extrinsics = []
        # for i, cam in enumerate(self.cameras):
        #     viewpad = pad_intrinsic[i]
        #     lidar2img_rt = out['projection_mat'][i]
        #     lidar2cam = (np.linalg.inv(viewpad) @ lidar2img_rt)
            
        #     lidar2ego = np.eye(4)
        #     lidar2ego[:3, :3] = Quaternion(
        #         forward_ref['lidar2ego_rotation']
        #     ).rotation_matrix
        #     lidar2ego[:3, 3] = np.array(forward_ref['lidar2ego_translation'])
        #     # extrinsic = data['forward_ref']['extrinsics'][0]
        #     cam2ego = lidar2ego @ inverse_T(lidar2cam)
        #     extrinsics.append(cam2ego)
        # extrinsics = np.stack(extrinsics, axis=0)
        
        # forward_dict = {
        #     'token': token,
        #     'mask': mask,
        #     'depth': out['gt_depth'],
        #     'extrinsics': extrinsics,
        #     ('K', 0): pad_intrinsic, 
        #     ('inv_K', 0): np.linalg.pinv(pad_intrinsic).copy()
        # }
        
        sample_nusc = self.dataset.get('sample', token)
        # import pdb; pdb.set_trace()
        sample = []
        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)

        # loop over all cameras            
        for i, cam in enumerate(self.cameras):
            cam_sample = self.dataset.get(
                'sample_data', sample_nusc['data'][cam])

            data = {
                # 'token': np.array(token),
                # 'sensor_name': np.array(cam),
                # 'filename': cam_sample['filename'],
                # 'rgb': self.get_current('rgb', cam_sample),
                'rgb': rgb[i],
                'intrinsics': intrinsic[i]
            }

            # if depth is returned            
            if self.with_depth:
                if 'gt_depth' in out:
                    depth = out['gt_depth'][0][i]
                else: depth = np.zeros(data['rgb'].shape[:2])
                data.update({
                    'depth':  depth # full scale depth
                })
                if gt_depth_aug is not None:
                    depth_aug = gt_depth_aug[0][i]
                data.update({
                    'depth_aug':  depth_aug # full scale depth
                })
                # import pdb; pdb.set_trace()
            # if pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics':self.get_current('extrinsics', cam_sample)
                })
            
            # if ego_pose is returned
            if self.with_ego_pose:
                data.update({
                    'ego_pose': self.get_cam_T_cam(cam_sample),
                })
                if gt_depth_aug is not None:
                    data.update({
                        'ego_pose_aug': self.get_cam_T_cam_aug(cam_sample, 
                                                           forward_ref['aug_rotation'], 
                                                           forward_ref['aug_translation'],
                                                           forward_ref)[0]
                    })
            # if mask is returned
            if self.with_mask:
                mask = [np.array(m) for m in mask]
                data.update({
                    'mask': mask[i]
                })        
            # if context is returned
            if self.has_context:
                data.update({
                    'contexts': contexts,
                })
                rgb_context = []
                for c in range (len(contexts)):
                    rgb_context.append(context_img[c * len(self.cameras) + i])
                data.update({
                    'rgb_context': rgb_context
                })

            sample.append(data)

        # apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]
            sample = [transform_mask_sample(smp, self.data_transform) for smp in sample]

        # sample_ = sample.copy()
        # stack and align dataset for our trainer
        # import pdb; pdb.set_trace()
        sample = stack_sample(sample)
        forward = align_dataset(sample, self.scales, contexts)
        # out.update(forward)
        out['forward'] = forward
        
        # import pdb; pdb.set_trace()
        
        return out
        
        # import pdb; pdb.set_trace()
        # data['forward'] = forward_dict
        # return data