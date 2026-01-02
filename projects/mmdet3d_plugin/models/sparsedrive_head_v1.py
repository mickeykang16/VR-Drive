from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head


@HEADS.register_module()
class V1SparseDriveHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        det_head = dict,
        map_head = dict,
        motion_plan_head = dict,
        init_cfg=None,
        **kwargs,
    ):
        super(V1SparseDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        if self.task_config['with_det']:
            self.det_head = build_head(det_head)
        if self.task_config['with_map']:
            self.map_head = build_head(map_head)
        if self.task_config['with_motion_plan']:
            self.motion_plan_head = build_head(motion_plan_head)

    def init_weights(self):
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
            self.map_head.init_weights()
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        original: bool
    ):
        if self.task_config['with_det']:
            det_output, instance_loss_det = self.det_head(feature_maps, metas, original)
        else:
            det_output = None

        if self.task_config['with_map']:
            map_output, instance_loss_map = self.map_head(feature_maps, metas, original)
        else:
            map_output = None
        
        if self.task_config['with_motion_plan']:
            motion_output, planning_output = self.motion_plan_head(
                det_output, 
                map_output, 
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
            )
        else:
            motion_output, planning_output = None, None
        # import pdb; pdb.set_trace()
        if (instance_loss_det is None) and (instance_loss_map is None):
            instance_loss = None
        elif (instance_loss_map is None):
            instance_loss = instance_loss_det
        elif (instance_loss_det is None):
            instance_loss = instance_loss_map
        else:
            instance_loss = (instance_loss_det + instance_loss_map)/2.0

        if instance_loss is not None:
            return [det_output, map_output, motion_output, planning_output], instance_loss
        else:
            return det_output, map_output, motion_output, planning_output

    def loss(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        losses = dict()
        if self.task_config['with_det']:
            loss_det = self.det_head.loss(det_output, data)
            losses.update(loss_det)
        
        if self.task_config['with_map']:
            loss_map = self.map_head.loss(map_output, data)
            losses.update(loss_map)

        if self.task_config['with_motion_plan']:
            motion_loss_cache = dict(
                indices=self.det_head.sampler.indices, 
            )
            loss_motion = self.motion_plan_head.loss(
                motion_output, 
                planning_output, 
                data, 
                motion_loss_cache
            )
            losses.update(loss_motion)
        
        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        if self.task_config['with_det']:
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)
        
        if self.task_config['with_map']:
            map_result= self.map_head.post_process(map_output)
            batch_size = len(map_result)

        if self.task_config['with_motion_plan']:
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_output,
                motion_output, 
                planning_output,
                data,
            )

        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_det']:
                results[i].update(det_result[i])
            if self.task_config['with_map']:
                results[i].update(map_result[i])
            if self.task_config['with_motion_plan']:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results
