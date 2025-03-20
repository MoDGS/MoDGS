#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def transfrom_by_Relative_Pose(train_cams,transformation_matrix,cam_info):
    """only for mono scene for adjutment of the camera pose

    Args:
        train_cam (_type_): _description_
        test_cams (_type_): _description_
        
    """
    # raise NotImplementedError()
    adjusted_cams = []
    for cam in train_cams:
        c2w=cam.view_world_transform.transpose(0, 1)
        transfromed_c2w = c2w@transformation_matrix
        # assert torch.allclose(torch.abs(transfromed_c2w[3,:3]).sum(),torch.tensor(0.0)), "must be 3by4 shape. last row [0,0,0,1]"
        world_view_transform = transfromed_c2w.inverse().transpose(0, 1)
        if not "cam" in cam_info.image_name:
            name = "cam" +"%02d"%(int(cam_info.image_name)+1)+"_" +cam.image_name
        else:
            name = cam_info.image_name+"_" +cam.image_name
        
        adjusted_cam  = MiniCam_Nvidia(
                name,
                image=cam.original_image, 
                FoVx = cam.FoVx,
                FoVy = cam.FoVy,
                image_width=cam.image_width,
                image_height =cam.image_height,
                world_view_transform=world_view_transform,
                projection_matrix=cam.projection_matrix,
                time=cam.time,
                        )
    
        adjusted_cams.append(adjusted_cam)
    return adjusted_cams
    
def get_realtive_pose(train_cam, test_cams,scale=1.0):
    """only for mono scene for adjutment of the camera pose

    Args:
        train_cam (_type_): _description_
        test_cams (_type_): _description_
    """
    c2w_train = train_cam.view_world_transform.transpose(0, 1)
    c2w_train[:3, 3] = c2w_train[:3, 3]/scale
    for test_cam in test_cams:
        
        c2w_test = test_cam.view_world_transform.transpose(0, 1)
        c2w_test[:3, 3] = c2w_test[:3, 3]/scale
        relative_pose = c2w_train.inverse()@c2w_test
        # relative_pose = train_cam.view_world_transform.transpose(0, 1).inverse()@test_cam.view_world_transform.transpose(0, 1)  
        test_cam.relative_pose = relative_pose## 3by4 
     
    return test_cams    
    pass 



class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, time=0,depth=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",mask=None,
                 **kwargs):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = torch.tensor(time).to(data_device) if time is not None else None
        self.kwargs = kwargs
        self.mask = torch.Tensor(mask).to(data_device) if mask is not None else None
        self.depth=torch.Tensor(depth).to(data_device) if depth is not None   else None
        # if self.mask is None:
        #     self.depth=torch.Tensor(depth).to(data_device) if depth is not None   else None
        # elif mask is not None:
        #     self.depth=torch.Tensor(depth).to(data_device)*self.mask if depth is not None  else None
        #     self.depth = (self.depth-self.depth.min())/(self.depth.max()-self.depth.min())
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0) # .to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        # self.image_width = 1386
        # self.image_height = 1014

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask # .to(self.data_device)
            if  (gt_alpha_mask<0.99999).sum()>0:
                print(image_name,"mask sum:",gt_alpha_mask.sum())
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width)) # , device=self.data_device)

        self.zfar = 100.0
        # self.znear = 0.01
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1) # .cuda() ## R 应该是w2c(3by4中的前三), 
        self.view_world_transform = self.world_view_transform.inverse() # .cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1) # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def get_depth_range(self):
        return np.array([self.depth[self.depth>1e-7].min().item(), self.depth.max().item()])

class Camera2(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, width, height,image_name,
                 uid, time=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 **kwargs):
        super(Camera2, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.time = time
        self.kwargs = kwargs
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = width
        self.image_height = height


        self.zfar = 100.0
        self.znear =  0.01
        # self.znear = 10 # 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1) # .cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1) # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.view_world_transform = self.world_view_transform.inverse() # .cuda()
        


class MiniCam_Nvidia:
    """Used from read relative cam """
    def __init__(self,
                name,
                image, 
                FoVx, FoVy,
                image_width,
                image_height,
                world_view_transform,
                projection_matrix,
                time=0,
                        ):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.time = time
        self.image_name = name
        self.original_image = image
        self.image_width = image_width
        self.image_height = image_height
        self.world_view_transform=world_view_transform
        self.projection_matrix= projection_matrix
    
        self.full_proj_transform=(self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center=camera_center
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.view_world_transform = self.world_view_transform.inverse() # .cuda()
        
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        pass 

   


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

