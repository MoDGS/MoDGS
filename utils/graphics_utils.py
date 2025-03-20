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
import math
import numpy as np
from typing import NamedTuple
import torch
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
def get_intrinsic_matrix(width, height, focal_length):
    # The intrinsic matrix is of the form:
    # [focal_length, 0, width/2]
    # [0, focal_length, height/2]
    # [0, 0, 1]
    intrinsic_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])
    return intrinsic_matrix
def project_from_rgbpcd(w2c,intrinsic,rgbpcd,img_shape):
    """rgbpcd: N*6"""
    
    """w2c:3by3 Matrix w2c @ Pworld[4,N] = Pc
        img_shape: (H,w) 
    RGBPCD:n*6 matrix,
    
    """
    H,W=img_shape
    N=rgbpcd.shape[0]
    rgb= rgbpcd[:,3:]
    pcd= rgbpcd[:,:3]
    pcd_home= np.concatenate([pcd, np.ones_like(pcd[...,:1])], axis=-1)
    
    pcd_cam_home = np.matmul(pcd_home,w2c.T)
    pcd_img =  np.matmul(pcd_cam_home[:,:3],intrinsic.T)
    pcd_img[:,:2]/=(pcd_img[:,2:]+1e-7)
    # print(pcd_img[:,2:].shape)
    # world_xyz = np.matmul(cam_xyz.reshape(-1,4), c2w.T)[...,:3]
    new_img = np.zeros([H,W,3])
    # new_depth = np.zeros([H,W])
    new_depth = np.full([H,W],np.inf)
    
    indY=np.floor(pcd_img[:,1]).astype(np.int32)
    Ymask = np.logical_and(indY<=H-1,indY>=-0.01)
    indX=np.floor(pcd_img[:,0]).astype(np.int32)
    Xmask = np.logical_and(indX<=W-1,indX>=-0.01)
    mask = np.logical_and(Xmask,Ymask)
    masked_depth =pcd_img[:,2][mask] 
    masked_color =rgb[:,:][mask]
    # print(pcd_img[:,2].shape)
    # print(mask.shape)
    
    new_depth[indY[mask],indX[mask]]=pcd_img[:,2][mask]
    new_img[indY[mask],indX[mask]]=rgb[:,:][mask]
    for i, (idx,idy) in enumerate(zip(indY[mask],indX[mask])):
        # ew_depth
        # print(i)
        # print(idx,idy)
        # print(new_depth[idx,idy])
        # print(mask_depth[i])
        # print(mask_depth.shape)
        
        if new_depth[idx,idy]> masked_depth[i]:
            new_depth[idx,idy]=masked_depth[i]
            new_img[idx,idy]=masked_color[i]
        # print(idx,idy)
    
    
    return pcd_img,new_depth,new_img
def reprojection2another_cam(uv,depth,cam1_c2w,cam2_c2w,intrinsic):
    """cam1_c2w:4by4 matrix:  cam1_c2w@P_c =P_w
    
    ## uv: 先x方向后y方向
    """
    
    ###Unprojecting uv at cam1 to world xyz
    ###
    img_xy=uv+0.5
    selected_depth = depth[uv[:,1],uv[:,0]] 
    print(img_xy)
    print(selected_depth)
    print(cam1_c2w,cam2_c2w)
    reverse_intrin = torch.linalg.inv(intrinsic).T
    cam_xy =  img_xy *  selected_depth[...,None]
    cam_xyz = torch.cat([cam_xy, selected_depth[...,None]], -1)
    cam_xyz = torch.matmul(cam_xyz, reverse_intrin)
    # mask_depth= cam_xyz[...,2]>1e-6
    cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], axis=-1)
    world_xyz = torch.matmul(cam_xyz.reshape(-1,4), cam1_c2w.T)[...,:4]
    
    ### projecting points at world xyz to cam1
    w2c= torch.linalg.inv(cam2_c2w)
    pcd_home= world_xyz
    pcd_cam_home = torch.matmul(pcd_home,w2c.T)
    pcd_img =  torch.matmul(pcd_cam_home[:,:3],intrinsic.T)
    
    
    projected_idx = pcd_img[:,:2]/(pcd_img[:,2:]+1e-7)
    
    
    # indY=torch.floor(pcd_img[:,1]).astype(np.int32)
    # Ymask = torch.logical_and(indY<=H-1,indY>=-0.01)
    # indX=torch.floor(pcd_img[:,0]).astype(np.int32)

    return  projected_idx


def project_from_rgbpcd_torch(w2c,intrinsic,rgbpcd,img_shape):
    """rgbpcd: N*6"""
    
    """w2c:3by3 Matrix w2c @ Pworld[4,N] = Pc
        img_shape: (H,w) 
    RGBPCD:n*6 matrix,
    
    """
    H,W=img_shape
    N=rgbpcd.shape[0]
    rgb= rgbpcd[:,3:]
    pcd= rgbpcd[:,:3]
    pcd_home= torch.cat([pcd, np.ones_like(pcd[...,:1])], axis=-1)
    
    pcd_cam_home = torch.matmul(pcd_home,w2c.T)
    pcd_img =  torch.matmul(pcd_cam_home[:,:3],intrinsic.T)
    pcd_img[:,:2]/=(pcd_img[:,2:]+1e-7)
    # print(pcd_img[:,2:].shape)
    # world_xyz = np.matmul(cam_xyz.reshape(-1,4), c2w.T)[...,:3]
    new_img = torch.zeros([H,W,3])
    # new_depth = np.zeros([H,W])
    new_depth = torch.full([H,W],torch.inf)
    
    indY=torch.floor(pcd_img[:,1]).astype(np.int32)
    Ymask = torch.logical_and(indY<=H-1,indY>=-0.01)
    indX=torch.floor(pcd_img[:,0]).astype(np.int32)
    Xmask = torch.logical_and(indX<=W-1,indX>=-0.01)
    mask = torch.logical_and(Xmask,Ymask)
    masked_depth =pcd_img[:,2][mask] 
    masked_color =rgb[:,:][mask]
    # print(pcd_img[:,2].shape)
    # print(mask.shape
    new_depth[indY[mask],indX[mask]]=pcd_img[:,2][mask]
    new_img[indY[mask],indX[mask]]=rgb[:,:][mask]
    for i, (idx,idy) in enumerate(zip(indY[mask],indX[mask])):
        # ew_depth
        # print(i)
        # print(idx,idy)
        # print(new_depth[idx,idy])
        # print(mask_depth[i])
        # print(mask_depth.shape)
        
        if new_depth[idx,idy]> masked_depth[i]:
            new_depth[idx,idy]=masked_depth[i]
            new_img[idx,idy]=masked_color[i]
        # print(idx,idy)
    
    
    return pcd_img,new_depth,new_img
def unproject_from_depthmap_torch(c2w,intrinsic,depth:torch.tensor,depth_mask=None):
    """depth: (h,w)"""
    """这个函数不对depth 为0的区域做mask，"""
    (h,w)=depth.shape
    px, py = torch.meshgrid(
        
        torch.arange(0, w, dtype=torch.float32),torch.arange(0, h, dtype=torch.float32),indexing='xy')
    # print(px.shape,px.max())
    # print(py.shape,py.max())
    img_xy = torch.stack([px+0.5, py+0.5], axis=-1).to(depth.device)
    # print(px)
    # print(px+0.5)
    reverse_intrin = torch.linalg.inv(intrinsic).T
    cam_xy =  img_xy * depth[...,None]
    cam_xyz = torch.cat([cam_xy, depth[...,None]], -1)
    cam_xyz = torch.matmul(cam_xyz, reverse_intrin)
    mask_depth= cam_xyz[...,2]>1e-6
    # cam_xyz = cam_xyz[mask_depth > 1e-7,:]
    cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], axis=-1)
    world_xyz = torch.matmul(cam_xyz.reshape(-1,4), c2w.T)[...,:3]
    return world_xyz,cam_xyz,img_xy,mask_depth

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))