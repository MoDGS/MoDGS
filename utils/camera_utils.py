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
import os

import torch
from scene.cameras import Camera, Camera2
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import copy
from utils.system_utils import resize_flow
import cv2
from tqdm import tqdm
WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_mask= cv2.resize(cam_info.mask, (resolution[0],resolution[1])) if cam_info.mask is not None else None
    if cam_info.image_path.split('/')[-3]=="rgb" or cam_info.image_path.split('/')[-3].startswith("rgb"): ## Hyper NeRF 或者iphone 数据集, 
        data_root = '/'.join(cam_info.image_path.split('/')[:-3])
    else:### ## 其他数据集
        data_root = '/'.join(cam_info.image_path.split('/')[:-2])
    folder = cam_info.image_path.split('/')[-2]
    
    image_name =  cam_info.image_path.split('/')[-1]
    # flow_folder = "flow_midas1"
    if "interlval1" in cam_info.image_path and os.path.exists(os.path.join(data_root, 'flow_RAFT1_interlval1')):
        flow_folder = "flow_RAFT1_interlval1"
    elif os.path.exists(os.path.join(data_root, 'flow_midas1')):
        flow_folder = "flow_midas1"
    elif os.path.exists(os.path.join(data_root, 'flow_RAFT1')):
        flow_folder = "flow_RAFT1"
    else:
        flow_folder = "flow_dontexist"
    
    ## Qingming 
    flow_folder = "None" ## TODO: 为了不用flow, 先设置为None
    
    
    fwd_flow_path = os.path.join(data_root, flow_folder, f'{os.path.splitext(image_name)[0]}_fwd.npz')
    bwd_flow_path = os.path.join(data_root, flow_folder, f'{os.path.splitext(image_name)[0]}_bwd.npz')
    # print(fwd_flow_path, bwd_flow_path)  
    if os.path.exists(fwd_flow_path):
        fwd_data = np.load(fwd_flow_path)
        fwd_flow = fwd_data['flow']
        fwd_flow_mask = fwd_data['mask']
        if fwd_flow.shape[0] != resolution[1] or fwd_flow.shape[1] != resolution[0]:
            fwd_flow = resize_flow(fwd_flow, resolution[1], resolution[0]) # h,w
            fwd_flow_mask = cv2.resize(fwd_flow_mask.astype(np.uint8), (resolution[0], resolution[1]), cv2.INTER_LINEAR)
        fwd_flow = torch.from_numpy(fwd_flow)
        fwd_flow_mask = torch.from_numpy(fwd_flow_mask)
    else:
        fwd_flow, fwd_flow_mask  = None, None
    if os.path.exists(bwd_flow_path):
        bwd_data = np.load(bwd_flow_path)
        bwd_flow = bwd_data['flow']
        bwd_flow_mask = bwd_data['mask']
        if bwd_flow.shape[0] != resolution[1] or bwd_flow.shape[1] != resolution[0]:
            bwd_flow = resize_flow(bwd_flow, resolution[1], resolution[0]) # h,w
            bwd_flow_mask = cv2.resize(bwd_flow_mask.astype(np.uint8), (resolution[0], resolution[1]), cv2.INTER_LINEAR)
        bwd_flow = torch.from_numpy(bwd_flow)
        bwd_flow_mask = torch.from_numpy(bwd_flow_mask)
    else:
        bwd_flow, bwd_flow_mask  = None, None
        
    ## LQM:logic_and for flow mask and alpha mask to get the final flow mask
    if bwd_flow is not None and resized_mask is not None:
        bwd_flow = bwd_flow*torch.tensor(resized_mask)[...,None]
        # bwd_flow_mask = torch.logical_and(bwd_flow_mask, torch.tensor(cam_info.mask))
    if fwd_flow is not None and resized_mask is not None:
        fwd_flow = fwd_flow*torch.tensor(resized_mask)[...,None]
        # fwd_flow_mask = torch.logical_and(fwd_flow_mask, torch.tensor(cam_info.mask))
    resized_depth=None
    if cam_info.depth is not None :
        resized_depth=cam_info.depth 
        if not cam_info.depth.squeeze().shape==(resolution[1],resolution[0]):
            resized_depth=cv2.resize(resized_depth, (resolution[0],resolution[1]))
    else :
        depth_folder_postfit=  folder[6:]     ## 获取/2x 或者1x   
        if args.depth_folder =="None":## if not specified, use the default depth folder
            depth_path_png=os.path.join(data_root, 'depth'+depth_folder_postfit, f'{os.path.splitext(image_name)[0]}.png')
        else:
            depth_path_png=os.path.join(data_root, args.depth_folder, f'{os.path.splitext(image_name)[0]}.png')
        if os.path.exists(depth_path_png):
            depth= cv2.imread(depth_path_png,cv2.IMREAD_UNCHANGED)
            resized_depth=cv2.resize(depth, (resolution[0],resolution[1]))
        else:## for Iphone dataset they has gt depth stored in npy file,
            depth_path_npy=os.path.join(data_root, 'depth',cam_info.image_path.split('/')[-2], f'{os.path.splitext(image_name)[0]}.npy')
            if os.path.exists(depth_path_npy):
                depth= np.load(depth_path_npy)*cam_info.coord_scale ## FIXME:  chekc这里是不是需要乘这个coord_scale.需要吗？
                resized_depth=cv2.resize(depth, (resolution[0],resolution[1]))
            del depth_path_npy
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,depth=resized_depth,
                  image_name=cam_info.image_name, uid=id, time=cam_info.time, data_device=args.data_device,
                  fwd_flow=fwd_flow, fwd_flow_mask=fwd_flow_mask,
                  bwd_flow=bwd_flow, bwd_flow_mask=bwd_flow_mask,
                  dict_other=cam_info.dict_other if hasattr(cam_info,"dict_other") else None, ### dict other 存储的是 Exhaustive raft info
                  mask=resized_mask if cam_info.mask is not None else None)

def loadCam2(args, id, cam_info, resolution_scale):
    
    # pass 
    # _,height,wight=cam_info.original_image.shape
    orig_w,orig_h = cam_info.image.size
    # return Camera2(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
    #                FoVx=cam_info.FovX, FoVy=cam_info.FovY, width=cam_info.width, height=cam_info.height,
    #                uid=id, time=cam_info.time, data_device=args.data_device)
    if args.resolution ==-1:
        resolution=1
    else:
        resolution=args.resolution
    resolution = round(orig_w/(resolution_scale * resolution)), round(orig_h/(resolution_scale *resolution))
    return Camera2(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, image_name=cam_info.image_name,
                   FoVx=cam_info.FovX, FoVy=cam_info.FovY, width=resolution[0], height=resolution[1],
                   uid=id, time=cam_info.time, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args,pcd_interval=1):
    camera_list = []

    for id, c in tqdm(enumerate(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    # LQM: add link to previous and next camera.
    for i in range(len(camera_list)):
        if not i == 0:## first one 
            # camera_list[i].prev = copy.deepcopy(camera_list[i-1])
            camera_list[i].prev = camera_list[i-1]
            try:
                assert int(camera_list[i].prev.image_name[-3:])==int(camera_list[i].image_name[-3:])-pcd_interval  ## 考虑dynerf的情况，image name不一定是数字
            except AssertionError as e:
                print("MISSMATCH PreviousLink: ",camera_list[i].prev.image_name, camera_list[i].image_name)
                # camera_list[i].prev = None
        else:
            camera_list[i].prev = None
        if not i == len(camera_list)-1: ## last one   
            # camera_list[i].next = copy.deepcopy(camera_list[(i+1)%len(camera_list)])
            camera_list[i].next = camera_list[(i+1)%len(camera_list)]
            try:
                assert int(camera_list[i].next.image_name[-3:])==int(camera_list[i].image_name[-3:])+pcd_interval
            except AssertionError as e:
                print("MISSMATCH NextLink: ",camera_list[i].next.image_name, camera_list[i].image_name)
                # camera_list[i].next = None
        else:
            camera_list[i].next = None
        
    return camera_list

def cameraList_from_camInfos_without_image(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam2(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
