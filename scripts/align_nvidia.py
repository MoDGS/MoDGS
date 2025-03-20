import os
from PIL import Image
import torch
import torch.nn as nn
import sys
# Get current working directory
current_directory = os.getcwd()

print('Current directory is: ', current_directory)
sys.path.append("./")
sys.path.append("./../")
print(sys.path)
from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary
import numpy as np
from utils.graphics_utils import unproject_from_depthmap_torch,get_intrinsic_matrix
from scene.dataset_readers import readDyColmapSceneInfo,readSelfMadeSceneInfo
from utils.graphics_utils import getWorld2View2
from PIL import Image
import json
from tqdm import tqdm
import os


def reprojection2another_cam(uv,depth,cam1_c2w,cam2_c2w,intrinsic):
    """cam1_c2w:4by4 matrix:  cam1_c2w@P_c =P_w
    
    ## uv: 先x方向后y方向
    """
    
    ###Unprojecting uv at cam1 to world xyz
    ###
    img_xy=uv+0.5
    selected_depth = depth[uv[:,1],uv[:,0]] 
    # print(img_xy)
    # print(selected_depth)
    # print(cam1_c2w,cam2_c2w)
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
    print("imgshape",img_shape)
    
    H,W=img_shape
    print("HW",H,W)
    N=rgbpcd.shape[0]
    rgb= rgbpcd[:,3:]
    pcd= rgbpcd[:,:3]
    pcd_home= torch.cat([pcd, torch.ones_like(pcd[...,:1])], axis=-1)
    
    pcd_cam_home = torch.matmul(pcd_home,w2c.T)
    pcd_img =  torch.matmul(pcd_cam_home[:,:3],intrinsic.T)
    pcd_img[:,:2]/=(pcd_img[:,2:]+1e-7)
    # print(pcd_img[:,2:].shape)
    # world_xyz = np.matmul(cam_xyz.reshape(-1,4), c2w.T)[...,:3]
    new_img = torch.zeros([H,W,3])
    # new_depth = np.zeros([H,W])
    new_depth = torch.full([H,W],torch.inf)
    
    indY=torch.floor(pcd_img[:,1]).to(torch.long)
    Ymask = torch.logical_and(indY<=H-1,indY>=-0.01)
    indX=torch.floor(pcd_img[:,0]).to(torch.long)
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





if __name__ == "__main__":
    
    ### read scene cames.
    
    
    ## Train cam _index
    # # Ballloon2-2:
    # input_idx= torch.tensor([[682,301],
    #                         [577,265],
    #                         ]).to(torch.long)
    # target_idx=torch.tensor([[712,286],
    #                         [608,249]
    #                         ]).cuda()
    # path = "XXXXX/Balloon2-2/dense"
    ## Ballloon1-2:
    # input_idx= torch.tensor([[523,35],
    #                         [463,266],
    #                         ]).to(torch.long)
    # ## cam3
    # target_idx=torch.tensor([[575,53],
    #                         [538,282],
    #                         ]).cuda()
    # path = "XXXXX/Balloon1-2/dense"
    # ## Skating-2:
    # input_idx= torch.tensor([[366,412],
    #                         [1236,516],
    #                         [488,340],
    #                         ]).to(torch.long)/2.0
    # ## cam3
    # target_idx=torch.tensor([[320,462],
    #                         [1194,565],
    #                         [461,389]
    #                         ]).cuda()/2.0
    # path = "XXXXX/Skating-2/dense"
    # ## Playground:
    # input_idx= torch.tensor([[583,448],
    #                          [1258,229],
    #                          [1020,407],
    #                         ]).to(torch.long)/2.0
    # ## cam3
    # target_idx=torch.tensor([[559,512],
    #                          [1220,300],
    #                          [1021,477]
    #                         ]).cuda()/2.0
    # path = "XXXXX/Playground/dense"
    # ## DynamicFace-2:
    # input_idx= torch.tensor([[957,374],
    #                          [970,819],
    #                         #  [317,679]
    #                         ]
    #                          ).to(torch.long)/2.0
    # ## cam3
    # target_idx=torch.tensor([[760,484],
    #                          [777,931],
    #                         #  [197,789]
    #                         ]).cuda()/2.0
    # path = "XXXXX/DynamicFace-2/dense"
    
    # ## Truck-2:
    # input_idx= torch.tensor([[1036,636],
    #                           [1253,663],
    #                           [1173,363]]
    #                          ).to(torch.long)/2.0
    # ## cam3
    # target_idx=torch.tensor([[1030,634],
    #                           [1248,660],
    #                           [1174,361]
    #                         ]).cuda()/2.0
    # path = "XXXXX/Truck-2/dense/"
    # ## Jumping:
    # input_idx= torch.tensor([[1052,521],
    #                          [1430,427],
    #                          [488,343]]
    #                          ).to(torch.long)/2.0
    # ## cam3
    # target_idx=torch.tensor([[1005,568],
    #                          [1377,472],
    #                          [461,391]
    #                         ]).cuda()/2.0
    # path = "XXXXX/Jumping/dense/"
    ## umbrella
    input_idx= torch.tensor([[677,206],
                             [639,488],
                             [1204,260]]
                             ).to(torch.long)/2.0
    ## cam3
    target_idx=torch.tensor([[699,183],
                             [667,469],
                             [1196,232]
                            ]).cuda()/2.0
    path = "XXXXXX/nvidia_data_full/Umbrella/dense/"
    
    input_idx= input_idx.cuda().to(torch.long)
    target_idx= target_idx.cuda().to(torch.long)
    ### Read Colmap Camera Info
    ### Read Colmap Camera Info
    ### Read Colmap Camera Info
    cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    images= None
    image_mode ="fg_bg"
    eval=True
    re_scale_json=None
    if os.path.exists(os.path.join(path,"re_scale.json")):
        print(f"Found re_scale.json, will load it")
        import torch
        with open(os.path.join(path,"re_scale.json"), 'r') as json_file:
            dict_rescale = json.load(json_file)
        print("rescale_jason_info:",dict_rescale)
        re_scale_json= dict_rescale
    scene_info_imgs =readSelfMadeSceneInfo(path, images="rgb/2x", eval=True,re_scale_json=re_scale_json,exhaustive_training=False,use_depthNonEdgeMsk=False)
    scene_info = readDyColmapSceneInfo(path, images,image_mode,eval,initPcdFromfirstframeDepth=False)
    
    ### Read Scene Info
    width = scene_info_imgs.train_cameras[0].width
    height = scene_info_imgs.train_cameras[0].height
    print(width,height)
    # focal_lengthX=806
    focal_lengthX=800
    intrinsic=torch.tensor(get_intrinsic_matrix(width=width, height= height, focal_length=focal_lengthX ))
    cam4 = scene_info.train_cameras[3] ## Training Camera
    cam3 = scene_info.train_cameras[2]
    cam5 = scene_info.train_cameras[4]
    c2w_traincam=np.linalg.inv(getWorld2View2(cam4.R, cam4.T))
    c2w_testcam3=np.linalg.inv(getWorld2View2(cam3.R, cam3.T))
    c2w_testcam5=np.linalg.inv(getWorld2View2(cam5.R, cam5.T))
    depth= torch.tensor(scene_info_imgs.train_cameras[0].depth)
    print("depth max min:",depth.max(),depth.min())
    
    
    
    
    ## optimize the colmap scene scale to match the dpeth scale.
    input_depth = depth.cuda()
    s= nn.Parameter(torch.Tensor([30.]).cuda().requires_grad_(True))
    optimizer = torch.optim.Adam([
                {'params':s, 'lr':1e-1},])
    last_step_loss = 100000.
    progress_bar = tqdm(range(10000), desc="Optimizing Scale  progress")
    
    for step in range(10000):
        optimizer.zero_grad()
        scaled_depth = s*input_depth
        # depth = s*input_depth+t
        idx_pre = reprojection2another_cam(input_idx.cuda(),scaled_depth,torch.tensor(c2w_traincam).cuda(),torch.tensor(c2w_testcam3).cuda(),torch.tensor(intrinsic).to(torch.float32).cuda())

        loss = torch.nanmean(torch.abs(idx_pre-target_idx))
        loss.backward()
        optimizer.step()
        if step%40==0:
            if abs(loss.item()-last_step_loss)<1e-5:
                break
            last_step_loss = loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.{3}f}","s":f"{s.item():.{2}f}"})
            progress_bar.update(40)
            # print("loss：",loss.item())
            # print("s:",s.item())
    print("optimized s:",s.item())
    print("loss:",loss.item())
    
    
    idx_pre_vali  =reprojection2another_cam(input_idx.cuda(),scaled_depth,torch.tensor(c2w_traincam).cuda(),torch.tensor(c2w_testcam5).cuda(),torch.tensor(intrinsic).to(torch.float32).cuda())
    print(idx_pre_vali)
    
    with open (os.path.join(path,"sparse","colmap_to_depth_scale.json"), 'w') as json_file:
        # json.dump({"scale":s.item(),"valid_pnts":idx_pre_vali.detach().cpu().numpy()},json_file)
        json.dump({"scale":s.item()},json_file)
        
        # return s.item(),t.item()
        
    # with open(os.path.join(args.source_path,"sparse","colmap_to_depth_scale.json"), 'r') as json_file:
    #     colmap_scale = json.load(json_file)
    #     colmap_to_depth_scale= colmap_scale["scale"]