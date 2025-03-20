import numpy as np
import sys   
# sys.setrecursionlimit(100000) #例如这里设置为十万  ??
import numpy as np
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state,dict_to_tensor
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussian_renderer import GaussianModel
from scene import GaussianModelTypes
import imageio
import numpy as np
from PIL import Image
import cv2
import torch.nn.functional as F
from glob import glob
from scene import Scene
from scene.iso_gaussian_model import SeperateRepreIsotropicGaussianModel,SeperateRepreGaussianModel
from plyfile import PlyData, PlyElement
import pickle
from utils.graphics_utils import get_intrinsic_matrix,unproject_from_depthmap_torch,project_from_rgbpcd

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return res


def show_img(img):
    return Image.fromarray((img*255).astype(np.uint8))
def minmax_norm(img):
    norm = (img-img.min())/(img.max()-img.min())
    return norm
def get_pxl_grid(width,height):
    xx = range(0, width)
    yy = range(0, height)  # , self.resized_h)
    xv, yv = np.meshgrid(xx, yy)
    p_ref = np.int32(np.stack((xv, yv), axis=-1))
    return p_ref

def unproject_from_depthmap(c2w,intrinsic,depth:np.array,depth_mask=None):
    """depth: (h,w)"""
    """这个函数不对depth 为0的区域做mask，"""
    (h,w)=depth.shape
    px, py = np.meshgrid(
        
        np.arange(0, w, dtype=np.float32),np.arange(0, h, dtype=np.float32),)
    # print(px.shape,px.max())
    # print(py.shape,py.max())
    img_xy = np.stack([px+0.5, py+0.5], axis=-1)
    # print(px)
    # print(px+0.5)
    reverse_intrin = np.linalg.inv(intrinsic).T
    cam_xy =  img_xy * depth[...,None]
    cam_xyz = np.concatenate([cam_xy, depth[...,None]], axis=-1)
    cam_xyz = np.matmul(cam_xyz, reverse_intrin)
    mask_depth= cam_xyz[...,2]>1e-7
    # cam_xyz = cam_xyz[mask_depth > 1e-7,:]
    cam_xyz = np.concatenate([cam_xyz, np.ones_like(cam_xyz[...,:1])], axis=-1)
    world_xyz = np.matmul(cam_xyz.reshape(-1,4), c2w.T)[...,:3]
    return world_xyz,cam_xyz,img_xy,mask_depth




            
def save_point_cloud_to_ply(points, colors, path):
    assert points.shape[1] == 6
    assert colors.shape[1] == 3
    assert points.shape[0] == colors.shape[0]
    if not os.path.exists(os.path.dirname(path)):
        os.path.makedirs(os.path.dirname(path))

    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    elements = np.empty(points.shape[0], dtype=dtype_full)

    attributes = np.concatenate((points, colors), axis=1)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)           
from tqdm import tqdm

class Flow3D_Extraction:
    """Torch Version of Flow3D_Extraction
    """
    def __init__(self,image_shape,intrinsic,train_cams,device = "cuda",rgbpcd_mod=True,threshold_len=None,expname="pointTrack",save_dir="/data/qingmingliu/Workspace/PointTrack",exhaustive_RAFT_mode=False,debug =False):
        self.expname = expname
        self.debug = debug
        print("Device==",device)
        if device is None or device == "none":
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if rgbpcd_mod:
            self.time_pcd = torch.full((1,0,6),torch.nan) ## N*T*6 , N the number of points, T the number of frames
        else:
            self.time_pcd = torch.full((1,0,3),torch.nan)
            raise NotImplementedError
        if save_dir is None:
            self.save_dir = "/data/qingmingliu/Workspace/PointTrack"
        self.save_dir = os.path.join(save_dir,str(self.expname))
        print("saving dir:",self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir,exist_ok=True)
        self.train_cam_origin = train_cams
        self.img_shape = image_shape ## H,W
        self.intrinsic = torch.Tensor(intrinsic)
        height,width = image_shape
        self.height=height
        self.width = width
        self.mapping_table = torch.zeros([height,width])
        for y in range(height):
            for x in range(width):
                self.mapping_table[y][x]=y*width+x
        self.mapping_table.to(torch.int32).to(self.device)     
        self.template_grid = torch.tensor(get_pxl_grid(width,height)).to(self.device)   
        print("[1]preprocessing the training image dataset:")
        if exhaustive_RAFT_mode:
            print("Exhaustive RAFT Mode:")
            self.preprocess_train_cams_exhaustiveRAFT(train_cams)
        else:
            print("Normal Trajectory Paring Mode:")
            self.preprocess_train_cams(train_cams)
        if threshold_len is None:
            self.threshold_len = len(self.train_cams)/8
        else:
            self.threshold_len = threshold_len
        
        print("[1]preprocessing the training image dataset:Done")
        
    def preprocess_train_cams(self,train_cams):
        
        self.train_cams= []
        for fram_id,cam in tqdm(enumerate(train_cams)):
            depth = cam.depth
            w2c = cam.world_view_transform
            c2w = cam.view_world_transform
            print("image_name:",cam.image_name)
            # world_xyz,cam_xyz,img_xy,mask_depth= unproject_from_depthmap(c2w.T.numpy(),intrinsic,depth.cpu().numpy())
            world_xyz,cam_xyz,img_xy,mask_depth= unproject_from_depthmap_torch(c2w.T.to(depth.device),self.intrinsic.to(depth),depth)
            world_xyz = world_xyz.reshape(mask_depth.shape+(3,))
            rgb_img = cam.original_image.permute(1,2,0)
            imgname =cam.image_name
            flow_dict={}
            flow_dict["fwd_flow"]=None
            flow_dict["fwd_flow_mask"]=None
            flow_dict["bwd_flow"]=None
            flow_dict["bwd_flow_mask"]=None
            if cam.kwargs["fwd_flow"] is not None:
                flow_dict["fwd_flow"]=cam.kwargs["fwd_flow"].to(self.device)
                flow_dict["fwd_flow_mask"]=cam.kwargs["fwd_flow_mask"].to(self.device)
            if cam.kwargs["bwd_flow"] is not None:
                flow_dict["bwd_flow"]=cam.kwargs["bwd_flow"].to(self.device)
                flow_dict["bwd_flow_mask"]=cam.kwargs["bwd_flow_mask"].to(self.device)
                
            item = (world_xyz.to(self.device),mask_depth.to(self.device),rgb_img.to(self.device),flow_dict,imgname)
            self.train_cams.append(item)
            
            if self.debug and fram_id>30:
                break
    def preprocess_train_cams_exhaustiveRAFT(self,train_cams):
        
        self.train_cams= {}
        for fram_id,cam in tqdm(enumerate(train_cams)):
            depth = cam.depth
            w2c = cam.world_view_transform
            c2w = cam.view_world_transform
            print("image_name:",cam.image_name)
            # world_xyz,cam_xyz,img_xy,mask_depth= unproject_from_depthmap(c2w.T.numpy(),intrinsic,depth.cpu().numpy())
            world_xyz,cam_xyz,img_xy,mask_depth= unproject_from_depthmap_torch(c2w.T.to(depth.device),self.intrinsic.to(depth),depth)
            world_xyz = world_xyz.reshape(mask_depth.shape+(3,))
            rgb_img = cam.original_image.permute(1,2,0)
            imgname =cam.image_name
            dict_others = dict_to_tensor(cam.kwargs["dict_other"],self.device)
            
                
            item = (world_xyz.to(self.device),mask_depth.to(self.device),rgb_img.to(self.device),dict_others,imgname)
            self.train_cams[imgname]=item   
            
            if self.debug and fram_id>30:
                break
    def visualize_visited_statetable(self):
        for idx,frame in tqdm(enumerate(self.train_cams)):
            valid_mask = self.states_table[idx,:,:]
            world_xyz,depth_mask,rgb_img,flow_dict,imgname = frame
            if not os.path.exists(os.path.join(self.save_dir,"masks")):
                os.makedirs(os.path.join(self.save_dir,"masks"))
            
            Image.fromarray((valid_mask *255).cpu().numpy().astype(np.uint8)).save(os.path.join(self.save_dir,"masks",f"visited_mask_{imgname}.jpg"))
            Image.fromarray((depth_mask *255).cpu().numpy().astype(np.uint8)).save(os.path.join(self.save_dir,"masks",f"depth_mask_{imgname}.jpg"))
    def visualize_flows(self):
        for idx,frame in tqdm(enumerate(self.train_cams)):
            valid_mask = self.states_table[idx,:,:]
            world_xyz,depth_mask,rgb_img,flow_dict,imgname = frame
            if not os.path.exists(os.path.join(self.save_dir,"flows")):
                os.makedirs(os.path.join(self.save_dir,"flow"))
            
            Image.fromarray((valid_mask *255).cpu().numpy().astype(np.uint8)).save(os.path.join(self.save_dir,"masks",f"visited_mask_{imgname}.jpg"))
            Image.fromarray((depth_mask *255).cpu().numpy().astype(np.uint8)).save(os.path.join(self.save_dir,"masks",f"depth_mask_{imgname}.jpg"))
            
    def extablish_points_trajectories(self,):
        """extablish the points trajectories, according to Discussion with LIU YUAN and Wang Peng """
        unvisited_coords=None
        print("[2] find  the point cloud trajectories:")
        T= len(self.train_cams)
        
        for frame_id, cam in tqdm(enumerate(self.train_cams)):
            world_xyz,depth_mask,rgb_img,flow_dict,imgname = cam
            
            if frame_id==0:
                world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[frame_id]
                trajectories= torch.full([depth_mask.sum(),1,9],torch.nan).to(self.device) ## xyz(world coords)+rgb (color)+ (img coords)+ time frame T
                trajectories[:,0,:3]=world_xyz[depth_mask]
                trajectories[:,0,3:6]=rgb_img[depth_mask]
                trajectories[:,0,6:8]=self.template_grid[depth_mask].to(trajectories) ## 先(w dimension,h dimension)
                trajectories[:,0,-1:]=frame_id
                latest_frame_pixel_mask = torch.ones([trajectories.shape[0],]).to(self.device).to(torch.bool) ## 表示
                 ## 
                # N,_=coords.shape ## N,2

                # ## construct 3d Points.
                # next_coords=flow_dict["fwd_flow"].round().to(torch.int32)+self.template_grid ## TODO：取整还是不取整
                # coords_maskY = torch.logical_and(next_coords[:,0]>=0,next_coords[:,0]<self.height)
                # coords_maskX = torch.logical_and(next_coords[:,1]>=0,next_coords[:,1]<self.width)
                # coords_mask= torch.logical_and(coords_maskY,coords_maskX)
                # coords_mask = torch.logical_and(coords_mask,flow_dict["fwd_flow_mask"].unsqueeze(0)) ## 去掉超出边界的点和flow mask为0的点。
                # coords_mask = torch.logical_and(coords_mask,depth_mask.unsqueeze(0))
                # n_valid_points = coords_mask.sum()
                # trajectories= torch.full([n_valid_points,1,9],torch.nan).to(self.device)
                # trajectories[:,0,:3]=world_xyz[coords_mask]
                # trajectories[:,0,3:6]=rgb_img[coords_mask]
                # trajectories[:,0,6:8]=rgb_img[coords_mask]
                # trajectories[:,0,-1:]=0
                
                pass
                
            if frame_id>=1 :#and False:
                if unvisited_coords is not None and unvisited_coords.shape[0]>0:
                    world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[frame_id]
                    trajectory_i= torch.full([unvisited_coords.shape[0],1,9],torch.nan).to(self.device) ## xyz(world coords)+rgb (color)+ (img coords)+ time frame T
                    trajectory_i[:,0,:3]=world_xyz[unvisited_coords[:,1],unvisited_coords[:,0]]
                    trajectory_i[:,0,3:6]=rgb_img[unvisited_coords[:,1],unvisited_coords[:,0]]
                    trajectory_i[:,0,6:8]=unvisited_coords[:,:]
                    trajectory_i[:,0,-1:]=frame_id
                    
                    trajectory,mask =self.backward_warp(frame_id,unvisited_coords)
                    if trajectory is not None:
                        ## merge all the trajectories
                        trajectorie_ = torch.cat([trajectory,trajectory_i],1)
                        trajectories = torch.cat([trajectories,trajectorie_],0)
                        new_lastest_frame_pixel_mask = torch.ones([trajectorie_.shape[0],]).to(self.device).to(torch.bool)
                        latest_frame_pixel_mask =torch.cat([latest_frame_pixel_mask,new_lastest_frame_pixel_mask ],0)     
                        
            trajectory,coords_mask,full_next_coords_depth_mask,unvisited_coords =self.forward_warp(frame_id=frame_id)
            if trajectory is not None:
                ## merge all the trajectories
                # trajectories = torch.cat([trajectories,trajectory],1)
                # coords_lastframe = torch.tensor(trajectories[latest_frame_pixel_mask,-1,6:8],dtype=torch.long)
                coords_lastframe = trajectories[latest_frame_pixel_mask,-1,6:8].to(torch.long)
                coords_1d = self.mapping_table[coords_lastframe[:,1],coords_lastframe[:,0]] ### 从2d坐标转换到1d坐标
                coords_1d = coords_1d.to(torch.long)
                
                empty_trajectory = torch.full([trajectories.shape[0],1,9],torch.nan).to(self.device)
                empty_trajectory[latest_frame_pixel_mask]= trajectory[coords_1d]
                trajectories = torch.cat([trajectories,empty_trajectory],1)
                
                new_latest_frame_pixel_mask = torch.zeros([trajectories.shape[0],]).to(self.device).to(torch.bool)
                new_latest_frame_pixel_mask[latest_frame_pixel_mask]=full_next_coords_depth_mask[coords_1d]
                latest_frame_pixel_mask = new_latest_frame_pixel_mask
                pass
                
                   

        self.time_pcd = trajectories   
        self.fwd_flow_matrix = torch.zeros((self.time_pcd.shape[0],T,3),device=self.device)
        fwd_flow_matrix = self.fwd_flow_matrix
        for t in tqdm(range(trajectories.shape[1]-1)): ## T-1条flow
            # for n in range(N):
            nonan_msk = torch.logical_not(torch.isnan(self.time_pcd[:,t,:]).any(1))
            pcd_t = self.time_pcd[nonan_msk,t,:]
            pcd_t1= self.time_pcd[nonan_msk,t+1,:]
            t1_nonan_msk = torch.logical_not(torch.isnan(pcd_t1).any(1))
            temp = torch.zeros((pcd_t.shape[0],3),device=self.device) 
            temp[t1_nonan_msk,:]=pcd_t1[t1_nonan_msk,:3]-pcd_t[t1_nonan_msk,:3]
            fwd_flow_matrix[nonan_msk,t,:]=temp
        np.save(os.path.join(self.save_dir,f"TimePcdTable_totalLength_{len(self.train_cams)}.npy"),self.time_pcd.cpu().numpy())    
        np.save(os.path.join(self.save_dir,f"TimePcdFlowMat_totalLength_{len(self.train_cams)}.npy"),self.fwd_flow_matrix.cpu().numpy())    
        
        print("[2] find  the point cloud trajectories:Done")
             
                # pass
    def forward_warp(self,frame_id):
        """Using fwd flow to warp the Generated point clouds to the next frame
        args:
            frame_id: int, the current frame id.
        return:
            N: int, the number of valid points( according to the depth mask valid pixel number)
            trajectory: (N,1,9) 3D points + rgb + 2d index of image coords()+time frame T
            coords_mask: (N,1) mask of the valid points(表示没有超出边界的点，以及fwd flow mask为1的点)
            full_next_coords_depth_mask: (N,1) mask of the valid points (表示warp 到下一帧的坐标点是有深度值的mask。)
            unvisited_coords: (N,2) 2D coords of the unvisited points which has valid depth value.
        """
        world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[frame_id]
        
        if flow_dict["fwd_flow"] is None or frame_id+1>=len(self.train_cams) :
            return None,None,None,None
        ## 
        # N,_=coords.shape ## N,2
        print(imgname)
        trajectory= torch.full([self.width*self.height,1,9],torch.nan).to(self.device) ## xyz(world coords)+rgb (color)+ (img coords)+ time frame T
        next_coords=flow_dict["fwd_flow"].round().to(torch.long)+self.template_grid ## TODO：取整还是不取整
        next_coords = next_coords[depth_mask] ## 现在直接去掉depth mask为0的点。
        
        coords_maskX = torch.logical_and(next_coords[:,0]>=0,next_coords[:,0]<self.width) ## (先w，后h方向)
        coords_maskY = torch.logical_and(next_coords[:,1]>=0,next_coords[:,1]<self.height)
        coords_mask= torch.logical_and(coords_maskY,coords_maskX)
        coords_mask = torch.logical_and(coords_mask,flow_dict["fwd_flow_mask"][depth_mask]) ## 去掉超出边界的点和flow mask为0的点。
        
        # coords_mask = torch.logical_and(coords_mask,depth_mask.unsqueeze(0))
        # coords_mask = coords_mask[depth_mask]
        n_valid_points = coords_mask.sum()
        # trajectories= torch.full([n_valid_points,1,9],torch.nan).to(self.device)
        coords=next_coords[coords_mask]
        
        # sampled_flow = flow_dict["fwd_flow"].permute(2,0,1).unsqueeze(0)[:,:,coords[:,0],coords[:,1]] ## (1,2,H,W)
        # sampled_flow_mask = flow_dict["fwd_flow_mask"][coords[:,0],coords[:,1]]
        # sampled_depth_mask = depth_mask[coords[:,0],coords[:,1]]
        
        world_xyz_next,depth_mask_next,rgb_img_next,flow_dict_next,imgname_next=self.train_cams[frame_id+1]
        sampled_world_xyz = world_xyz_next[coords[:,1],coords[:,0],:] ##world_xyz_next: (3,H,W),
        sampled_rgb = rgb_img_next[coords[:,1],coords[:,0]]
        sampled_depth_mask = depth_mask_next[coords[:,1],coords[:,0]]
        
        unvisited_mask = depth_mask_next.clone() ## 如果下一帧的点有深度值，但是没有被fwd warp覆盖到，那么就是unvisited的点。
        unvisited_mask[coords[:,1],coords[:,0]]=False
        unvisited_coords = torch.stack(torch.where(unvisited_mask)[::-1],1) ## ## unvisited_coords(N_unvisited,2)  (坐标，先w，后h方向)

        full_next_coords_depth_mask  = torch.zeros([trajectory.shape[0],]).to(self.device).to(torch.bool)
        temp= torch.zeros([depth_mask.sum(),1]).to(self.device).to(torch.bool) ### 为什么要用一个中间值，因为发现连续索引赋值会出现问题。
        temp[coords_mask,0]=sampled_depth_mask
        full_next_coords_depth_mask[depth_mask.reshape(-1)]=temp[:,0]
        # full_next_coords_depth_mask[depth_mask.reshape(-1)][coords_mask,0]=sampled_depth_mask
        
        
        
        trajectory_temp= torch.full([depth_mask.sum(),1,9],torch.nan).to(self.device)
        trajectory_temp[coords_mask,0,:3]=sampled_world_xyz
        trajectory_temp[coords_mask,0,3:6]=sampled_rgb
        trajectory_temp[coords_mask,0,6:8]=coords.to(torch.float32)
        trajectory_temp[coords_mask,0,-1:]=frame_id+1
        trajectory[depth_mask.reshape(-1)]=trajectory_temp
        
        # trajectory[coords_mask,0,:3]=sampled_world_xyz
        # trajectory[coords_mask,0,3:6]=sampled_rgb
        # trajectory[coords_mask,0,6:8]=coords
        # trajectory[coords_mask,0,-1:]=frame_id+1
        
        trajectory[torch.logical_not(full_next_coords_depth_mask)]=torch.nan
        # query_points = torch.tensor(coords).to(flow_dict["bwd_flow"].device)
        ## tensor
        
        return trajectory,coords_mask,full_next_coords_depth_mask,unvisited_coords

        
        
        
    def backward_warp(self,frame_id,coords,align_corners=True):
        """Using bwd flow to warp the Generated point clouds to the previous frame
            coords: (N,2) 2D coordinates  ## (坐标先w,后h方向)
            递归式的backward warp去补全前面的frame里面缺失的点， 在第一次调用的时候实际上coord是整形，但是在在被下一次调用的的时候，coords是float类型的，所以用grid sample。
                        
        """
        world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[frame_id]
        
        if flow_dict["bwd_flow"] is None:
            return None,None
        ## 
        N,_=coords.shape
        trajectory= torch.full([N,1,9],torch.nan).to(self.device) ## xyz(world coords)+rgb (color)+ (img coords)+ time frame T
        # flow_dict=self.train_cams[frame_id][-2]
        world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[frame_id]
        query_points = torch.tensor(coords).to(flow_dict["bwd_flow"].device)
        normalize_query_points = 2*query_points/torch.tensor([self.width-1,self.height-1],device=self.device)-1.0
        
        ## tensor
    
        # normalize_query_points = normalize_query_points[:,[1,0]]## 逆天，这经过测试这个好像不需要。
        ### 需要把normalize_query_points的最后一个维度换一下顺序。从先w方向坐标后h方向坐标换成  先h，后w
        sampled_flow = F.grid_sample(flow_dict["bwd_flow"].permute(2,0,1).unsqueeze(0),normalize_query_points.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        sampled_flow_mask = F.grid_sample(flow_dict["bwd_flow_mask"].unsqueeze(0).unsqueeze(0).float(),normalize_query_points.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        sampled_flow = sampled_flow.permute(0,2,3,1).squeeze(0).squeeze(0)
        valid_mask = sampled_flow_mask.squeeze(0).squeeze(0).squeeze(0)>1.0-1e-5 ## 这里 1.0-1e-7 是为了防止采样点落在边界上。
        sampled_flow = sampled_flow#[valid_mask]
        query_points= query_points#[valid_mask]
        
        
        ## 这里需要考虑一下，如遇coords_previous_img超出边界的情况。
        coords_previous_img=sampled_flow+query_points## 这里coords 又是先w，后h的，
        coords_maskX = torch.logical_and(coords_previous_img[:,0]>=0,coords_previous_img[:,0]<self.width) ## (先w，后h方向)
        coords_maskY = torch.logical_and(coords_previous_img[:,1]>=0,coords_previous_img[:,1]<self.height)
        coords_mask= torch.logical_and(coords_maskY,coords_maskX)
        valid_mask= torch.logical_and(coords_mask,valid_mask)
        coords_previous_img=coords_previous_img[valid_mask]
        
        previous_frame_data = self.train_cams[frame_id-1]
        world_xyz_pre,depth_mask_pre,rgb_img_pre,flow_dict_pre,_=previous_frame_data
        # normalize_query_points_pre = 2*coords_previous_img/torch.tensor([self.height-1,self.width-1],device=self.device)-1.0
        normalize_query_points_pre = 2*coords_previous_img/torch.tensor([self.width-1,self.height-1],device=self.device)-1.0
        # normalize_query_points_pre = normalize_query_points_pre[:,[1,0]] ### 需要把normalize_query_points的最后一个维度换一下顺序。从先w方向坐标后h方向坐标换成  先h，后w
        ## TODO:考虑depthmask，如果depthmask为0，那么就不需要进行采样。
        ## 以及 bwd flow mask，如果mask为0，那么就不需要进行采样。
        
        
        sampled_world_xyz = F.grid_sample(world_xyz_pre.permute(2,0,1).unsqueeze(0),normalize_query_points_pre.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        sampled_depth_mask = F.grid_sample(depth_mask_pre.unsqueeze(0).unsqueeze(0).float(),normalize_query_points_pre.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        sampled_rgb = F.grid_sample(rgb_img_pre.permute(2,0,1).unsqueeze(0).float(),normalize_query_points_pre.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        ### filter invalid points
        # valid_mask_depth = sampled_depth_mask.squeeze()>1.0-1e-5  ## 这里还不能用 squeeze（），如果只有一个值的话，会全部出错 ## 因为这个mask是双线性插值得到的，所以判断小于1的话，是不在mask区域。
        valid_mask_depth = sampled_depth_mask.squeeze(0).squeeze(0).squeeze(0)>1.0-1e-5 ## 因为这个mask是双线性插值得到的，所以判断小于1的话，是不在mask区域。
        selected_xyz = sampled_world_xyz.permute(0,2,3,1).squeeze(0).squeeze(0)[valid_mask_depth]
        selected_coords = coords_previous_img[valid_mask_depth]
        selected_rgb = sampled_rgb.permute(0,2,3,1).squeeze(0).squeeze(0)[valid_mask_depth]
        
        
        
        ### add valid points to tranjectory ### 连续索引赋值会出现问题，所以用一个中间值。
        temp_trajectory = torch.full([torch.sum(valid_mask),1,9],torch.nan).to(self.device)
        temp_trajectory[valid_mask_depth,0,:3]=selected_xyz
        temp_trajectory[valid_mask_depth,0,3:6]=selected_rgb
        temp_trajectory[valid_mask_depth,0,6:8]=selected_coords ## (先w，后h方向)
        temp_trajectory[valid_mask_depth,0,-1:]=frame_id-1
        
        trajectory[valid_mask]=temp_trajectory
        # trajectory[valid_mask][valid_mask_depth,0,:3]=selected_xyz
        # trajectory[valid_mask][valid_mask_depth,0,3:6]=selected_rgb
        # trajectory[valid_mask][valid_mask_depth,0,6:]=selected_coords
        # trajectory[valid_mask][valid_mask_depth,0,-1:]=frame_id
        mask = torch.zeros_like(valid_mask).to(valid_mask)
        mask_temp =torch.zeros([valid_mask.sum(),]).to(valid_mask)
        mask_temp[valid_mask_depth]=True
        mask[valid_mask]=mask_temp
        ### add valid Points to trajectory
    
        
        if frame_id-1>=0:
            ##recursion backward warp
            trajectory_pre,mask_pre = self.backward_warp(frame_id-1,selected_coords )
            # backward_warp = torch.cat([sampled_world_xyz,rgb_img[coords[0],coords[1]].unsqueeze(0)],-1)
            if trajectory_pre is not None:
                f_num= trajectory_pre.shape[1]
                trajectory_previous= torch.full([N,f_num,9],torch.nan).to(self.device)
                
                
                trajectory_previous[mask]=trajectory_pre ## TODO:check 一下会不会有和numpy一样的行为（多重索引无法赋值)
                trajectory =torch.cat([trajectory_previous,trajectory],1)   
        
            
        return trajectory,mask
            

        pass       
    def backward_warp1(self,frame_id,coords,align_corners=True):
        """Using bwd flow to warp the Generated point clouds to the previous frame
            coords: (N,2) 2D coordinates  ## (坐标先w,后h方向)
            递归式的backward warp去补全前面的frame里面缺失的点， 在第一次调用的时候实际上coord是整形，但是在在被下一次调用的的时候，coords是float类型的，所以用grid sample。
                        
        """
        world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[frame_id]
        
        if flow_dict["bwd_flow"] is None:
            return None,None
        ## 
        N,_=coords.shape
        trajectory= torch.full([N,1,9],torch.nan) ## xyz(world coords)+rgb (color)+ (img coords)+ time frame T
        # flow_dict=self.train_cams[frame_id][-2]
        world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[frame_id]
        query_points = torch.tensor(coords).to(flow_dict["bwd_flow"].device)
        normalize_query_points = 2*query_points/torch.tensor([self.width-1,self.height-1],device=self.device)-1.0
        ## tensor
    
        normalize_query_points = normalize_query_points[:,[1,0]]
        ### 需要把normalize_query_points的最后一个维度换一下顺序。从先w方向坐标后h方向坐标换成  先h，后w
        sampled_flow = F.grid_sample(flow_dict["bwd_flow"].permute(2,0,1).unsqueeze(0),normalize_query_points.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        sampled_flow_mask = F.grid_sample(flow_dict["bwd_flow_mask"].unsqueeze(0).unsqueeze(0).float(),normalize_query_points.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        sampled_flow = sampled_flow.permute(0,2,3,1).squeeze(0).squeeze(0)
        valid_mask = sampled_flow_mask.squeeze(0).squeeze(0).squeeze(0)>1.0-1e-5 ## 这里 1.0-1e-7 是为了防止采样点落在边界上。
        sampled_flow = sampled_flow[valid_mask]
        query_points= query_points[valid_mask]
        
        
        
        coords_previous_img=sampled_flow+query_points## 这里coords 又是先w，后h的，
        previous_frame_data = self.train_cams[frame_id-1]
        world_xyz_pre,depth_mask_pre,rgb_img_pre,flow_dict_pre,_=previous_frame_data
        normalize_query_points_pre = 2*coords_previous_img/torch.tensor([self.height-1,self.width-1],device=self.device)-1.0
        normalize_query_points_pre = normalize_query_points_pre[:,[1,0]] ### 需要把normalize_query_points的最后一个维度换一下顺序。从先w方向坐标后h方向坐标换成  先h，后w
        ## TODO:考虑depthmask，如果depthmask为0，那么就不需要进行采样。
        ## 以及 bwd flow mask，如果mask为0，那么就不需要进行采样。
        
        
        sampled_world_xyz = F.grid_sample(world_xyz_pre.permute(2,0,1).unsqueeze(0),normalize_query_points_pre.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        sampled_depth_mask = F.grid_sample(depth_mask_pre.unsqueeze(0).unsqueeze(0).float(),normalize_query_points_pre.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        sampled_rgb = F.grid_sample(rgb_img_pre.permute(2,0,1).unsqueeze(0).float(),normalize_query_points_pre.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
        ### filter invalid points
        valid_mask_depth = sampled_depth_mask.squeeze()>1.0-1e-5 ## 因为这个mask是双线性插值得到的，所以判断小于1的话，是不在mask区域。
        selected_xyz = sampled_world_xyz.permute(0,2,3,1).squeeze()[valid_mask_depth]
        selected_coords = coords_previous_img[valid_mask_depth]
        selected_rgb = sampled_rgb.permute(0,2,3,1).squeeze()[valid_mask_depth]
        
        
        
        ### add valid points to tranjectory
        temp_trajectory = torch.full([torch.sum(valid_mask),1,9],torch.nan).to(self.device)
        temp_trajectory[valid_mask_depth,0,:3]=selected_xyz
        temp_trajectory[valid_mask_depth,0,3:6]=selected_rgb
        temp_trajectory[valid_mask_depth,0,6:8]=selected_coords
        temp_trajectory[valid_mask_depth,0,-1:]=frame_id
        
        trajectory[valid_mask]=temp_trajectory
        # trajectory[valid_mask][valid_mask_depth,0,:3]=selected_xyz
        # trajectory[valid_mask][valid_mask_depth,0,3:6]=selected_rgb
        # trajectory[valid_mask][valid_mask_depth,0,6:]=selected_coords
        # trajectory[valid_mask][valid_mask_depth,0,-1:]=frame_id
        mask = torch.zeros_like(valid_mask).to(valid_mask)
        mask[valid_mask][valid_mask_depth]=True
        ### add valid Points to trajectory
    
        
        if frame_id-1>=0:
            ##recursion backward warp
            trajectory_pre,mask_pre = self.backward_warp(self,frame_id-1,selected_coords )
            # backward_warp = torch.cat([sampled_world_xyz,rgb_img[coords[0],coords[1]].unsqueeze(0)],-1)
            if trajectory_pre is not None:
                f_num= trajectory_pre.shape[1]
                trajectory_previous= torch.full([N,f_num,9],torch.nan)
                
                
                trajectory_previous[mask][mask_pre ]=trajectory_pre[mask_pre] ## TODO:check 一下会不会有和numpy一样的行为（多重索引无法赋值)
                trajectory =torch.cat([trajectory_previous,trajectory],1)   
        
            
        return trajectory,mask
        
        
    
    
        pass
    
    def check_unvisited_validpnts(self,):
        total = 0
        valid_depth_total = 0
        for idx,frame in tqdm(enumerate(self.train_cams)):
            world_xyz,depth_mask,rgb_img,flow_dict,imgname = frame
            not_visit_validpxl = torch.logical_and(torch.logical_not(self.states_table[idx,:,:].to(depth_mask.device)),depth_mask)
            total+=torch.sum(not_visit_validpxl)
            valid_depth_total+=torch.sum(depth_mask)
        self.number_unvisited_validpnts = total
        print("===============number_unvisited_validpnts:",total)
        with open(os.path.join(self.save_dir,"masks","number_unvisited_validpnts.txt"),"w") as f:
            f.write("number_unvisite d_validDepth_pnts:"+str(total)+"\n")
            f.write("totoal_pixel_number:"+str(self.height*self.width*len(self.train_cams))+"\n")
            f.write("totoal_valid_depth_number:"+str(valid_depth_total)+"\n")
            # f.write(str(total))
    # pass
    def recursion_head(self,):
        points_trajectories=[]
        print("[2] find  the point cloud trajectories:")
        self.states_table = torch.zeros([len(self.train_cams),self.height,self.width],dtype=torch.bool,device=self.device)
        for i in tqdm(range(0,len(self.train_cams))):
            
            world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[i]
            # self.add_points(depth_map,rgb_img,flow_dict)
            for idy in range(self.height):
                for idx in range(self.width):
                    if not depth_mask[idy,idx]:
                        
                        continue
                    if  i==0: ## 如果这个点有深度，并且是第一帧，那么就开始递归。
                        trajectory=[]
                        cur={"frame_id":i,"coord_uv":(idy,idx),"coord_xyz":world_xyz[idy,idx],"rgb":rgb_img[idy,idx],"start":True}
                        trajectory.append(cur)
                        self.states_table[i,idy,idx]=True ## 标记当前点已经访问过了。
                        if flow_dict["fwd_flow_mask"] is not None and  flow_dict["fwd_flow_mask"][idy,idx] : ## 如果flow mask 为1，说明这个点在下一帧中可能有对应的点。反之则消失。
                            # next_idy,nex_idx = (idy,idx) +flow_dict["fwd_flow"][idy,idx].round().astype(np.int32)
                            next_idx,next_idy = torch.tensor((idx,idy)) +flow_dict["fwd_flow"][idy,idx].round().to(torch.int32)
                            ## flow 是先w，后h
                            
                            self.recursion_step(i+1,(next_idy,next_idx),trajectory)
                        points_trajectories.append(trajectory)
                    elif i>0 and i%5==0 and  not self.states_table[i,idy,idx] : 
                    
                    # elif i>0  and flow_dict["bwd_flow_mask"] is not None and not(flow_dict["bwd_flow_mask"][idy,idx]): 
                        ## 如果这个点有深度，并且不是是第一帧，并且没有被访问过，并且这个点的bwd_flow_mask为False(说明这个点会产生轨迹)，那么就开始递归。
                        
                        cur={"frame_id":i,"coord_uv":(idy,idx),"coord_xyz":world_xyz[idy,idx],"rgb":rgb_img[idy,idx],"start":True}
                        self.states_table[i,idy,idx]=True
                        trajectory=[]
                        trajectory.append(cur)
                        if flow_dict["fwd_flow_mask"] is not None and flow_dict["fwd_flow_mask"][idy,idx] : ## 如果flow mask 为1，说明这个点在下一帧中可能有对应的点。反之则消失。
                            # next_idy,nex_idx = (idy,idx) +flow_dict["fwd_flow"][idy,idx].round().astype(np.int32)
                            next_idx,next_idy = torch.tensor((idx,idy)) +flow_dict["fwd_flow"][idy,idx].round().to(torch.int32)
                            self.recursion_step(i+1,(next_idy,next_idx),trajectory)
                        points_trajectories.append(trajectory)
        
        self.points_trajectories =points_trajectories
        # for item  in points_trajectories:
        #     print(len())
            
    def recursion_step(self,frame_id,coords,trajectory):
        
        idy,idx=coords ## 二维图像的坐标。
        if idx<0 or idx>=self.width or idy<0 or idy>=self.height or frame_id>=len(self.train_cams) or self.states_table[frame_id,idy,idx]:## TODO： 这里需要考虑一下，判断是否访问过是不是太强了
            return
        world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[frame_id]
        # if not depth_mask[idy,idx] or (flow_dict["bwd_flow_mask"] is not None and not(flow_dict["bwd_flow_mask"][idy,idx])) :
                    ## 如果这个坐标depth没有值没有点  or  这个坐标的bwdmask是0(说明这里将会产生点，这种情况会单独处理)  or 这个点已经被访问过 直接返回。
       
        self.states_table[frame_id,idy,idx]=True
        if not depth_mask[idy,idx]:

            
            return 
        else:
            cur={"frame_id":frame_id,"coord_uv":(idy,idx),"coord_xyz":world_xyz[idy,idx],"rgb":rgb_img[idy,idx]}
            trajectory.append(cur)
            if flow_dict["fwd_flow_mask"] is not None and flow_dict["fwd_flow_mask"][idy,idx] and frame_id+1<len(self.train_cams) :## 如果flow mask 为1，说明这个点在下一帧中可能有对应的点。反之则消失。 判断是否是最后一帧。
                # next_idy,nex_idx = (idy,idx) +flow_dict["fwd_flow"][idy,idx].round().astype(np.int32)
                next_idx,next_idy = torch.tensor((idx,idy)) +flow_dict["fwd_flow"][idy,idx].round().to(torch.int32)
                
                self.recursion_step(frame_id+1,(next_idy,next_idx),trajectory)
        
        
    def align_matrix(self,):
        print("[2]align the time_pcd matrix:")
        N=len(self.points_trajectories)
        T=len(self.train_cams)
        
        start_end=[]
        time_pcd = []
        # time_pcd = torch.full((N,T,6),torch.nan)
        self.points_trajectories.sort(key=len,reverse=True)
        # sorted(self.points_trajectories,key=len,reverse=True)
        
        for i,trajectory in tqdm(enumerate(self.points_trajectories)):
            if len(trajectory)<self.threshold_len:
                break
            time_pcd_i = torch.full((1,T,6),torch.nan)
            start_end_i=torch.full((1,2),0).to(torch.int32)
            for j,point in enumerate(trajectory):
                if j==0:
                    start_end_i[0,0]=point["frame_id"]
                if j==len(trajectory)-1:
                    start_end_i[0,1]=point["frame_id"]
                t =point["frame_id"]
                time_pcd_i[0,t,:3]=point["coord_xyz"]
                time_pcd_i[0,t,3:]=point["rgb"]
            time_pcd.append(time_pcd_i)
            start_end.append(start_end_i)
        start_end = torch.cat(start_end,0)
        time_pcd = torch.cat(time_pcd,0)
        self.start_end = start_end
        self.time_pcd=time_pcd
        self.fwd_flow_matrix = torch.zeros((time_pcd.shape[0],T,3))
        fwd_flow_matrix = self.fwd_flow_matrix
        for t in tqdm(range(T-1)):
            # for n in range(N):
            nonan_msk = torch.logical_not(torch.isnan(time_pcd[:,t,:]).any(1))
            pcd_t = time_pcd[nonan_msk,t,:]
            pcd_t1= time_pcd[nonan_msk,t+1,:]
            t1_nonan_msk = torch.logical_not(torch.isnan(pcd_t1).any(1))
            temp = torch.zeros((pcd_t.shape[0],3)) 
            temp[t1_nonan_msk,:]=pcd_t1[t1_nonan_msk,:3]-pcd_t[t1_nonan_msk,:3]
            fwd_flow_matrix[nonan_msk,t,:]=temp

        
        # if not os.path.exists(os.path.join(self.save_dir,str(self.expname))):
        #     makedirs(os.path.join(self.save_dir,str(self.expname)))
        np.save(os.path.join(self.save_dir,f"time_pcd_SE_timelen{len(self.train_cams)}.npy"),start_end.cpu().numpy())    
        np.save(os.path.join(self.save_dir,f"time_pcd_timelen{len(self.train_cams)}.npy"),time_pcd.cpu().numpy())    
        np.save(os.path.join(self.save_dir,f"time_pcd_fwdmatrix_timelen{len(self.train_cams)}.npy"),fwd_flow_matrix.cpu().numpy())    
        print("[2]align the time_pcd matrix:Done")
        #153524
    def sepreately_save_trajectories(self,):
        print("[3] saving Pcd and Flow")

        time_pcd = self.time_pcd
        N,T=time_pcd.shape[0],time_pcd.shape[1]
        # T=len(self.train_cams)            
        for t in tqdm(range(T)):
            # for n in range(N):
            nonan_msk = torch.logical_not(torch.isnan(time_pcd[:,t,:]).any(1))
            pcd_t = time_pcd[nonan_msk,t,:]
            if t<T-1:
                pcd_t_flow = self.fwd_flow_matrix[nonan_msk,t,:]
            if (pcd_t[:,3:6] <1.0+1e-7).all():
                 pcd_t[:,3:6]*=255
            imgname = self.train_cams[t][-1]
            np.savetxt(os.path.join(self.save_dir,f"time_pcd_frame{imgname}.txt"),pcd_t[:,:6].cpu().numpy(),delimiter=" ")
            if t<T-1:
                np.save(os.path.join(self.save_dir,f"time_pcd_fwd_flow_frame{imgname}.npy"),pcd_t_flow.cpu().numpy())
                # np.save
        print("[3] saved Pcd and Flow:Done")
        
        pass
    
    ## MODEL VERSION 3.0
    def build_correspondence(self,):
        """model version 3.0 build exhausitve raft correspondence , and save the correspondence pairs.
        """
        print("[2] find  the point cloud trajectories:")
        T= len(self.train_cams)
        trajectories_pair_list = []
        align_corners=True
        for frame_id, cam in tqdm(enumerate(self.train_cams)):
            # world_xyz,depth_mask,rgb_img,flow_dict,imgname = cam
            world_xyz,depth_mask,rgb_img,flow_dict,imgname=self.train_cams[frame_id]
            print("sovling frame:",frame_id,"imgname:",imgname)
            trajectory= torch.full([depth_mask.sum(),7],torch.nan).to(self.device) ## xyz(world coords)+rgb (color)+ (img coords)+ time frame T
            trajectory[:,:3]=world_xyz[depth_mask]
            trajectory[:,3:6]=rgb_img[depth_mask]
            trajectory[:,-1:]=frame_id
            # latest_frame_pixel_mask = torch.ones([trajectories.shape[0],]).to(self.device).to(torch.bool) ## 表示
            # pcd_pre=None
            # pcd_next=None
            current_frame = {"frame_id":frame_id,"pcd":trajectory}
            
            if not frame_id==0:
                ## bwd 
                pcd_pre = torch.full([depth_mask.sum(),7],torch.nan).to(self.device)

                pre_coords=flow_dict["bwd_flow"][depth_mask]+self.template_grid[depth_mask] #
                valid_mask = flow_dict["bwd_flow_mask"][depth_mask]
                
                coords_maskX = torch.logical_and(pre_coords[:,0]>=0,pre_coords[:,0]<self.width) ## (先w，后h方向)
                coords_maskY = torch.logical_and(pre_coords[:,1]>=0,pre_coords[:,1]<self.height)
                coords_mask= torch.logical_and(coords_maskY,coords_maskX)
                valid_mask= torch.logical_and(coords_mask,valid_mask)
                
                query_points = pre_coords.clone().to(flow_dict["bwd_flow"].device)
                
                pre_frame_data = self.train_cams[frame_id-1]
                world_xyz_pre,depth_mask_pre,rgb_img_pre,flow_dict_pre,_=pre_frame_data
                normalize_query_points_pre = 2*query_points/torch.tensor([self.width-1,self.height-1],device=self.device)-1.0
        
                sampled_world_xyz = F.grid_sample(world_xyz_pre.permute(2,0,1).unsqueeze(0),normalize_query_points_pre.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
                sampled_depth_mask = F.grid_sample(depth_mask_pre.unsqueeze(0).unsqueeze(0).float(),normalize_query_points_pre.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
                sampled_rgb = F.grid_sample(rgb_img_pre.permute(2,0,1).unsqueeze(0).float(),normalize_query_points_pre.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
                
                valid_mask_depth = sampled_depth_mask.squeeze(0).squeeze(0).squeeze(0)>1.0-1e-5 ## 因为这个mask是双线性插值得到的，所以判断小于1的话，是不在mask区域。
                valid_mask = torch.logical_and(valid_mask,valid_mask_depth)
                
                pcd_pre[valid_mask,:3]=sampled_world_xyz.permute(0,2,3,1).squeeze(0).squeeze(0)[valid_mask]
                pcd_pre[valid_mask,3:6]=sampled_rgb.permute(0,2,3,1).squeeze(0).squeeze(0)[valid_mask]
                pcd_pre[valid_mask,-1:]=frame_id-1
                current_frame["pcd_pre"]=pcd_pre
                current_frame["pcd_pre_msk"]=valid_mask
                pass
            if not frame_id==T-1:
                
                pcd_next = torch.full([depth_mask.sum(),7],torch.nan).to(self.device)
                ## fwd

               
                next_coords=flow_dict["fwd_flow"][depth_mask]+self.template_grid[depth_mask] #
                
                valid_mask = flow_dict["fwd_flow_mask"][depth_mask]
                coords_maskX = torch.logical_and(next_coords[:,0]>=0,next_coords[:,0]<self.width) ## (先w，后h方向)
                coords_maskY = torch.logical_and(next_coords[:,1]>=0,next_coords[:,1]<self.height)
                coords_mask= torch.logical_and(coords_maskY,coords_maskX)
                valid_mask= torch.logical_and(coords_mask,valid_mask)
                
                query_points = next_coords.clone().to(flow_dict["fwd_flow"].device)
                ### get next frame data
                next_frame_data = self.train_cams[frame_id+1]
                world_xyz_next,depth_mask_next,rgb_img_next,flow_dict_next,_=next_frame_data
                normalize_query_points_next = 2*query_points/torch.tensor([self.width-1,self.height-1],device=self.device)-1.0
        
                sampled_world_xyz = F.grid_sample(world_xyz_next.permute(2,0,1).unsqueeze(0),normalize_query_points_next.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
                sampled_depth_mask = F.grid_sample(depth_mask_next.unsqueeze(0).unsqueeze(0).float(),normalize_query_points_next.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
                sampled_rgb = F.grid_sample(rgb_img_next.permute(2,0,1).unsqueeze(0).float(),normalize_query_points_next.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
                
                valid_mask_depth = sampled_depth_mask.squeeze(0).squeeze(0).squeeze(0)>1.0-1e-5 ## 因为这个mask是双线性插值得到的，所以判断小于1的话，是不在mask区域。
                valid_mask = torch.logical_and(valid_mask,valid_mask_depth)
                
                pcd_next[valid_mask,:3]=sampled_world_xyz.permute(0,2,3,1).squeeze(0).squeeze(0)[valid_mask]
                pcd_next[valid_mask,3:6]=sampled_rgb.permute(0,2,3,1).squeeze(0).squeeze(0)[valid_mask]
                pcd_next[valid_mask,-1:]=frame_id+1
                
                current_frame["pcd_next"]=pcd_next
                current_frame["pcd_next_msk"]=valid_mask
                
                pass
            
            trajectories_pair_list.append(current_frame)
        self.points_trajectories_pair_list = trajectories_pair_list
    def save_correspondence_pairs(self,):
        with open(os.path.join(self.save_dir,f"ModelVersion3_Res_Res{self.width}X{self.height}_{str(len(self.points_trajectories_pair_list))}.pkl"), 'wb') as f:
            pickle.dump(self.points_trajectories_pair_list, f)
        
        pass
    ## MODEL VERSION 4.0
    def build_exhaustive_correspondence(self,):
        # pass
        """model version 4.0 build exhausitve raft correspondence , and save the correspondence pairs.
        """
        print("[2] find  the point cloud trajectories(Model Version 4.0):")
        T= len(self.train_cams)
        trajectories_pair_list = []
        align_corners=True
        # imgname_to_index_dict =  {}
        # for frame_id, cam in tqdm(enumerate(self.train_cams)):
            
        #     imgname_to_index_dict[imgname] = frame_id ### 获得 imgname 和 frame_id 的对应关系 如 0_0000.png -> 0
        
        pbar =tqdm(self.train_cams.keys())
        for frame_name in  pbar: ### 遍历 query frame 
            # world_xyz,depth_mask,rgb_img,flow_dict,imgname = cam
            world_xyz,depth_mask,rgb_img,exhaustive_flow_dict,imgname=self.train_cams[frame_name ]
            print("sovling frame:",frame_name ,"imgname:",imgname)
            pbar.set_postfix({"sovling frame:":frame_name ,"imgname:":imgname}, refresh=True)
            assert int(frame_name) == int(imgname), "frame_name and imgname should be the same"
            frame_id = frame_name
            
            trajectory= torch.full([depth_mask.sum(),6],torch.nan).to(self.device) ## xyz(world coords)+rgb (color)+ (img coords)+ time frame T
            trajectory[:,:3]=world_xyz[depth_mask]
            trajectory[:,3:6]=rgb_img[depth_mask]
            # trajectory[:,-1:]=frame_id
            current_frame = {"frame_id":frame_id,"pcd":trajectory,"imgname":imgname,"target_dicts":{}}
            pbar_inner = tqdm(self.train_cams.keys())
            for frame_name_target in pbar_inner: ## 遍历target frame
                if frame_name==frame_name_target:
                    # print("frame_name==frame_name_target")
                    # print(frame_name,frame_name_target)
                    continue
                world_xyz_target,depth_mask_target,rgb_img_target,_,_=self.train_cams[frame_name_target]
                pbar_inner.set_postfix({"frame_name_target:":frame_name_target}, refresh=True)
                # print("frame_name_target:",frame_name_target)、
                dict_key  = f"{frame_name}_{frame_name_target}"
                flow = exhaustive_flow_dict["rafts"][dict_key]
                flow_mask = exhaustive_flow_dict["raft_msks"][dict_key]
             

                pcd_target = torch.full([depth_mask.sum(),6],torch.nan).to(self.device)
                target_coords=flow[depth_mask]+self.template_grid[depth_mask] #
                valid_mask = flow_mask[depth_mask][:,0] ##NOTE: 这里只选了channel 0
                # print("valid sum flow msk,channel0",valid_mask[:,0].sum(),"channel1",valid_mask[:,1].sum(),"channel1and2",torch.logical_and(valid_mask[:,1],valid_mask[:,0]).sum())
                # break 
                
                
                coords_maskX = torch.logical_and(target_coords[:,0]>=0,target_coords[:,0]<self.width) ## (先w，后h方向)
                coords_maskY = torch.logical_and(target_coords[:,1]>=0,target_coords[:,1]<self.height)
                coords_mask= torch.logical_and(coords_maskY,coords_maskX)
                valid_mask= torch.logical_and(coords_mask,valid_mask)
                
                query_points = target_coords.clone().to(flow.device)
                
                normalize_query_points_target = 2*query_points/torch.tensor([self.width-1,self.height-1],device=self.device)-1.0
        
                sampled_world_xyz = F.grid_sample(world_xyz_target.permute(2,0,1).unsqueeze(0),normalize_query_points_target.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
                sampled_depth_mask = F.grid_sample(depth_mask_target.unsqueeze(0).unsqueeze(0).float(),normalize_query_points_target.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
                sampled_rgb = F.grid_sample(rgb_img_target.permute(2,0,1).unsqueeze(0).float(),normalize_query_points_target.unsqueeze(0).unsqueeze(0),mode="bilinear",padding_mode="border",align_corners=align_corners)
                
                valid_mask_depth = sampled_depth_mask.squeeze(0).squeeze(0).squeeze(0)>1.0-1e-5 ## 因为这个mask是双线性插值得到的，所以判断小于1的话，是不在mask区域。
                valid_mask = torch.logical_and(valid_mask,valid_mask_depth)
                
                pcd_target[valid_mask,:3]=sampled_world_xyz.permute(0,2,3,1).squeeze(0).squeeze(0)[valid_mask]
                pcd_target[valid_mask,3:6]=sampled_rgb.permute(0,2,3,1).squeeze(0).squeeze(0)[valid_mask]
                # pcd_target[valid_mask,-1:]=frame_id-1
                target_dict = {"frame_id":frame_name_target,"pcd":pcd_target,"pcd_target_msk":valid_mask}
                current_frame["target_dicts"][dict_key]=target_dict
                
            trajectories_pair_list.append(current_frame)
        self.points_trajectories_pair_list = trajectories_pair_list
    def save_exhaustive_correspondence_pairs(self,):
        with open(os.path.join(self.save_dir,f"PointTrackModelVersion4_exhaustive_correspondence_filterdepthEdge_Res{self.width}X{self.height}_{str(len(self.points_trajectories_pair_list))}.pkl"), 'wb') as f:
            pickle.dump(self.points_trajectories_pair_list, f)
            print("saving at:",os.path.join(self.save_dir,f"PointTrackModelVersion4_exhaustive_correspondence_filterdepthEdge_Res{self.width}X{self.height}_{str(len(self.points_trajectories_pair_list))}.pkl"))
        
        pass

    def get_points(self,frame_t):
        pass
    def draw_histgram(self,):
        import matplotlib.pyplot as plt
        
        time_pcd  = self.time_pcd
        # torch.isnan(time_pcd[:,:,:]).any(-1).shape
        number_ = torch.sum(torch.logical_not(torch.isnan(time_pcd[:,:,:]).any(-1)),-1)
        plt.figure(1)
        plt.hist(number_.cpu().numpy(), bins=1000, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
        # 显示横轴标签
        plt.xlabel("trajectory length")
        plt.yscale('log')
        # 显示纵轴标签
        plt.ylabel("frequency")
        # plt.show()
        plt.savefig(os.path.join(self.save_dir,f"histgram_{len(time_pcd)}"+'.jpg'))
        plt.close()
    def update_points(self,frame_t):
        pass








if __name__=="__main__":
    import configargparse
    import os
    parser = configargparse.ArgumentParser(description="Training script parameters")## LQM
    parser.add_argument('--exhaustive_training',  action='store_true',default=False)
    
    dataset = ModelParams(parser, sentinel=True)
    dataset =parser.parse_args(sys.argv[1:])
    
    print(sys.argv[1:])
    dataset.model_path="./Jupyter_test_exported"

    exhaustive_RAFT_mode =dataset.exhaustive_training
    dataset.sh_degree=3
    dataset.eval=True
    dataset.approx_l=4
    dataset.approx_l_global=-1
    dataset.timestamp=None
    dataset.resolution=-1
    dataset.depth_folder="xxx"
    dataset.random_init_pcd=False
    dataset.data_device="cpu"
    print("Sorece-path:,",dataset.source_path)    
    print("image folder:,",dataset.images)    
    # gt_imgpaths=os.path.join(dataset.source_path,"rgb/2x/")
    # gt_imgpaths=sorted(glob(os.path.join(gt_imgpaths,"1_*.png"))+glob(os.path.join(gt_imgpaths,"2_*.png")))
    

    gaussians =SeperateRepreIsotropicGaussianModel(dataset.sh_degree, dataset.approx_l,dataset.approx_l_global)
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    train_cams= scene.getTrainCameras()
    test_cams= scene.getTestCameras()
    from utils.graphics_utils import getWorld2View2,fov2focal
    width = train_cams[0].image_width
    height = train_cams[0].image_height
    focal_lengthX = fov2focal(train_cams[0].FoVx,width)
    focal_lengthY = fov2focal(train_cams[0].FoVy,height)
    print("Focals:",focal_lengthX,focal_lengthY)
    print("Width Height:",width,height)
    
    intrinsic=get_intrinsic_matrix(width=width, height= height, focal_length=focal_lengthX )

    if not  dataset.exhaustive_training:
        ###########################################
        ############# MODEL VERSION 3.#############
        ###########################################
        pointtracker = Flow3D_Extraction((height,width),intrinsic,train_cams,rgbpcd_mod=True,exhaustive_RAFT_mode=exhaustive_RAFT_mode,
                                        debug=False,
                                        save_dir = dataset.source_path,
                                        expname="./",
                                        device = "cpu",threshold_len=1)
    
    else:
        ###########################################
        ############# MODEL VERSION 4.#############
        ###########################################
        pointtracker = Flow3D_Extraction((height,width),intrinsic,train_cams,rgbpcd_mod=True,exhaustive_RAFT_mode=exhaustive_RAFT_mode,
                                       debug=False,
                                       save_dir = dataset.source_path,
                                       expname="./",
                                    #    expname="torchversion_pointTrack_ModelVersion4/Exhaustive_selfmadeMonoDynerf_cutroastedbeef",
                                       device = "cpu",threshold_len=1)
    


    if not dataset.exhaustive_training:
        pointtracker.build_correspondence()
        pointtracker.save_correspondence_pairs()
    # ## version 4.0 
    else:
        pointtracker.build_exhaustive_correspondence()
        pointtracker.save_exhaustive_correspondence_pairs()