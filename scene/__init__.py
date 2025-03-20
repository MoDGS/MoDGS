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
import random
import torch
import json
import copy
from PIL import Image
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import BasisGaussianModel
# from scene.original_gaussian_model import GaussianModel as OriginalGaussianModel
from scene.iso_gaussian_model import SeperateRepreIsotropicGaussianModel,SeperateRepreGaussianModel
from scene.PointTrack_gaussian_model import PointTrackIsotropicGaussianModel,TimeTableGaussianModel,Original_GaussianModel
from scene.cameras import Camera,get_realtive_pose,transfrom_by_Relative_Pose
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, cameraList_from_camInfos_without_image, camera_to_JSON
from torch.utils import data
from utils.general_utils import PILtoTorch
from utils.render_utils import forward_circle_poses,forward_circle_poses_for_staticCams
# from pointTrack import unproject_from_depthmap_torch,get_intrinsic_matrix
from utils.graphics_utils import unproject_from_depthmap_torch,get_intrinsic_matrix
from utils.graphics_utils import fov2focal
from PIL import ImageFile
from gaussian_renderer import original_render

ImageFile.LOAD_TRUNCATED_IMAGES = True

GaussianModelTypes = {
    "Kai_GaussianModel": BasisGaussianModel,
    "Original_GaussianModel": Original_GaussianModel,
    "SeperateRepreIsotropicGaussianModel" :SeperateRepreIsotropicGaussianModel,
    "SeperateRepreGaussianModel": SeperateRepreGaussianModel,
    "TimeTable_GaussianModel": TimeTableGaussianModel,
    "PointTrackIsotropicGaussianModel": PointTrackIsotropicGaussianModel,
    # "HyperNeRF": readHypernerfSceneInfo,
}

class Dataset(data.Dataset):
    def __init__(self, cams, args):
        self.cams = cams
        self.args = args

    def __getitem__(self, index):
        cam_info = self.cams[index]
        # image = cam_info.image
        image = Image.open(cam_info.image_path)
        resized_image = torch.from_numpy(np.array(image)) / 255.0

        if len(resized_image.shape) == 3:
            resized_image = resized_image.permute(2, 0, 1)
        else:
            resized_image = resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
        
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                      image=resized_image, gt_alpha_mask=None,
                      image_name=cam_info.image_name, uid=cam_info.uid, time=cam_info.time, data_device=self.args.data_device)        

    def __len__(self):
        return len(self.cams)


class FlowDataset(data.Dataset):
    def __init__(self, cams, args):
        self.cams = cams
        self.args = args

    def __getitem__(self, index):
        cam_info = self.cams[index]
        # image = cam_info.image
        image = Image.open(cam_info.image_path)
        data_root = '/'.join(cam_info.image_path.split('/')[:-2])
        folder = cam_info.image_path.split('/')[-2]
        image_name =  cam_info.image_path.split('/')[-1]
        fwd_flow_path = os.path.join(data_root, 'flow', f'{os.path.splitext(image_name)[0]}_fwd.npz')
        bwd_flow_path = os.path.join(data_root, 'flow', f'{os.path.splitext(image_name)[0]}_bwd.npz')
        # print(fwd_flow_path, bwd_flow_path)
        if os.path.exists(fwd_flow_path):
            fwd_data = np.load(fwd_flow_path)
            fwd_flow = torch.from_numpy(fwd_data['flow'])
            fwd_flow_mask = torch.from_numpy(fwd_data['mask'])
        else:
            fwd_flow, fwd_flow_mask  = None, None
        if os.path.exists(bwd_flow_path):
            bwd_data = np.load(bwd_flow_path)
            bwd_flow = torch.from_numpy(bwd_data['flow'])
            bwd_flow_mask = torch.from_numpy(bwd_data['mask'])
        else:
            bwd_flow, bwd_flow_mask  = None, None
        
        # image = np.zeros((3, 128, 128))
        resized_image = torch.from_numpy(np.array(image)) / 255.0

        if len(resized_image.shape) == 3:
            resized_image = resized_image.permute(2, 0, 1)
        else:
            resized_image = resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
        
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                      image=resized_image, gt_alpha_mask=None,
                      image_name=cam_info.image_name, uid=cam_info.uid,
                      time=cam_info.time, data_device=self.args.data_device,
                      fwd_flow=fwd_flow, fwd_flow_mask=fwd_flow_mask,
                      bwd_flow=bwd_flow, bwd_flow_mask=bwd_flow_mask)

    def __len__(self):
        return len(self.cams)


class Scene:

    gaussians=None# : GaussianModel

    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        
        self.model_path = args.model_path
        if args.timestamp is not None:
            self.timestamp = args.timestamp
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path,self.timestamp, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.vis_cameras = {}
        self.use_loader = False
        import glob

        if (os.path.exists(os.path.join(args.source_path, "sparse")) or os.path.exists(os.path.join(args.source_path, "poses_bounds.npy"))) and os.path.exists(os.path.join(args.source_path, "mv_images")):
            # scene_info = sceneLoadTypeCallbacks["DyColmap"](args.source_path, args.images,args.image_mode, args.eval,initPcdFromfirstframeDepth=args.initPcdFromfirstframeDepth)
            print("Found video file, Selfmade Nvidia Dataset!")
            
            scene_info = sceneLoadTypeCallbacks["SelfMade"](args.source_path,args.images,  args.eval,re_scale_json=None,exhaustive_training=args.exhaustive_training)

            
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            folder=glob.glob(os.path.join(args.source_path, "cam*"))
            if len(folder) ==1:
                print("Found poses_bounds file and only has only ONE cam , assuming MonoDyNeRF data set!")
                scene_info = sceneLoadTypeCallbacks["MonoDyNeRF"](args.source_path,args.images, args.eval)
            
            else:
                print("Found poses_bounds file, assuming DyNeRF data set!")
                scene_info = sceneLoadTypeCallbacks["DyNeRF"](args.source_path, args.eval)
                self.use_loader = True
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            if os.path.exists(os.path.join(args.source_path, "raft_exhaustive")) and  os.path.exists(os.path.join(args.source_path, "emf.json")):
                print("Found *raft_exhaustive* Folder ,dataset.json and emf.json, assuming Iphone_raft_exhaustive_data set!")
                scene_info = sceneLoadTypeCallbacks["RaftExhaustive"](args.source_path, args.eval,args.random_init_pcd)
            elif  os.path.exists(os.path.join(args.source_path, "emf.json")):
                print("Found dataset.json and emf.json, assuming Iphone data set!")
                scene_info = sceneLoadTypeCallbacks["IphoneData"](args.source_path, args.eval,args.random_init_pcd)
            else:
                print("Found dataset.json Only, assuming HyperNeRF data set!")
                scene_info = sceneLoadTypeCallbacks["HyperNeRF"](args.source_path, args.eval)
        elif len(glob.glob(os.path.join(args.source_path, "*.mp4")))>=1:
            print("Found video file, Selfmade Dataset!")
            scene_info = sceneLoadTypeCallbacks["SelfMade"](args.source_path,args.images,  args.eval,re_scale_json=None,exhaustive_training=args.exhaustive_training)
           
        else:
            assert False, "Could not recognize scene type!"
        self.time_delta = scene_info.time_delta

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)


        self.cameras_extent = scene_info.nerf_normalization["radius"]

        

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)

            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Video Cameras")
            self.vis_cameras[resolution_scale] = cameraList_from_camInfos_without_image(scene_info.vis_cameras, resolution_scale, args)
            if shuffle:
                random.shuffle(self.train_cameras[resolution_scale])  # Multi-res consistent random shuffling
                random.shuffle(self.train_cameras[resolution_scale])  # Multi-res consistent random shuffling

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,self.timestamp,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            if args.model_version=="SeperateRepreIsotropicGaussianModel" and args.approx_l_global>0:
                self.gaussians.load_global_feature(os.path.join(self.model_path,self.timestamp,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "global_feature.npy"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            
        self.original_pcd=scene_info
        ## LQM
        self.firstframeCam=[]
        minindex=100000
        firstcam=None
        for cam in self.train_cameras[1.0]:
            if int(cam.image_name[-3:])<minindex:
                minindex=int(cam.image_name[-3:])
                firstcam=cam
                # break
        self.firstframeCam.append(firstcam)
        

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path,self.timestamp, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if hasattr(self.gaussians,"has_global_feature") and self.gaussians.has_global_feature:
            self.gaussians.save_global_feature(os.path.join(point_cloud_path, "global_feature.npy"))
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]
    def getFirstFrameTrainCameras(self, scale=1.0):##
        return self.firstframeCam

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getVisCameras(self, scale=1.0):
        return self.vis_cameras[scale]

from dataloader.timePcdTable_dataset import BaseCorrespondenceDataset
from model.neuralsceneflowprior import BasicTrainer
class PointTrackScene:

    gaussians=None# : GaussianModel
    PCD_INTERVAL=8 ## NOTE: 8 for Iphone dataset
    def __init__(self, args : ModelParams, gaussians:TimeTableGaussianModel,timePcd_dataset:BaseCorrespondenceDataset=None,net_trainer:BasicTrainer=None,load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        
        self.model_path = args.model_path
        self.args=args
        if args.timestamp is not None:
            self.timestamp = args.timestamp
        self.loaded_iter = None
        self.gaussians = gaussians
        re_scale_json=None
        if timePcd_dataset is not None:
            self.timePcd_dataset=timePcd_dataset
            re_scale_json = self.timePcd_dataset.get_rescale_json()
            print("Using TimePcdDataset:{}".format(self.timePcd_dataset.__class__.__name__))
            print("Using Rescale Json:{}".format(re_scale_json))
            self.PCD_INTERVAL= self.timePcd_dataset.PCD_INTERVAL
        if net_trainer is not None:
            self.net_trainer=net_trainer
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path,self.timestamp, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.vis_cameras = {}
        self.spiral_cameras = {}
        self.use_loader = False
        align_metric_cameras=False
        print("Found sparse and mv_image, Selfmade Nvidia/DyNeRF Dataset!")
        if re_scale_json is None:
            try:
                if os.path.exists(os.path.join(args.source_path,"re_scale.json")):
                    print(f"Found re_scale.json, will load it")
                    with open(os.path.join(args.source_path,"re_scale.json"), 'r') as json_file:
                        re_scale_json = json.load(json_file)
                    print(re_scale_json)
            except Exception as e:
                print(f"Error in loading re_scale.json:{e}")
        import glob
        if   ("nvidia" in args.source_path or "DyNeRF" in args.source_path) and os.path.exists(os.path.join(args.source_path, "mv_images")):
            # scene_info = sceneLoadTypeCallbacks["DyColmap"](args.source_path, args.images,args.image_mode, args.eval,initPcdFromfirstframeDepth=args.initPcdFromfirstframeDepth)

                    # re_scale_json = self.timePcd_dataset.get_rescale_json()
                    
            scene_info = sceneLoadTypeCallbacks["SelfMade"](args.source_path,
                                                            args.images, 
                                                            args.eval,re_scale_json=re_scale_json,
                                                            exhaustive_training=args.exhaustive_training,use_depthNonEdgeMsk=args.use_depthNonEdgeMsk )
            if os.path.exists(os.path.join(args.source_path,"sparse","colmap_to_depth_scale.json")):
                colmap_scene_info  = sceneLoadTypeCallbacks["read_Nvidia_cam_info"](args.source_path,)
                align_metric_cameras = True ## For metric evaluation.  表示有 colmap scene info ，需要去校准。
            if os.path.exists(os.path.join(args.source_path,"colmap_to_depth_scale.json")):
                colmap_scene_info  = sceneLoadTypeCallbacks["read_DyNeRF_cam_info"](args.source_path,)
                align_metric_cameras = True ## For metric evaluation.  表示有 colmap scene info ，需要去校准。
                

        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            if os.path.exists(os.path.join(args.source_path, "raft_exhaustive")) and  os.path.exists(os.path.join(args.source_path, "emf.json")):
                print("Found *raft_exhaustive* Folder ,dataset.json and emf.json, assuming Iphone_raft_exhaustive_data set!")
                scene_info = sceneLoadTypeCallbacks["RaftExhaustive"](args.source_path, args.eval,args.random_init_pcd)
            elif os.path.exists(os.path.join(args.source_path, "emf.json")):
                print("Found dataset.json and emf.json, assuming Iphone data set!")
                scene_info = sceneLoadTypeCallbacks["IphoneData"](args.source_path, args.eval,args.random_init_pcd,re_scale_json=re_scale_json)
            else:
                print("Found dataset.json Only, assuming HyperNeRF data set!")
                scene_info = sceneLoadTypeCallbacks["HyperNeRF"](args.source_path, args.eval)

        elif len(glob.glob(os.path.join(args.source_path, "*.mp4")))>=1:
            print("Found video file, Selfmade Dataset!")
            scene_info = sceneLoadTypeCallbacks["SelfMade"](args.source_path,
                                                            args.images, 
                                                            args.eval,re_scale_json=re_scale_json,
                                                            exhaustive_training=args.exhaustive_training,use_depthNonEdgeMsk=args.use_depthNonEdgeMsk )
        else:
            assert False, "Could not recognize scene type!"
        self.time_delta = scene_info.time_delta



        self.cameras_extent = scene_info.nerf_normalization["radius"]

        
        max_frame_T = max([int(cam.image_name[-4:]) for cam in scene_info.train_cameras+scene_info.test_cameras ])
        if hasattr(self.gaussians,"set_max_frame") and isinstance(self.gaussians,TimeTableGaussianModel):
            self.gaussians.set_max_frame(max_frame_T)
        self.Max_frame_T = max_frame_T
        print("Max Frame T:{}".format(max_frame_T))


        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args,self.PCD_INTERVAL)

            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args,self.PCD_INTERVAL)
            print("Loading Video Cameras")
            if shuffle:
                random.shuffle(self.train_cameras[resolution_scale])  # Multi-res consistent random shuffling
                random.shuffle(self.train_cameras[resolution_scale])  # Multi-res consistent random shuffling
            if align_metric_cameras:
                print("Loading Nvidia Colmap Cameras")
                train_cam =cameraList_from_camInfos_without_image(colmap_scene_info["train"], resolution_scale, args)
                test_cams =cameraList_from_camInfos_without_image(colmap_scene_info["test"], resolution_scale, args)

                print(f"Found re_scale.json, will load it")
                if os.path.exists(os.path.join(args.source_path,"sparse","colmap_to_depth_scale.json")):
                    with open(os.path.join(args.source_path,"sparse","colmap_to_depth_scale.json"), 'r') as json_file:
                        colmap_scale = json.load(json_file)
                        colmap_to_depth_scale= colmap_scale["scale"]

                elif os.path.exists(os.path.join(args.source_path,"colmap_to_depth_scale.json")):
                    with open(os.path.join(args.source_path,"colmap_to_depth_scale.json"), 'r') as json_file:
                        colmap_scale = json.load(json_file)
                        colmap_to_depth_scale= colmap_scale["scale"]
                if hasattr(args,"Factor_ColmapDepthAlign"):
                    print("Using Factor_ColmapDepthAlign",args.Factor_ColmapDepthAlign)
                    colmap_to_depth_scale = args.Factor_ColmapDepthAlign*colmap_to_depth_scale
                assert len(train_cam)==1,"Only one camera is supported for now"
                test_cams = get_realtive_pose(train_cam[0],test_cams,colmap_to_depth_scale)
                train_cams  = self.train_cameras[resolution_scale]
                self.test_cameras_metric={}
                for cam in test_cams:
                    transformation_matrix = cam.relative_pose
                    test_cams =  transfrom_by_Relative_Pose(train_cams,transformation_matrix,cam)
                    if not "cam" in  cam.image_name:
                        key = "cam%02d"%(int(cam.image_name)+1)
                    else:
                        key = cam.image_name
                    self.test_cameras_metric[key]=test_cams
                
                
                
                

        print("Generating Spiral Cameras")
        from utils.render_utils import forward_circle_poses
        from scene.dataset_readers import get_caminfo_from_poses,update_fov,update_time
        self.forward_circle_poses = forward_circle_poses(self.train_cameras[1.0])
        FovX= scene_info.train_cameras[0].FovX
        FovY= scene_info.train_cameras[0].FovY
        height= scene_info.train_cameras[0].height
        width = scene_info.train_cameras[0].width
        if scene_info.scene_type=="SelfMade":
            time_list = [-1,]
        else:
            time_list = [-1,]+list(np.linspace(0,1,4))
        for time in time_list:
            # prin("Generating Spiral Cameras at time ",time)
            spiral_caminfo =  get_caminfo_from_poses(self.forward_circle_poses,FovY=FovY,FovX=FovX,width=width,height=height,fix_time=time)
            self.spiral_cameras[time] = cameraList_from_camInfos_without_image(spiral_caminfo, 1.0, args)
        
        print("Spiral Cameras Generated")
        

        print("Generating Spiral Cameras for static cams")
        self.forward_circle_poses = forward_circle_poses_for_staticCams(self.train_cameras[1.0])
        print("Generating Spiral Cameras for static cams",len(self.forward_circle_poses))
        spiral_caminfo =  get_caminfo_from_poses(self.forward_circle_poses,FovY=FovY,FovX=FovX,width=width,height=height,fix_time=-1)
        focal_factor = list(np.linspace(1,0.8,20))+ list(np.linspace(0.8,1.0,20))
        focal_change_info =[]

        self.vis_cameras[1.0]= cameraList_from_camInfos_without_image(spiral_caminfo, 1.0, args)
        
        
        
        

        
        for cam in self.vis_cameras[1.0]:
            print("{:.3f}".format(cam.time),end="")
        import pickle
        with open(os.path.join(self.args.source_path,f"vis_cam_MonoGS.pkl"), 'wb') as f:
            pickle.dump(self.vis_cameras[1.0], f)    
            
            
        ## LQM
        self.firstframeCam=[]
        minindex=100000
        firstcam=None
        for cam in self.train_cameras[1.0]:
            if int(cam.image_name[-3:])<minindex:
                minindex=int(cam.image_name[-3:])
                firstcam=cam
                # break
        self.firstframeCam.append(firstcam)
        
        ### construct camera dict  用于Exhaustive pair trainings. 需要一个dict来存储所有的cameras，便于query。
        self.train_cameras_dict = {}
        train_cams = self.getTrainCameras()
        self.train_cameras_name2idx = {}
        for idx,cam in enumerate(train_cams):
            self.train_cameras_dict[cam.image_name] = cam
            self.train_cameras_name2idx[cam.image_name] = idx
          
          
            
        ## FOR DEPTH Plane loss
        from scene.dataset_readers import get_ray_directions
        from utils.graphics_utils import fov2focal
        self.cam_direction = get_ray_directions(height,width,(fov2focal(FovX,width),fov2focal(FovY,height))) ## # (H, W, 3)
        self.cam_direction = torch.Tensor( self.cam_direction/ np.linalg.norm(self.cam_direction,axis=-1,keepdims=True)).cuda().permute(2,0,1).contiguous()
        
        
        
        
        ### scene_min_depth_max_depth
        cams = self.getTrainCameras()
        depth_ranges = [cam.get_depth_range()   for cam in cams if cam.depth is not None]
        depth_ranges = np.asarray(depth_ranges)
        self.near, self.far = np.min(depth_ranges[:,0]), np.max(depth_ranges[:,1])
        
        
    

        self.is_overfit_aftergrowth = False
        self.growth_info = []

        
        
        


    
       
    def getTrain_cam_Byname(self, imagename):
        return self.train_cameras_dict[imagename]
    
    def get_top_error_frame(self, topk=4    ):
        
        sorted_error_dict = sorted(self.error_dict.items(), key=lambda item: item[1],reverse=True)
        
        return dict([(key,self.render_pkg_dict[key]) for key,_ in sorted_error_dict[:topk]])
        # return sorted_error_dict[:topk])
    
       
        

      

            
 

            
            

    

    def rescale_time(self, viewpoint_time,pcd_time=None): 
        """Rescale the time to the time of the PCD

        Args:
            viewpoint_time (_type_): _description_
            pcd_time (_type_): _description_

        Returns:
            _type_: _description_
        """
        if pcd_time is not None and  abs(viewpoint_time.item() - pcd_time.item())<1e-6:
            return viewpoint_time.unsqueeze(0)
        ## NOTE 2024年5月2日20:20:26 注释掉
        # time =    viewpoint_time.unsqueeze(0)*float(self.Max_frame_T)/(len(self.timePcd_dataset))/self.PCD_INTERVAL
        time =    viewpoint_time.unsqueeze(0)
        return time
        
        
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path,self.timestamp, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]
    def TrainCameras_reset(self,scale=1.0):
        self.train_cameras[scale] = copy.deepcopy(self.train_cameras_copy[scale])
    def updateTrainCameras(self, scale=1.0):
        pass
    
        
    
    def getFirstFrameTrainCameras(self, scale=1.0):##
        return self.firstframeCam

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    def getMetricTestCameras(self):
        if hasattr(self,"test_cameras_metric"):
            return self.test_cameras_metric
        else:
            return None
             # return self.test_cameras_metric[scale]

    def getVisCameras(self, scale=1.0):
        return self.vis_cameras[scale]
    def getSpiralCameras(self, time=-1):
        ## time is the index of the spiral cameras
        return self.spiral_cameras[time]
    def getCoTrainingCameras(self, scale=1.0):
        assert hasattr(self,"timePcd_dataset"), "No timePcd_dataset found"
        viewpoints=  self.getTrainCameras(scale)
        pcd_pair = self.timePcd_dataset.getTrainingPairs()
        if len(viewpoints)==len(pcd_pair):
            # "Viewpoints and Pcd pair not match"
            return list(zip(viewpoints,list(pcd_pair.values())))
        else:  
            # print("camera Length Donot Match.")     
            viewpoints=  self.getTrainCameras(scale)
            pcd_pair = self.timePcd_dataset.getTrainingPairs()
            # assert len(viewpoints)==len(pcd_pair), "Viewpoints and Pcd pair not match"
            coTraining_list= []
            for viewpoint in viewpoints: ## TODO: check the time of the viewpoint
                index = int(int(viewpoint.image_name[-4:])/self.PCD_INTERVAL) ## 取最近的PCD
                coTraining_list.append((viewpoint,pcd_pair[index])) ## FIXME : exhaustive pairs occupy too much memory
                # coTraining_list.append((viewpoint,copy.deepcopy(pcd_pair[index]))) ## copy.deepcopy to avoid change the original pcd_pair
        
        
        return coTraining_list
        
        
    def getCoTrainingCameras_extendsTestCam(self, scale=1.0):
        assert hasattr(self,"timePcd_dataset"), "No timePcd_dataset found"
        viewpoints=  self.getTrainCameras(scale)
        pcd_pair = self.timePcd_dataset.getTrainingPairs()
        assert  len(viewpoints)==len(viewpoints), "Viewpoints and Pcd pair not match"
        
        return list(zip(viewpoints,list(pcd_pair.values())))
    def getCoTestingCameras(self, scale=1.0):
        assert hasattr(self,"timePcd_dataset"), "No timePcd_dataset found"
        viewpoints=  self.getTestCameras(scale)
        pcd_pair = self.timePcd_dataset.getTrainingPairs()
        # assert len(viewpoints)==len(pcd_pair), "Viewpoints and Pcd pair not match"
        cotesting_list= []
        for viewpoint in viewpoints:
            index = int(int(viewpoint.image_name[-4:])/self.PCD_INTERVAL)
            cotesting_list.append((viewpoint,pcd_pair[index]))
        
        
        return cotesting_list
    def sample_target_frame(self,cur_frame_name:str):
        assert hasattr(self,"timePcd_dataset"), "No timePcd_dataset found"
        pass 
    
    def check_valid_CoTrainingPairs(self):
        assert hasattr(self,"timePcd_dataset"), "No timePcd_dataset found"
        zipped_data = self.getCoTrainingCameras()
        for pair in zipped_data:
            viewpoint,pcd_pair = pair
            assert  int(float(viewpoint.image_name[-4:])/self.PCD_INTERVAL)== int(pcd_pair["index"]), "match error"
            # f"Viewpoints{viewpoint.image_name} and Pcd pair{int(pcd_pair["index"])} not match"
        return True

# for cam in scene_info.train_cameras:
#     print( int(cam.image_name),cam.image_name,cam.time)
        # print(cam.image_name)
        # break