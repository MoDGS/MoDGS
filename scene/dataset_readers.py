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
import os
import sys
import glob
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.hyper_camera import Camera as HyperNeRFCamera
from typing import NamedTuple, Optional
import cv2
import imageio
from kornia import create_meshgrid
# import glob
from utils.system_utils import resize_flow
from utils.general_utils import dict_to_tensor
import copy
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float
    mask:Optional[np.array]=None
    coord_scale:Optional[float]=None ### only used for the Iphone dataset, used for the adjust the scale of depth
    depth:Optional[np.array]=None ### only used for the Iphone dataset, used for the adjust the scale of depth
    dict_other:Optional[dict]=None 

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    vis_cameras: list
    nerf_normalization: dict
    ply_path: str
    time_delta: float
    scene_type:Optional[str]=None 

    
def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions
def move_camera_pose(pose, progress,radii = 0.1):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    # radii = 0.01
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose
def move_camera_pose2(pose, progress,radii = 0.1):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    # radii = 0.01
    center = np.array([np.cos(t), -np.sin(t), -0.4*np.sin( 0.5*t)]) * radii
    # center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    # print("progress,",progress,"before:",center)
    # center[2]+=radii
    # print("after:",center)
    pose[:3, 3] += pose[:3, :3] @ center
    return pose


def get_caminfo_from_poses(poses,FovY=None,FovX=None,width=None,height=None,fix_time=-1):
    camera_list = []
    ## Pose: [R|T] w2c matrix with shape 3x4
    for idx, pose in enumerate(poses):
        R= pose[:3,:3].T
        T=pose[:3,3]
        if fix_time == -1: ## 时间变化，根据pose的顺序来
            time=idx/(len(poses)-1) ## LQM:fix bugs ,align with time in readsceneInfo
            # time=id/len(poses)
            # (len(self.forward_circle_poses)-1)
        elif fix_time >=0:
            time = fix_time
        empty_image = np.full((height, width, 3), 255, dtype=np.uint8)
        empty_image=Image.fromarray(empty_image)
        camera_list.append(CameraInfo(uid=-1,
                                      R=R, 
                                      T=T, 
                                      FovX=FovX,
                                      FovY=FovY,
                                      image= empty_image,
                                      height=height,
                                      width=width,
                                      image_path="None",
                                      image_name="None",
                                      time=time,
                                    ))
    return camera_list

def update_fov(cam,FovX,FovY):
    new_cam =copy.deepcopy(cam)
    # new_cam.FovX=FovX
    # new_cam.FovY=FovY
    new_cam =cam._replace(FovX=FovX)._replace(FovY=FovY)
    # print("update fov",new_cam.FovX,new_cam.FovY)
    return new_cam
def update_time(cam,time):
    new_cam =copy.deepcopy(cam)
    # new_cam.FovX=FovX
    # new_cam.FovY=FovY
    new_cam =cam._replace(time=time)
    # print("update time",new_cam.time)
    return new_cam

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo



def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        time=float(image_name)/len(cam_extrinsics)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=time)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
def readDyColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []##cam_extrinsics, cam_intrinsics, images_folder
    import glob
    list_img=glob.glob(os.path.join(images_folder, "*.png"))+glob.glob(os.path.join(images_folder, "*.jpg"))
    len_cam = min(len(cam_extrinsics), len(list_img))
    ##  取最小值，因为有可能有些cam intrinsics，没有对应的图片
    time_steps=torch.linspace(0.0, 1.0, len_cam)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)) ## LQM：2024年3月24日13:33:15。这里把w2c变成c2w了？
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        # image_path = os.path.join(images_folder, os.path.basename(extr.name).replace(".jpg",".png"))
        # mask_path = os.path.join(masks_folder, os.path.basename(extr.name).replace(".jpg",".png"))
        image_name = os.path.basename(image_path).split(".")[0]
        if  not os.path.exists(image_path):
            print("SKip reading Images, as the image does not exist: ",image_path)
            continue
        
        image = Image.open(image_path)
        # mask= cv2.resize(imageio.imread(mask_path)/255., image.size, 
        #                 interpolation=cv2.INTER_NEAREST)
        # mask = np.float32(mask > 1e-3)
        # if image_mode == "fg":
        #     image=Image.fromarray((image*mask[:,:,None]).astype(np.uint8))
        # elif image_mode == "bg":
        #     image=Image.fromarray((image*(1-mask[:,:,None])).astype(np.uint8))
        # elif image_mode == "fg_bg":
        #     # image=Image.fromarray((image*(1-mask[:,:,None])).astype(np.uint8))
        #     pass
        # else:
        #     assert False, "image_mode not handled"
        time=time_steps[int(image_name)].item()
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=time,
                            #   mask=mask
                              )
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']

    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    x_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("x")]
    y_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("y")]
    z_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("z")]
    x_names = sorted(x_names, key = lambda x: int(x.replace('x', '')))
    y_names = sorted(y_names, key = lambda y: int(y.replace('y', '')))
    z_names = sorted(z_names, key = lambda z: int(z.replace('z', '')))
    assert len(x_names) == len(y_names) == len(z_names)
    x = np.zeros((colors.shape[0], len(x_names)))
    y = np.zeros((colors.shape[0], len(y_names)))
    z = np.zeros((colors.shape[0], len(z_names)))
    for idx, attr_name in enumerate(x_names):
        x[:, idx] = np.asarray(plydata.elements[0][attr_name])
    for idx, attr_name in enumerate(y_names):
        y[:, idx] = np.asarray(plydata.elements[0][attr_name])
    for idx, attr_name in enumerate(z_names):
        z[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    positions = np.stack([x, y, z], axis=-1)
    assert len(positions.shape) == 3
    assert positions.shape[-1] == 3
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
def load_for_pure_pcd(path):
    plydata = PlyData.read(path)

    """
    file_path : path to .ply file: '/yy/XX.ply'
    returns : point_cloud: size (N,3)
    """
    # Read the .ply file
    

    # Get the 'vertex' element
    vertex = plydata['vertex']

    # Convert the vertex data to a numpy array
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    #  points
    return BasicPointCloud(points=points[:,None,:], colors=np.random.rand(*points.shape), normals=np.zeros_like(points))

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = []
    for t in range(xyz.shape[1]):
        dtype.extend([(f'x{t}', 'f4'), (f'y{t}', 'f4'), (f'z{t}', 'f4')])
    dtype = dtype + [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz[:, 0, :])

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz.reshape(xyz.shape[0], -1), normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
def unproject_from_depth(c2w,intrinsic,depth_map):
    """generate point clound by unprojecting depth map.             
    Args:
        c2w: 4x4 camera to world matrix.
        intrinsic: 3x3 intrinsic matrix.
        depth_map: HxW depth map.
    """
    H, W = depth_map.shape
    xmap, ymap = np.meshgrid(range(W), range(H), indexing='xy')
    xmap = xmap.flatten()
    ymap = ymap.flatten()
    depth_map = depth_map.flatten()
    cam_points = np.stack([xmap * depth_map, ymap * depth_map, depth_map,
                           np.ones_like(depth_map)], axis=0)
    cam_points = np.linalg.inv(intrinsic) @ cam_points
    cam_points = cam_points[:3, :]
    cam_points = np.linalg.inv(c2w[:3, :3]) @ (cam_points - c2w[:3, 3:])
    return cam_points.T.reshape(H, W, 3)        

def readDyColmapSceneInfo(path, images,image_mode, eval,initPcdFromfirstframeDepth=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    mask_dir="fine_mask"
    cam_infos_unsorted = readDyColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # cam_infos_unsorted = readDyColmapCameras(image_mode,cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir),masks_folder=os.path.join(path, mask_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos)]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) ]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    # vis_cam_infos=[train_cam_infos[12]]
    N_views = len(cam_infos)
    vis_cam_id =12
    vis_cam_info = train_cam_infos[vis_cam_id]
    val_times = torch.linspace(0.0, 1.0, N_views)
    vis_cam_infos=[]
    for idx,itm in enumerate(list(val_times)):
        vis_cam_infos.append(vis_cam_info._replace(time=val_times[idx].item()))


    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D_ours.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        xyz = xyz[:, None, :]
        num_pts=xyz.shape[0]
        # xyz = np.concatenate([xyz, np.zeros((num_pts, 1, 3))], axis=1)
        xyz = np.concatenate([xyz, np.zeros((num_pts, 0, 3))], axis=1) ## TODO weird
        # xyz = xyz[:, :]
        storePly(ply_path, xyz, rgb)

    
    if initPcdFromfirstframeDepth:# is not None:
        pcd= load_for_pure_pcd(os.path.join(path, "init_pcd_from_firstframe_depth.ply"))
            # pass
            # self.initialze_depth
    else:   
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    time_delta = val_times[1].item() - val_times[0].item()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,vis_cameras=vis_cam_infos, time_delta=time_delta)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        # if 'time' in frames[0]:
        #     times = np.array([frame['time'] for idx, frame in enumerate(frames)])
        #     time_idx = times.argsort()
        # else:
        #     time_idx = [0 for f in frames]
        # print(times)
        # print(time_idx)
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            time = frame['time'] if 'time' in frame else 0

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], time=time))
            
    return cam_infos


# https://github.com/albertpumarola/D-NeRF/blob/main/load_blender.py
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    # https://github.com/albertpumarola/D-NeRF/blob/main/load_blender.py
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def generateCamerasFromTransforms(path, transformsfile, extension=".png"):
    cam_infos = []

    # https://github.com/albertpumarola/D-NeRF/blob/main/load_blender.py
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])    

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
    cam_name = os.path.join(path, frames[0]["file_path"] + extension)        
    image_path = os.path.join(path, cam_name)
    image_name = Path(cam_name).stem
    image = Image.open(image_path)
    width = image.size[0]
    height = image.size[1]

    for idx, (c2w, time) in enumerate(zip(render_poses, render_times)):
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        fovy = focal2fov(fov2focal(fovx, width), height)
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                                    image=None, image_path=None, image_name=None,
                                    width=width, height=height, time=time))
            
    return cam_infos


def init_random_points(ply_path):
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        # time_length = max([c.time for c in train_cam_infos]) + 1
        # time_length = 2
        # xyz = np.random.random((num_pts, 1, 3)) * 2.6 - 1.3
        # xyz = np.tile(xyz, (1, time_length, 1))
        # xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3)), np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3))], axis=1)
        xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 16, 3))], axis=1)
        # xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 3, 3)), np.ones((num_pts, 1, 3))], axis=1)
        # xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3)), np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3))], axis=1)
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    pcd = fetchPly(ply_path)
    # except:
        # pcd = None
    return pcd

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    vis_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json")

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d_ours.ply")
    pcd = init_random_points(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           vis_cameras=vis_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, time_delta=1/len(train_cam_infos))
    return scene_info


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses



def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


def readDynerfSceneInfo(path, eval):
    blender2opencv = np.eye(4)
    downsample = 2

    poses_arr = np.load(os.path.join(path, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    near_fars = poses_arr[:, -2:]
    videos = glob.glob(os.path.join(path, "cam??"))
    videos = sorted(videos)
    assert len(videos) == poses_arr.shape[0]

    H, W, focal = poses[0, :, -1]
    focal = focal / downsample
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    poses, pose_avg = center_poses(
        poses, blender2opencv
    )  # Re-center poses so that the average is near the center.
    
    near_original = near_fars.min()
    scale_factor = near_original * 0.75
    near_fars /= (
        scale_factor  # rescale nearest plane so that it is at z = 4/3.
    )
    # print(scale_factor)
    poses[..., 3] /= scale_factor
    
    image_dirs = [video.replace('.mp4', '') for video in videos]
    val_index = [0]
    images = [sorted(glob.glob(os.path.join(d, "images/*.png")), key=lambda x:int(os.path.splitext(os.path.basename(x))[0]))[:300] for d in image_dirs]
    train_cam_infos = []
    for idx, image_paths in enumerate(images):
        if idx in val_index:
            continue
        p = poses[idx]
        for image_path in image_paths:
            image_name = os.path.basename(image_path).split(".")[0]
            time  = float(image_name) / 300
            image = Image.open(image_path)
            uid = idx * 1000 + int(image_name)
            pose = np.eye(4)
            pose[:3, :] = p[:3, :]
            R = -pose[:3, :3]
            R[:, 0] = -R[:, 0]
            T = -pose[:3, 3].dot(R)
            height = image.height
            width = image.width
            FovY = focal2fov(focal, height)
            FovX = focal2fov(focal, width)
            # R = pose[:3, :3]
            # T  = pose[:3, 3]

            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                  image_path=image_path, image_name=image_name, width=width, height=height, time=time)
            train_cam_infos.append(cam_info)

    test_cam_infos = []
    for idx, image_paths in enumerate(images):
        if idx not in val_index:
            continue
        p = poses[idx]
        for image_path in image_paths:
            image_name = os.path.basename(image_path).split(".")[0]
            time  = float(image_name) / 300
            image = Image.open(image_path)
            uid = idx * 1000 + int(image_name)
            pose = np.eye(4)
            pose[:3, :] = p[:3, :]
            R = -pose[:3, :3]
            R[:, 0] = -R[:, 0]
            T = -pose[:3, 3].dot(R)
            # R = pose[:3, :3]
            # T  = pose[:3, 3]
            
            height = image.height
            width = image.width
            FovY = focal2fov(focal, height)
            FovX = focal2fov(focal, width)

            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                  image_path=image_path, image_name=image_name, width=width, height=height, time=time)
            test_cam_infos.append(cam_info)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    widht, height = train_cam_infos[0].width, train_cam_infos[0].height
    # Sample N_views poses for validation - NeRF-like camera trajectory.
    N_views = 120
    val_poses = get_spiral(poses, near_fars, N_views=N_views)
    val_times = torch.linspace(0.0, 1.0, val_poses.shape[0])
    vis_cam_infos = []
    for idx, (pose, time) in enumerate(zip(val_poses, val_times)):
        p = pose
        uid = idx
        pose = np.eye(4)
        pose[:3, :] = p[:3, :]
        R = -pose[:3, :3]
        R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)
        # R = pose[:3, :3]
        # T  = pose[:3, 3]
            
        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=None, image_name=None, width=width, height=height, time=time)
        vis_cam_infos.append(cam_info)


    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d_ours.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2_000 # 100_000
        print(f"Generating random point cloud ({num_pts})...")
        threshold = 3
        xyz_max = np.array([1.5*threshold, 1.5*threshold, -0*threshold])
        xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])        
        xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 16, 3))], axis=1)
        # xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3))], axis=1)

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           vis_cameras =vis_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, time_delta=1/300)
    return scene_info



def readHypernerfCamera(uid, camera, image_path, time):
    height, width = int(camera.image_shape[0]), int(camera.image_shape[1])
    image_name = os.path.basename(image_path).split(".")[0]
    R = camera.orientation.T
    # T = camera.translation.T
    T = - camera.position @ R
    image = Image.open(image_path)    
    FovY = focal2fov(camera.focal_length, height)
    FovX = focal2fov(camera.focal_length, width)
    return CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                      image_path=image_path, image_name=image_name, width=width, height=height, time=time)    


def readHypernerfSceneInfo(path, eval):
    # borrow code from https://github.com/hustvl/TiNeuVox/blob/main/lib/load_hyper.py
    use_bg_points = False
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)    
    
    near = scene_json['near']
    far = scene_json['far']
    coord_scale = scene_json['scale']
    scene_center = scene_json['center']
    
    all_imgs = dataset_json['ids']
    val_ids  = dataset_json['val_ids']
    add_cam = False
    if len(val_ids) == 0:
        i_train = np.array([i for i in np.arange(len(all_imgs)) if (i%4 == 0)])
        i_test = i_train+2
        i_test = i_test[:-1,]
    else:
        add_cam = True
        train_ids = dataset_json['train_ids']
        i_test = []
        i_train = []
        for i in range(len(all_imgs)):
            id = all_imgs[i]
            if id in val_ids:
                i_test.append(i)
            if id in train_ids:
                i_train.append(i)

    print('i_train',i_train)
    print('i_test',i_test)
    all_cams = [meta_json[i]['camera_id'] for i in all_imgs]
    all_times = [meta_json[i]['time_id'] for i in all_imgs]
    max_time = max(all_times)
    all_times = [meta_json[i]['time_id']/max_time for i in all_imgs]
    selected_time = set(all_times)
    ratio = 0.5

    all_cam_params = []
    for im in all_imgs:
        camera = HyperNeRFCamera.from_json(f'{path}/camera/{im}.json')
        camera = camera.scale(ratio)
        camera.position = camera.position - scene_center
        camera.position = camera.position * coord_scale
        all_cam_params.append(camera)

    all_imgs = [f'{path}/rgb/{int(1/ratio)}x/{i}.png' for i in all_imgs]
    h, w = all_cam_params[0].image_shape
    if use_bg_points:
        with open(f'{path}/points.npy', 'rb') as f:
            points = np.load(f)
        bg_points = (points - scene_center) * coord_scale
        bg_points = torch.tensor(bg_points).float()    

    train_cam_infos = [readHypernerfCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_train]
    test_cam_infos = [readHypernerfCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_test]

    vis_cam_infos = [readHypernerfCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in np.argsort(all_cams, kind='stable')]

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "points3d_ours.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")
    #     threshold = 3
    #     xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
    #     xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])        
    #     xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 10, 3))], axis=1)

    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)

    # pcd = fetchPly(ply_path)

    ply_path = os.path.join(path, "points.npy")
    xyz = np.load(ply_path, allow_pickle=True)
    xyz = (xyz - scene_center) * coord_scale
    xyz = xyz.astype(np.float32)[:, None, :]
    xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 12, 3))], axis=1)
    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))   

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           vis_cameras=vis_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, time_delta=1/max_time)
    return scene_info

def readIphoneDataCamera(uid, camera, image_path, time):
    """Reads a camera from the iPhone data set. Mainly borrow from readHypernerfCamera"""
    height, width = int(camera.image_shape[0]), int(camera.image_shape[1])
    image_name = os.path.basename(image_path).split(".")[0]
    R = camera.orientation.T
    # T = camera.translation.T
    T = - camera.position @ R
    # R = camera.orientation
    # # T = camera.translation.T
    # T = camera.position
    try:
        image = Image.open(image_path)    
    except:
        # print(f" Image Not Found: {image_path}")
        return None
    FovY = focal2fov(camera.focal_length, height)
    FovX = focal2fov(camera.focal_length, width)
    mask=None
    mask_val_path=os.path.join(os.path.dirname(image_path.replace("rgb","covisible").replace("1x","2x")),"val",image_name+".png")
    if os.path.exists(mask_val_path):
        mask= np.asarray(imageio.imread(mask_val_path)/255.)### 只会读取验证集，covisible的mask
    cam= CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                      image_path=image_path, image_name=image_name, width=width, height=height,mask=mask, time=time,coord_scale=camera.coord_scale)    
    # cam.coord_scale = camera.coord_scale
    return cam 
def readIphoneDataSceneInfo(path, eval,random_init_pcd=False,re_scale_json=None):
    use_bg_points = False
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)    
    
    near = scene_json['near']
    far = scene_json['far']
    coord_scale = scene_json['scale']
    scene_center = scene_json['center']
    
    all_imgs = dataset_json['ids']
    val_ids  = dataset_json['val_ids']
    add_cam = False
    if len(val_ids) == 0:
        i_train = np.array([i for i in np.arange(len(all_imgs)) if (i%4 == 0)])
        i_test = i_train+2
        i_test = i_test[:-1,]
    else:
        add_cam = True
        train_ids = dataset_json['train_ids']
        i_test = []
        i_train = []
        for i in range(len(all_imgs)):
            id = all_imgs[i]
            if id in val_ids:
                i_test.append(i)
            if id in train_ids:
                i_train.append(i)

    print('i_train',i_train)
    print('i_test',i_test)
    all_cams = [meta_json[i]['camera_id'] for i in all_imgs]
    all_times = [meta_json[i]['appearance_id'] for i in all_imgs] ## TODO: 这里的time是appearance_id
    time_scalefactor= 1.0
    max_time = max(all_times)/time_scalefactor ##为了把时间缩小到time_scalefactor倍
    all_times = [meta_json[i]['appearance_id']/max_time for i in all_imgs]
    selected_time = set(all_times)
    ratio = 0.5
    ## FIXME LQM_DEBUG_SCALE_FACTOR 这个参数是为了验证diff-rasterization的bug而做的。
    LQM_DEBUG_SCALE_FACTOR=1
    all_cam_params = []
    for im in all_imgs:
        camera = HyperNeRFCamera.from_json(f'{path}/camera/{im}.json')
        camera = camera.scale(ratio)
        camera.coord_scale = coord_scale
        camera.position = camera.position - scene_center
        camera.position = camera.position * coord_scale*LQM_DEBUG_SCALE_FACTOR
        if re_scale_json is not None:
            mean_xyz = np.array(re_scale_json["mean_xyz"])
            scale_xyz = np.array(re_scale_json["scale"])
            camera.position = (camera.position - mean_xyz)*scale_xyz
            camera.coord_scale = coord_scale*scale_xyz ## 后面在load cam的时候还会用到。所以需要在这里改。
        
        all_cam_params.append(camera)

    all_imgs = [f'{path}/rgb/{int(1/ratio)}x/{i}.png' for i in all_imgs]
    h, w = all_cam_params[0].image_shape
    if use_bg_points:
        with open(f'{path}/points.npy', 'rb') as f:
            points = np.load(f)
        bg_points = (points - scene_center) * coord_scale*LQM_DEBUG_SCALE_FACTOR
        bg_points = torch.tensor(bg_points).float()    
    print("No Camera Rotation")
    
    
    
    # train_cam_infos = [readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_train]
    # test_cam_infos = [readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_test]
    # vis_cam_infos = [readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in np.argsort(all_cams, kind='stable')]
    
    train_cam_infos=[]
    for i in i_train:
        cam= readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i])
        if cam is not None:
            train_cam_infos.append(cam)
    # train_cam_infos = [readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_train[:30]]
    test_cam_infos = []
    for i in i_test:
        cam= readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i])
        if cam is not None:
            test_cam_infos.append(cam)
    vis_cam_infos = []
    # for i in np.argsort(all_cams, kind='stable'):
    #     cam= readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i])
    #     if cam is not None:
    #         vis_cam_infos.append(cam)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        # test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    #
    # ## Random Init PCD
    # ply_path = os.path.join(path, "points3d_ours.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")
    #     threshold = 3
    #     xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
    #     xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])        
    #     xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 10, 3))], axis=1)
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # pcd = fetchPly(ply_path)
    

    ply_path = os.path.join(path, "points.npy")
    xyz = np.load(ply_path, allow_pickle=True)
    xyz = (xyz - scene_center) * coord_scale*LQM_DEBUG_SCALE_FACTOR
    if random_init_pcd:
        # xyz = xyz[np.random.choice(xyz.shape[0], 10000)]
        num_pts = 50000
        xyz_max = np.max(xyz, axis=0)
        xyz_min= np.min(xyz, axis=0)
        xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 16, 3))], axis=1)
    else:  
        xyz = xyz.astype(np.float32)[:, None, :]
        xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 12, 3))], axis=1)
    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))   

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           vis_cameras=vis_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, time_delta=1/max_time)
    return scene_info



def readRaftExhaustiveDataCamera(uid, camera, image_path, time):
    """Reads a camera from the iPhone data set. Mainly borrow from readHypernerfCamera"""
    height, width = int(camera.image_shape[0]), int(camera.image_shape[1])
    image_name = os.path.basename(image_path).split(".")[0]
    R = camera.orientation.T
    # T = camera.translation.T
    T = - camera.position @ R
    # R = camera.orientation
    # # T = camera.translation.T
    # T = camera.position
    try:
        image = Image.open(image_path)    
    except:
        # print(f" Image Not Found: {image_path}")
        return None
    FovY = focal2fov(camera.focal_length, height)
    FovX = focal2fov(camera.focal_length, width)
    mask=None
    mask_val_path=os.path.join(os.path.dirname(image_path.replace("rgb","covisible").replace("1x","2x")),"val",image_name+".png")
    if os.path.exists(mask_val_path):
        mask= np.asarray(imageio.imread(mask_val_path)/255.)### 只会读取验证集，covisible的mask
    basedir ="/".join(image_path.split("/")[:-3])
    exhaustive_raft_dirs = sorted(glob.glob(os.path.join(basedir,"raft_exhaustive",image_name+".png"+"*.npy")))
    exhaustive_raft_mask_dirs = sorted(glob.glob(os.path.join(basedir,"raft_masks",image_name+".png"+"*.png")))
    assert len(exhaustive_raft_dirs)==len(exhaustive_raft_mask_dirs), "raft and mask not match"
    raft_dict = {}
    raft_msks_dict = {}
    for raft_dir,msk_dir in zip(exhaustive_raft_dirs,exhaustive_raft_mask_dirs):
        raft_name  = ''.join(os.path.basename(raft_dir).split(".")[:-1]).replace("png","").replace("jpg","")
        raft_msk_name  = ''.join(os.path.basename(msk_dir).split(".")[:-1]).replace("png","").replace("jpg","")
        assert raft_name == raft_msk_name , "raft and mask not match"
        raft_np= np.load(raft_dir)
        raft_dict[raft_name]=raft_np
        raft_msks_dict[raft_msk_name]=np.asarray(imageio.imread(msk_dir)>0)
    
    cam= CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                      image_path=image_path, image_name=image_name, width=width, height=height,
                      mask=mask, time=time,coord_scale=camera.coord_scale,dict_other={"rafts":raft_dict,"raft_msks":raft_msks_dict})    
    # cam.coord_scale = camera.coord_scale
    return cam 
def readRaftExhaustiveDataSceneInfo(path, eval,random_init_pcd=False,re_scale_json=None):
    use_bg_points = False
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)    
    
    near = scene_json['near']
    far = scene_json['far']
    coord_scale = scene_json['scale']
    scene_center = scene_json['center']
    
    all_imgs = dataset_json['ids']
    val_ids  = dataset_json['val_ids']
    add_cam = False
    if len(val_ids) == 0:
        i_train = np.array([i for i in np.arange(len(all_imgs)) if (i%4 == 0)])
        i_test = i_train+2
        i_test = i_test[:-1,]
    else:
        add_cam = True
        train_ids = dataset_json['train_ids']
        i_test = []
        i_train = []
        for i in range(len(all_imgs)):
            id = all_imgs[i]
            if id in val_ids:
                i_test.append(i)
            if id in train_ids:
                i_train.append(i)

    print('i_train',i_train)
    print('i_test',i_test)
    all_cams = [meta_json[i]['camera_id'] for i in all_imgs]
    all_times = [meta_json[i]['appearance_id'] for i in all_imgs] ## TODO: 这里的time是appearance_id
    time_scalefactor= 1.0
    max_time = max(all_times)/time_scalefactor ##为了把时间缩小到time_scalefactor倍
    all_times = [meta_json[i]['appearance_id']/max_time for i in all_imgs]
    selected_time = set(all_times)
    ratio = 0.5
    ## FIXME LQM_DEBUG_SCALE_FACTOR 这个参数是为了验证diff-rasterization的bug而做的。
    LQM_DEBUG_SCALE_FACTOR=1
    all_cam_params = []
    for im in all_imgs:
        camera = HyperNeRFCamera.from_json(f'{path}/camera/{im}.json')
        camera = camera.scale(ratio)
        camera.coord_scale = coord_scale
        camera.position = camera.position - scene_center
        camera.position = camera.position * coord_scale*LQM_DEBUG_SCALE_FACTOR
        if re_scale_json is not None:
            mean_xyz = np.array(re_scale_json["mean_xyz"])
            scale_xyz = np.array(re_scale_json["scale"])
            camera.position = (camera.position - mean_xyz)*scale_xyz
            camera.coord_scale = coord_scale*scale_xyz ## 后面在load cam的时候还会用到。所以需要在这里改。
        
        all_cam_params.append(camera)

    all_imgs = [f'{path}/rgb/{int(1/ratio)}x/{i}.png' for i in all_imgs]
    h, w = all_cam_params[0].image_shape
    if use_bg_points:
        with open(f'{path}/points.npy', 'rb') as f:
            points = np.load(f)
        bg_points = (points - scene_center) * coord_scale*LQM_DEBUG_SCALE_FACTOR
        bg_points = torch.tensor(bg_points).float()    
    print("No Camera Rotation")
    
    
    
    # train_cam_infos = [readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_train]
    # test_cam_infos = [readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_test]
    # vis_cam_infos = [readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in np.argsort(all_cams, kind='stable')]
    
    train_cam_infos=[]
    for i in i_train:
        cam= readRaftExhaustiveDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i])
        if cam is not None:
            train_cam_infos.append(cam)
    # train_cam_infos = [readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_train[:30]]
    test_cam_infos = []
    for i in i_test:
        cam= readRaftExhaustiveDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i])
        if cam is not None:
            test_cam_infos.append(cam)
    vis_cam_infos = []
    # for i in np.argsort(all_cams, kind='stable'):
    #     cam= readIphoneDataCamera(i, all_cam_params[i], all_imgs[i], all_times[i])
    #     if cam is not None:
    #         vis_cam_infos.append(cam)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        # test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    #
    # ## Random Init PCD
    # ply_path = os.path.join(path, "points3d_ours.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")
    #     threshold = 3
    #     xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
    #     xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])        
    #     xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 10, 3))], axis=1)
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # pcd = fetchPly(ply_path)
    

    ply_path = os.path.join(path, "points.npy")
    xyz = np.load(ply_path, allow_pickle=True)
    xyz = (xyz - scene_center) * coord_scale*LQM_DEBUG_SCALE_FACTOR
    if random_init_pcd:
        # xyz = xyz[np.random.choice(xyz.shape[0], 10000)]
        num_pts = 50000
        xyz_max = np.max(xyz, axis=0)
        xyz_min= np.min(xyz, axis=0)
        xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 16, 3))], axis=1)
    else:  
        xyz = xyz.astype(np.float32)[:, None, :]
        xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 12, 3))], axis=1)
    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))   

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           vis_cameras=vis_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, time_delta=1/max_time)
    return scene_info
    
    pass
    
def readMonoDyNeRFSceneInfo(path, images, eval,initPcdFromfirstframeDepth=False):


    # cam_idxs= list(range(start,end))
    focal = 1460.7543404163334
    identical_pose=np.eye(4)
    # cam_intrinsic=np.array( [ 1.0923847397901995e+03, 0., 5.3667369287475469e+02, 0.,
    #    1.0970820622714029e+03, 4.9056644177497316e+02, 0., 0., 1. ]).reshape(3,3)
    reading_dir = "images" if images == None else images
    mask_dir="images"
    images_folder=os.path.join(path, reading_dir)
    image_paths =sorted(glob.glob(os.path.join(images_folder, "*.png")), key=lambda x:int(os.path.splitext(os.path.basename(x))[0]))[:300]
    train_cam_infos = []
    for idx, image_path in enumerate(image_paths):
        # /
        # for image_path in image_paths:
        image_name = os.path.basename(image_path).split(".")[0]
        time  = float(image_name) / 300
        image = Image.open(image_path)
        uid = idx * 1000 + int(image_name)
        pose = np.copy(identical_pose)
        # pose[:3, :] = p[:3, :]

        # R = -pose[:3, :3]
        # R[:, 0] = -R[:, 0]
        
        R = pose[:3, :3]
        # R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)
        height = image.height
        width = image.width
        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)
        # R = pose[:3, :3]
        # T  = pose[:3, 3]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, time=time,mask=None)
        train_cam_infos.append(cam_info)




    # train_cam_infos = cam_infos
    test_cam_infos = train_cam_infos[:10]
    # vis_cam_infos=[train_cam_infos[12]]
    N_views = 300
    vis_cam_info = train_cam_infos[12]
    val_times = torch.linspace(0.0, 1.0, N_views)
    vis_cam_infos=[]
    for idx,itm in enumerate(list(val_times)):
        vis_cam_infos.append(vis_cam_info._replace(time=val_times[idx].item()))


    # nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization = {"translate": np.array([0,0,-1]), "radius": 3}

    ply_path = os.path.join(path, "D3dgs_points3D_ours.ply")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        ## random sample points in world space [-1,-1,-1] to [1,1,1]
        num_pts =5000
        print(f"Generating random point cloud ({num_pts})...")
        
        print(f"Generating random point cloud ({num_pts})...")
        threshold = 2
        xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
        xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])        
        xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 0, 3))], axis=1)
        # xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3))], axis=1)

        shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        pcd = BasicPointCloud(points=xyz, colors=np.random.rand(xyz.shape[0],3), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos[:30],
                           test_cameras=test_cam_infos[:3],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,vis_cameras=vis_cam_infos[:30], time_delta=1/len(train_cam_infos))
    return scene_info


import numpy as np

def rotate_camera_around_z(camera_position, angle_in_degrees):
    # Convert the angle to radians
    angle_in_radians = np.deg2rad(angle_in_degrees)

    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_in_radians), -np.sin(angle_in_radians), 0],
        [np.sin(angle_in_radians), np.cos(angle_in_radians), 0],
        [0, 0, 1]
    ])

    # Rotate the camera position
    rotated_camera_position = np.dot(rotation_matrix, camera_position)

    return rotated_camera_position

def readSelfMade_ExhautiveInfo(image_path,hw ,):
    height, width = hw
    basedir ="/".join(image_path.split("/")[:-3])
    image_name = os.path.basename(image_path).split(".")[0]
    exhaustive_raft_dirs = sorted(glob.glob(os.path.join(basedir,"raft_exhaustive",image_name+".png"+"*.npy")))
    exhaustive_raft_mask_dirs = sorted(glob.glob(os.path.join(basedir,"raft_masks",image_name+".png"+"*.png")))
    if len(exhaustive_raft_dirs)==0 or len(exhaustive_raft_mask_dirs)==0 :
        exhaustive_raft_dirs = sorted(glob.glob(os.path.join(basedir,"raft_exhaustive",image_name+".jpg"+"*.npy")))
        exhaustive_raft_mask_dirs = sorted(glob.glob(os.path.join(basedir,"raft_masks",image_name+".jpg"+"*.png")))

    assert len(exhaustive_raft_dirs)==len(exhaustive_raft_mask_dirs), "raft and mask not match"
    raft_dict = {}
    raft_msks_dict = {}
    for raft_dir,msk_dir in zip(exhaustive_raft_dirs,exhaustive_raft_mask_dirs):
        raft_name  = ''.join(os.path.basename(raft_dir).split(".")[:-1]).replace("png","").replace("jpg","")
        raft_msk_name  = ''.join(os.path.basename(msk_dir).split(".")[:-1]).replace("png","").replace("jpg","")
        assert raft_name == raft_msk_name , "raft and mask not match"
        raft_np= np.load(raft_dir)
        resized_raft = resize_flow(raft_np,height,width)
        resized_mask= cv2.resize(imageio.imread(msk_dir)/255., (width,height),  interpolation=cv2.INTER_NEAREST)
        raft_dict[raft_name]=resized_raft
        raft_msks_dict[raft_msk_name]=np.asarray(resized_mask)>0
    
    dict_other={"rafts":raft_dict,"raft_msks":raft_msks_dict}
    return dict_other
    
def readSelfMadeSceneInfo(path, images=None, eval=False,re_scale_json=None,exhaustive_training=False,use_depthNonEdgeMsk=True):
    red_color = "\033[91m"
    reset_color = "\033[0m"

    use_depthNonEdgeMsk=True
    # use_depthNonEdgeMsk=False
    # 打印当前时间，使用红色字体
    # print(f"{red_color}+++++: setting use depth mask ==False")
    print(f"{red_color}+++++: setting use depth mask =={use_depthNonEdgeMsk},{reset_color}")
    import json 
    json_path = os.path.join(path, "__scene.json")
    scene_json=None
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            scene_json = json.load(f)
    else:
        default_s=0.3
        default_t=0.5
        print("Scene Json Not Found.Using deafult scale and shift",default_s,default_t)
        
    rescale_scene_json=None
    if os.path.exists(os.path.join(path, "__re_scale_scene.json")):
        rescale_scene_json=json.load(open(os.path.join(path, "__re_scale_scene.json")))
    # cam_idxs= list(range(start,end))
    if "DyNeRF" in path:
        focal = 1462.12543404163334/2 ## 给定2x的尺度。
        print("DyNeRF Focal: 1462.12543404163334/2")
    else: ## 其他场景假定focal 都是800# #FIXME: 实际上Nvidia的focal是变幻的
        focal = 800
    identical_pose=np.eye(4)
    # cam_intrinsic=np.array( [ 1.0923847397901995e+03, 0., 5.3667369287475469e+02, 0.,
    #    1.0970820622714029e+03, 4.9056644177497316e+02, 0., 0., 1. ]).reshape(3,3)
    reading_dir = "rgb/2x" if images == None else images
    scale_ratio = float(2/int(images[-2].strip("/"))) if images is not None else 1  ## 如果这里是None，那么就是1，如果是4x，那么就是1/2
    focal = focal*scale_ratio
    print("scaled focal ",focal)
    images_folder=os.path.join(path, reading_dir)
    image_paths =sorted(glob.glob(os.path.join(images_folder, "*.png"))+glob.glob(os.path.join(images_folder, "*.jpg")),\
                    key=lambda x:int(os.path.splitext(os.path.basename(x))[0])) 
    assert len(image_paths)>0, "No images found."
    if os.path.exists(os.path.join(path, "rgb_interlval1")):
        max_time =  len(glob.glob(os.path.join(path, "rgb_interlval1", "*.png"))\
                +glob.glob(os.path.join(path, "rgb_interlval1", "*.jpg")))-1
    else:
        max_time =max([ int(os.path.splitext(os.path.basename(imgpath))[0])   for imgpath in image_paths ])
    print("Max Time: ",max_time)
    train_cam_infos = []
    test_cam_infos=[]
    # test_infos=[]
    N_views= len(image_paths)
    if  os.path.exists(os.path.join(path, "GeoWizardOut/depth_valid_msk")) and use_depthNonEdgeMsk:
        print("depth_valid_msk exists, Use it to filter out edge points.")
    ### LOAD Training Cameras.
    # print("Two Frame Overfit.")
    # for idx, image_path in enumerate(image_paths[:1]):
    for idx, image_path in enumerate(image_paths):
        # /
        # for image_path in image_paths:
        
        image_name = os.path.basename(image_path).split(".")[0]
        time  = float(image_name)/max_time
        print("{:.3f}".format(time),end=" ")
        image = Image.open(image_path)
        uid = idx * 1000 + int(image_name)
        pose = np.copy(identical_pose)
        # pose[:3, :] = p[:3, :]

        # R = -pose[:3, :3]
        # R[:, 0] = -R[:, 0]
        
        R = pose[:3, :3]
        if re_scale_json is not None:
            mean_xyz = np.array(re_scale_json["mean_xyz"])
            scale_xyz = np.array(re_scale_json["scale"])
            pose[:3, 3] = (pose[:3, 3] - mean_xyz)*scale_xyz

        # R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)
        height = image.height
        width = image.width
        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)
        depht_path = os.path.join(path,"GeoWizardOut/depth_npy",image_name+".npy")
        try:
            depth = np.load(depht_path)
                        ## for rescale predicted Mono-Depth from Geowizard
            if scene_json is not None:
                if not np.isnan(scene_json[image_name+".npy"]["scale"]) or  not np.isnan(scene_json[image_name+".npy"]["shift"]):
                    depth =scene_json[image_name+".npy"]["scale"]*depth + scene_json[image_name+".npy"]["shift"]
                else: 
                    print(f"depth {image_name} has nan, using mean scale and shift instead.")
                    depth =scene_json["mean_s"]*depth + scene_json["mean_t"]
            else:
                depth =default_s*depth + default_t ## default scale and shift
            if rescale_scene_json is not None and image_name+".npy" in rescale_scene_json.keys():
                depth = depth*rescale_scene_json[image_name+".npy"] 
            
            if re_scale_json is not None:
                depth = depth*scale_xyz
            depth= cv2.resize(depth,(width,height))
        except:
            print(f"depth {image_name} not found.")
            depth=None


        if depth is not None and  os.path.exists(os.path.join(path, "GeoWizardOut/depth_valid_msk")) and  use_depthNonEdgeMsk:
            depth_valid_msk  = Image.open(os.path.join(path, "GeoWizardOut/depth_valid_msk",image_name+".npy"+".png"))
            depth_valid_msk= depth_valid_msk.resize((width,height))
            depth = depth*(np.asarray(depth_valid_msk )>0.)
            # pass
        dict_other= None    
        if exhaustive_training:
            try:
                dict_other =readSelfMade_ExhautiveInfo(image_path,(height,width))
                dict_other =dict_to_tensor(dict_other,device="cpu")
            except:
                print(f"Exhaustive Info {image_name} not found.")
                dict_other=None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height,depth=depth,
                                dict_other= dict_other if dict_other is not None else None,
                                time=time,mask=None)
        train_cam_infos.append(cam_info)
  
    ## Ugly Codes  #FIXME later 
    for idx, image_path in enumerate(image_paths):
        
        image_name = os.path.basename(image_path).split(".")[0]
        time  = float(image_name) / max_time
        image = Image.open(image_path)
        uid = idx * 1000 + int(image_name)
        pose = np.copy(identical_pose)
        # pose[:3, :] = p[:3, :]

        # R = -pose[:3, :3]
        # R[:, 0] = -R[:, 0]
        
        if re_scale_json is not None:
            mean_xyz = np.array(re_scale_json["mean_xyz"])
            scale_xyz = np.array(re_scale_json["scale"])
            pose[:3, 3] = (pose[:3, 3] - mean_xyz)*scale_xyz
        pose = move_camera_pose(pose, idx/len(image_paths),radii = 0.15)
        R = pose[:3, :3]
        # R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)
        height = image.height
        width = image.width
        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)


        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height,depth=None, time=time,mask=None)
        test_cam_infos.append(cam_info)
    
    





    vis_cam_info = train_cam_infos[0]
    val_times = torch.linspace(0.0, 1.0, N_views)
    vis_cam_infos=[]
    for idx,itm in enumerate(list(val_times)):
        vis_cam_infos.append(vis_cam_info._replace(time=val_times[idx].item()))


    # nerf_normalization = getNerfppNorm(train_cam_infos)  ## since we normalize the scene into [-1,1], we don't need to normalize the nerf.
    nerf_normalization = {"translate": np.array([0,0,-1]), "radius": 1.0}

    ply_path = os.path.join(path, "D3dgs_points3D_ours.ply")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        num_pts =5000
        print(f"Generating random point cloud ({num_pts})...")
        
        threshold = 2
        xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
        xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])        
        xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 0, 3))], axis=1)

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=np.random.rand(xyz.shape[0],3), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_delta=1/len(train_cam_infos),
                           scene_type="SelfMade")
    return scene_info



def read_Nvidia_cam_Info(path, images=None):
    train_cam_id=4
    test_cam_id =[3,5,]
    # test_cam_id =[0,1,2,3,5,6,7,8,9,10]
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    reading_dir = "images" if images == None else images
    # mask_dir="fine_mask"
    # train_cam = None
    cam_infos_unsorted = readDyColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos = [cam_infos[train_cam_id-1],]
    test_cam_infos = [cam_infos[i-1] for i in test_cam_id]
    
    
    return {"train":train_cam_infos, "test":test_cam_infos}

## mianly brrowed from sc-gs
def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    img_paths = sorted(glob.glob(os.path.join(path, 'mv_images/00000/*.png'))+glob.glob(os.path.join(path, 'mv_images/00000/*.jpg')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    cam_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in cam_list:
        img_path =img_paths[i]
        c2w = poses[i]
        # images_names = sorted(os.listdir(video_path))
        n_frames = num_images
        image_name = os.path.basename(img_path).split(".")[0]
        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        # for idx, image_name in enumerate(images_names[:1]):
        # for idx, image_name in enumerate(images_names[:num_images]):
        image_path = img_path
        image = Image.open(image_path)
        frame_time = 0.0 / 1000000
        idx=0 

        FovX = focal2fov(focal, image.size[0])
        FovY = focal2fov(focal, image.size[1])

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                    image=image,
                                    image_path=image_path, image_name=image_name,
                                    width=image.size[0], height=image.size[1], time=frame_time))

    return cam_infos

## mianly brrowed from sc-gs
def read_DyNeRF_cam_info(path,):
    
    hold_id=[0]
    train_cam_id=0
    # test_cam_id = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,]## FIXME for teasr make/
    test_cam_id = [5,6,]## FIXME for teasr make
    
    print("Reading Training Camera")### 这里需要反过来。
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="test", hold_id=hold_id,
                                            num_images=1)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id, 
                                            num_images=1)
    # print("Reading Training Camera")
    # train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
    #                                         num_images=1)

    # print("Reading Training Camera")
    # test_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="test", hold_id=hold_id, 
    #                                         num_images=1)

    # train_cam_infos = [train_cam_infos[train_cam_id-1],]
    test_cam_infos = sorted(test_cam_infos, key = lambda x : x.image_name)
    test_cam_infos = [test_cam_infos[i-1] for i in test_cam_id]
    
    return {"train":train_cam_infos, "test":test_cam_infos}

sceneLoadTypeCallbacks = {
    "DyColmap": readDyColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DyNeRF": readDynerfSceneInfo,
    "HyperNeRF": readHypernerfSceneInfo,
    "MonoDyNeRF": readMonoDyNeRFSceneInfo,
    "IphoneData": readIphoneDataSceneInfo,
    "SelfMade": readSelfMadeSceneInfo,
    # "SelfRaMade": readSelfMadeSceneInfo,
    "RaftExhaustive": readRaftExhaustiveDataSceneInfo,
    # "SelfMadeNvidia": readSelfMadeNvidiaSceneInfo,
    "read_Nvidia_cam_info":read_Nvidia_cam_Info,
    "read_DyNeRF_cam_info":read_DyNeRF_cam_info,
}
