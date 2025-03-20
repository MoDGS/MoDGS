import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

# from asset import example_scene_name2inter_ids, blended_mvs_ids
# from dataset.database import BaseDatabase, ExampleDatabase
# from utils.base_utils import pose_inverse, transform_points_Rt
import numpy as np

def generate_spiral_poses(radii, height, n_rotations, n_poses):
    # 生成等间隔的角度
    angles = np.linspace(0, 2*np.pi*n_rotations, n_poses)
    
    # 计算每个点的位置
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = np.linspace(-height/2, height/2, n_poses)
    
    # 创建一个空的姿态矩阵
    poses = np.empty((n_poses, 4, 4))
    
    # 填充姿态矩阵
    for i in range(n_poses):
        # 创建一个单位矩阵
        poses[i] = np.eye(4)
        
        # 设置位置
        poses[i, :3, 3] = [x[i], y[i], z[i]]
        
        # 设置方向
        poses[i, :3, :3] = look_at([x[i], y[i], z[i]], [0, 0, 0])
        
    return poses

def look_at(position, target):
    # 计算前向向量
    forward = np.array(target) - np.array(position)
    forward /= np.linalg.norm(forward)
    
    # 计算右向量
    right = np.cross([0, 0, 1], forward)
    right /= np.linalg.norm(right)
    
    # 计算上向量
    up = np.cross(forward, right)
    
    # 创建旋转矩阵
    rotation = np.stack([right, up, forward])
    
    return rotation

def transform_points_Rt(pts, R, t):
    t = t.flatten()
    return pts @ R.T + t[None, :]
def pose_inverse(pose):
    R = pose[:, :3].T
    t = - R @ pose[:, 3:]
    return np.concatenate([R, t], -1)
def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec2, vec1_avg))
    vec1 = normalize(np.cross(vec0, vec2))
    m = np.stack([-vec0, vec1, vec2, pos], 1)
    return m
def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([-np.sin(theta), np.cos(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.])) - c)
        render_poses.append(np.concatenate([viewmatrix(z, up, c)], 1))
    return render_poses

def forward_circle_poses(cams):
    
    poses = [cam.view_world_transform.transpose(0, 1).numpy()[:3,:4] for cam in cams] ##cam.view_world_transform第四行是t，1形式，转换成0001形式的 C2W
    # poses_inv = [pose_inverse(pose) for pose in poses]
    
    
    cam_pts = np.asarray(poses)[:, :, 3]
    cam_rots = np.asarray(poses)[:, :, :3]
    down = cam_rots[:, :, 1]
    lookat = cam_rots[:, :, 2]

    avg_cam_pt = (np.max(cam_pts,0)+np.min(cam_pts,0))/2
    avg_down = np.mean(down,0)
    avg_lookat = np.mean(lookat,0)
    avg_pose_inv = viewmatrix(avg_lookat, avg_down, avg_cam_pt) ## 带有 inv的都是c2w?
    avg_pose = pose_inverse(avg_pose_inv)

    cam_pts_in_avg_pose = transform_points_Rt(cam_pts,avg_pose[:,:3],avg_pose[:,3]) # n,3
    range_in_avg_pose = np.percentile(np.abs(cam_pts_in_avg_pose), 90, 0)

    # depth_ranges = [cams.get_depth_range(img_id) for img_id in cams.get_img_ids()]
    
    depth_ranges = [cam.get_depth_range()   for cam in cams if cam.depth is not None]
    depth_ranges = np.asarray(depth_ranges)
    near, far = np.mean(depth_ranges[:,0]), np.mean(depth_ranges[:,1])
    dt = .75
    mean_dz = 1. / (((1. - dt) / near + dt / far))
    z_delta = near * 0.2
    range_in_avg_pose[2] = z_delta
    shrink_ratio = 0.8
    range_in_avg_pose*=shrink_ratio

    render_poses=render_path_spiral(avg_pose_inv,avg_down,range_in_avg_pose,mean_dz,0.,1,60)
    render_poses=[pose_inverse(pose) for pose in render_poses]
    render_poses=np.asarray(render_poses)
    return render_poses ## w2c？ N*3*4
def forward_circle_poses_for_staticCams(cams):
    
    poses = [cam.view_world_transform.transpose(0, 1).numpy()[:3,:4] for cam in cams] ##cam.view_world_transform第四行是t，1形式，转换成0001形式的 C2W
    # poses_inv = [pose_inverse(pose) for pose in poses]
    
    rad_predefined = 0.1
    pull_over_factor=1.0 ##FIXME: LQM pull over 让摄像机远离物体
    # cam_pts = np.asarray(poses)[:, :, 3]
    cam_pts = np.asarray(poses)[:, :, 3]*pull_over_factor ##FIXME: LQM pull over
    cam_rots = np.asarray(poses)[:, :, :3]
    
    down = cam_rots[:, :, 1]
    lookat = cam_rots[:, :, 2]

    avg_cam_pt = (np.max(cam_pts,0)+np.min(cam_pts,0))/2
    avg_down = np.mean(down,0)
    avg_lookat = np.mean(lookat,0)
    avg_pose_inv = viewmatrix(avg_lookat, avg_down, avg_cam_pt) ## 带有 inv的都是c2w?
    avg_pose = pose_inverse(avg_pose_inv)

    cam_pts_in_avg_pose = transform_points_Rt(cam_pts,avg_pose[:,:3],avg_pose[:,3]) # n,3
    range_in_avg_pose = np.percentile(np.abs(cam_pts_in_avg_pose), 90, 0)
    # range_in_avg_pose= np.ones_like(range_in_avg_pose)*0.2*np.abs(avg_cam_pt).max()
    # range_in_avg_pose= np.ones_like(range_in_avg_pose)*0.2*np.abs(avg_cam_pt).max()
    # range_in_avg_pose= np.ones_like(range_in_avg_pose)*0.03*np.abs(avg_cam_pt).max()
    range_in_avg_pose= np.ones_like(range_in_avg_pose)*0.015*np.abs(avg_cam_pt).max()

    print("avg_cam_pt",avg_cam_pt)
    print("range_in_avg_pose",range_in_avg_pose)

    # depth_ranges = [cams.get_depth_range(img_id) for img_id in cams.get_img_ids()]
    
    depth_ranges = [cam.get_depth_range()   for cam in cams if cam.depth is not None]
    depth_ranges = np.asarray(depth_ranges)
    near, far = np.mean(depth_ranges[:,0]), np.mean(depth_ranges[:,1])
    near, far = near*pull_over_factor, far*pull_over_factor#FIXME: LQM pull over
    dt = .75
    mean_dz = 1. / (((1. - dt) / near + dt / far))
    z_delta = near * 0.2
    range_in_avg_pose[2] = z_delta
    shrink_ratio = 0.8
    range_in_avg_pose*=shrink_ratio
    
    def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
        render_poses = []
        rads = np.array(list(rads) + [1.])
        print("function")
        # for theta in np.linspace(0., 0.5* np.pi, int(N/5))[:-1]:
        #     c = np.dot(c2w[:3, :4], np.array([0.0, np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        #     z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.])) - c)
        #     render_poses.append(np.concatenate([viewmatrix(z, up, c)], 1))
        for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
            c = np.dot(c2w[:3, :4], np.array([-np.sin(theta), np.cos(theta), -np.sin(theta * zrate), 1.]) * rads)
            z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.])) - c)
            render_poses.append(np.concatenate([viewmatrix(z, up, c)], 1))
        # for theta in np.linspace(0., 0.5* np.pi, int(N/5))[::-1]:
        #     c = np.dot(c2w[:3, :4], np.array([0.0, np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        #     z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.])) - c)
        #     render_poses.append(np.concatenate([viewmatrix(z, up, c)], 1))
        return render_poses


    render_poses=render_path_spiral(avg_pose_inv,avg_down,range_in_avg_pose,mean_dz,0.,1,len(cams))
    render_poses=[pose_inverse(pose) for pose in render_poses]
    render_poses=np.asarray(render_poses)
    return render_poses ## w2c？ N*3*4