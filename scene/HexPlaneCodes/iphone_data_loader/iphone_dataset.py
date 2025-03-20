from builtins import breakpoint
from lib2to3.pgen2.driver import load_grammar
import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
from torch.utils.data import Dataset
import math
import numpy as np
import torch
from PIL import Image
from .util import triangulate, points_bound, points_centroid, points_bounding_size

def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


class Iphone_dataset(Dataset):
    def __init__(self, 
                 datadir,
                 split="train",
                 ratio=2.0,
                 use_bg_points=False,
                 cal_fine_bbox=False,
                 add_cam=False,
                 is_stack=False,
                 time_scale=1.0,
                 bbox=1.5,
                 N_random_pose=120):
        self.img_wh = (int(720 / ratio), int(960 / ratio))
        
        self.white_bg = False
        self.time_scale = time_scale
        self.is_stack = is_stack
        self.N_random_pose = N_random_pose
        from .camera import Camera
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)
        with open(f'{datadir}/extra.json', 'r') as f:
            extra_json = json.load(f)


        self.near = scene_json['near']
        self.far = scene_json['far']
        self.near_far = [self.near, self.far]
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']
        self.bbox = np.array(extra_json['bbox'], dtype=np.float32)
        self.scene_bbox = torch.tensor(self.bbox)
        self._lookat = np.array(extra_json["lookat"], dtype=np.float32)
        self._up = np.array(extra_json["up"], dtype=np.float32)


        self.split = split
        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']
        if len(self.val_id) == 0:
            self.i_train = []
            self.i_test = []
            for i in range(len(self.all_img)):
                if i % 4 == 3:
                    self.i_test.append(self.all_img[i])
                elif i % 4 == 1:
                    self.i_train.append(self.all_img[i])
                else:
                    pass 
        else:
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(id)
                if id in self.train_id:
                    self.i_train.append(id)
        
        print('self.i_train',self.i_train)
        print('self.i_test',self.i_test)
        if self.split == "train":
            self.all_img_idx = self.i_train
        else:
            self.all_img_idx = self.i_test
        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img_idx]
        max_time = max([meta_json[i]['warp_id'] for i in self.all_img])
        self.all_time = [meta_json[i]['warp_id']/max_time for i in self.all_img_idx]

        self.appearance_id = [meta_json[i]['appearance_id'] for i in self.all_img_idx]
        self.warp_id = [meta_json[i]['warp_id'] for i in self.all_img_idx]
        self.camera_id = [meta_json[i]['camera_id'] for i in self.all_img_idx]

        self.selected_time = set(self.all_time)
        self.ratio = ratio
        # all poses
        self.all_cam_params = []
        for im in self.all_img_idx:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            camera = camera.scale(1/ratio)
            camera.position = camera.position - self.scene_center
            camera.position = camera.position * self.coord_scale
            self.all_cam_params.append(camera)

        self.all_img = [f'{datadir}/rgb/{int(ratio)}x/{i}.png' for i in self.all_img_idx]
        self.all_depth = [f'{datadir}/depth/{int(ratio)}x/{i}.npy' for i in self.all_img_idx]
        self.h, self.w = self.all_cam_params[0].image_shape

        self.use_bg_points = use_bg_points
        if use_bg_points:
            with open(f'{datadir}/points.npy', 'rb') as f:
                points = np.load(f)
            self.bg_points = (points - self.scene_center) * self.coord_scale
            self.bg_points = torch.tensor(self.bg_points).float()
        print(f'total {len(self.all_img)} images ',
                'use bg_point=',self.use_bg_points)
        self.load_meta()
        self.add_cam = add_cam
        # if split == "train":
        self.init_image_path()

    def init_image_path(self):
        camera = self.all_cam_params
        origins = torch.Tensor([c.position for c in camera])
        directions = torch.Tensor([c.optical_axis for c in camera])
        weights = torch.ones_like(directions[:, 0])
        look_at = triangulate(origins, origins + directions, weights)
        look_at = look_at.numpy()
        origins = origins.numpy()
        directions = directions.numpy()
        weights = weights.numpy()
        print('look_at', look_at)
        avg_position = np.mean(origins, axis=0)
        print('avg_position', avg_position)
        up = -np.mean([c.orientation[..., 1] for c in camera], axis=0)
        print('up', up)
        bounding_size = points_bounding_size(origins) / 2
        x_scale = 1.0 
        y_scale = 1.0
        xs = x_scale * bounding_size
        ys = y_scale * bounding_size
        radius = 0.75

        ref_camera = camera[0]
        print(ref_camera.position)
        z_offset = -0.1

        angles = np.random.rand(self.N_random_pose) * 2 * math.pi
        positions = []
        for angle in angles:
            x = np.cos(angle) * radius * xs
            y = np.sin(angle) * radius * ys
            position = np.array([x, y, z_offset])
            # Make distance to reference point constant.
            position = avg_position + position
            positions.append(position)

        positions = np.stack(positions)
        random_rays = []
        for position in positions:
            camera = ref_camera.look_at(position, look_at, up)
            pixels = camera.get_pixel_centers()
            rays_dir = torch.tensor(camera.pixels_to_rays(pixels)).float().view([-1,3])
            rays_ori = torch.tensor(camera.position[None, :]).float().expand_as(rays_dir)
            random_rays += [torch.cat([rays_ori, rays_dir], 1)]
        self.image_path_rays = torch.stack(random_rays,0).reshape(-1,*self.img_wh[::-1], 6)
        random_times = self.time_scale * torch.arange(self.N_random_pose) / (self.N_random_pose - 1) * 2.0 - 1.0
        self.image_path_time = random_times
        
        # calculate the average pose
        average_camera = ref_camera.look_at(avg_position, look_at, up)
        pixels = average_camera.get_pixel_centers()
        rays_dir = torch.tensor(average_camera.pixels_to_rays(pixels)).float().view([-1,3])
        rays_ori = torch.tensor(average_camera.position[None, :]).float().expand_as(rays_dir)
        self.avg_pose_rays = torch.cat([rays_ori, rays_dir], 1)
        self.avg_pose_times = torch.arange(100) / (100 - 1) * 2.0 - 1.0

    def load_meta(self):
        all_rgbs, all_rays, all_times, all_cams =[], [], [], []
        all_depths = []
        for index in range(len(self.all_img)):
            rays_dir, rays_ori, rays_color, cosa = self.load_raw(index)
            cur_time = torch.ones_like(rays_dir[:, 0:1]) * self.all_time[index]
            camera_idx = torch.ones_like(rays_dir[:, 0:1]) * self.all_cam[index]
            if self.split == "train":
                depth = np.load(self.all_depth[index]) * self.coord_scale
                depth = depth / cosa
                if not self.is_stack:
                    depth = torch.tensor(depth).view(-1, 1)
                else:
                    depth = torch.tensor(depth)
            else:
                depth = torch.zeros_like(rays_dir[:, 0:1])
            all_rgbs.append(rays_color)
            all_rays += [torch.cat([rays_ori, rays_dir], 1)]
            all_times += [cur_time]
            all_cams += [camera_idx]
            all_depths += [depth]
        
        if not self.is_stack:
            all_rgbs = torch.cat(all_rgbs, 0)
            all_rays = torch.cat(all_rays, 0)
            all_times = torch.cat(all_times, 0)
            all_cams = torch.cat(all_cams, 0)
            all_depths = torch.cat(all_depths, 0)
        
        else:
            all_rays = torch.stack(all_rays, 0)
            all_rgbs = torch.stack(all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)
            all_times = torch.stack(all_times, 0)
            all_cams = torch.stack(all_cams, 0)
            all_depths = torch.stack(all_depths, 0)
        
        all_times = self.time_scale * (all_times * 2.0 - 1.0)

        self.all_rgbs = all_rgbs
        self.all_rays = all_rays
        self.all_times = all_times
        self.all_cams = all_cams
        self.all_depths = all_depths

    def load_raw(self, idx):
        image = Image.open(self.all_img[idx]).convert("RGB")
        camera = self.all_cam_params[idx]
        pixels = camera.get_pixel_centers()
        rays_dir = torch.tensor(camera.pixels_to_rays(pixels)).float().view([-1,3])
        rays_ori = torch.tensor(camera.position[None, :]).float().expand_as(rays_dir)
        rays_color = torch.tensor(np.array(image)).view([-1,3])/255.
        cosa = camera.pixels_to_cosa(camera.get_pixel_centers())
        return rays_dir, rays_ori, rays_color, cosa

    def __len__(self):
        if self.split == "train":
            return self.all_rays.shape[0]
        else:
            return self.all_rgbs.shape[0]

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'time': self.all_times[idx],
                      'depth': self.all_depths[idx]}
        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            time = self.all_times[idx]
            sample = {'rays': rays,
                      'rgbs': img,
                      'time': time}
        if self.use_bg_points:
            sample.update({"bg_points", self.bg_points})
        if self.add_cam:
            sample.update({"cam":self.all_cams[idx]})
        return sample

    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0., 1., render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times
