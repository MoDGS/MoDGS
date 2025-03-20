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
import numpy as np
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from utils.general_utils import  build_rotation_basisGSmodel as build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.loss_utils import localsmoothness_loss

class IsotropicGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, L: int):
        self.active_sh_degree = 0
        self.L = L
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation,
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        time_length = self._xyz.shape[1]
        # self._xyz = nn.Parameter(model_args[1].repeat(1, time_length, 1).requires_grad_(True))
        # self._rotation = nn.Parameter(model_args[5].repeat(1, time_length, 1).requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling.repeat(1,3))
    
    @property
    def get_rotation(self):
        # return self._rotation
        return self.rotation_activation(self._rotation, dim=-1)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[:, 0, :])).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 1)
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], fused_point_cloud.shape[1], 4), device="cuda")
        rots[:, 0, 0] = 1
        # rots[:, 3, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        ## Spatial learning rate scaling, for the optimizer FIXME,
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = []
        for t in range(self._xyz.shape[1]):
            l.extend([f'x{t:03}', f'y{t:03}', f'z{t:03}'])
        l.extend(['nx', 'ny', 'nz'])
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.get_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[-1]):
            for t in range(self._rotation.shape[-2]):
                l.append(f'rot_{t:03}_{i}')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().flatten(start_dim=1).cpu().numpy()
        normals = np.zeros((xyz.shape[0], 3))
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self.get_scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().flatten(start_dim=1).cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        x_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("x")]
        y_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("y")]
        z_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("z")]
        x_names = sorted(x_names, key = lambda x: int(x.replace('x', '')))
        y_names = sorted(y_names, key = lambda y: int(y.replace('y', '')))
        z_names = sorted(z_names, key = lambda z: int(z.replace('z', '')))
        assert len(x_names) == len(y_names) == len(z_names)
        x = np.zeros((opacities.shape[0], len(x_names)))
        y = np.zeros((opacities.shape[0], len(y_names)))
        z = np.zeros((opacities.shape[0], len(z_names)))
        for idx, attr_name in enumerate(x_names):
            x[:, idx] = np.asarray(plydata.elements[0][attr_name])
        for idx, attr_name in enumerate(y_names):
            y[:, idx] = np.asarray(plydata.elements[0][attr_name])
        for idx, attr_name in enumerate(z_names):
            z[:, idx] = np.asarray(plydata.elements[0][attr_name])
        xyz = np.stack((x, y, z),  axis=-1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: (int(x.split('_')[-1]), int(x.split('_')[-2])))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rots = rots.reshape(xyz.shape[0], -1, 4)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()### 因为前面是clone。所以这一步，只去前面的点，不取后面没有被克隆的点。
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) ## 大于一定的scale

        # print(grads.shape)
        # print(selected_pts_mask.shape)
        stds = self.get_scaling[selected_pts_mask].repeat(N*self.get_xyz.shape[1],1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1,1).reshape(-1, 3, 3)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1).unsqueeze(1).reshape(-1, self.get_xyz.shape[1], 3) + self.get_xyz[selected_pts_mask].repeat(N, 1, 1)
        new_xyz[:, 1:, :] = self.get_xyz[selected_pts_mask].repeat(N, 1, 1)[:, 1:, :]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) ## 梯度大于阈值。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)## scale小于一定尺度。 ## 这里的dim1是什么？
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze() ## opacity 过小
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size ## 屏幕空间的点过大。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent ## 世界坐标系下面过大
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
class SeperateRepreGaussianModel:
    """different representation for xyz and xyz_coefficient, and different learning rate for them
    """
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, L: int):
        self.active_sh_degree = 0
        self.L = L
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._xyz_coefficient = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self.get_xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        xyz,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation,
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self._xyz=xyz[:,:1,:]
        self._xyz_coefficient=xyz[:,1:,:]
        time_length = self._xyz.shape[1]
        # self._xyz = nn.Parameter(model_args[1].repeat(1, time_length, 1).requires_grad_(True))
        # self._rotation = nn.Parameter(model_args[5].repeat(1, time_length, 1).requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        # return self._rotation
        return self.rotation_activation(self._rotation, dim=-1)
    
    @property
    def get_xyz(self):
        return torch.cat([self._xyz,self._xyz_coefficient],dim=1)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        
        fused_point_cloud_xyz = torch.tensor(np.asarray(pcd.points[:,:1,:])).float().cuda()
        fused_point_cloud_xyz_coefficient = torch.tensor(np.asarray(pcd.points[:,1:,:])).float().cuda()
        fused_point_cloud = torch.cat([fused_point_cloud_xyz,fused_point_cloud_xyz_coefficient],dim=1)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud_xyz.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[:, 0, :])).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], fused_point_cloud.shape[1], 4), device="cuda")
        rots[:, 0, 0] = 1
        # rots[:, 3, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud_xyz.requires_grad_(True))
        self._xyz_coefficient = nn.Parameter(fused_point_cloud_xyz_coefficient.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._xyz_coefficient], 'lr': training_args.position_coeff_lr_init * self.spatial_lr_scale, "name": "xyz_coefficient"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.xyz_coefficient_scheduler_args = get_expon_lr_func(lr_init=training_args.position_coeff_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_coeff_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_coeff_lr_delay_mult,
                                                    max_steps=training_args.position_coeff_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        current_lr={}
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                current_lr["xyz_lr"]=param_group['lr']
                param_group['lr'] = lr
                # return lr
            if param_group["name"] == "xyz_coefficient":
                lr = self.xyz_coefficient_scheduler_args(iteration)
                current_lr["xyz_coeff_lr"]=param_group['lr']
                param_group['lr'] = lr
                # return lr
        return  current_lr
    def construct_list_of_attributes(self):
        l = []
        for t in range(self.get_xyz.shape[1]):
            l.extend([f'x{t:03}', f'y{t:03}', f'z{t:03}'])
        l.extend(['nx', 'ny', 'nz'])
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[-1]):
            for t in range(self._rotation.shape[-2]):
                l.append(f'rot_{t:03}_{i}')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().flatten(start_dim=1).cpu().numpy()
        normals = np.zeros((xyz.shape[0], 3))
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().flatten(start_dim=1).cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        # raise NotImplementedError
        plydata = PlyData.read(path)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        x_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("x")]
        y_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("y")]
        z_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("z")]
        x_names = sorted(x_names, key = lambda x: int(x.replace('x', '')))
        y_names = sorted(y_names, key = lambda y: int(y.replace('y', '')))
        z_names = sorted(z_names, key = lambda z: int(z.replace('z', '')))
        assert len(x_names) == len(y_names) == len(z_names)
        x = np.zeros((opacities.shape[0], len(x_names)))
        y = np.zeros((opacities.shape[0], len(y_names)))
        z = np.zeros((opacities.shape[0], len(z_names)))
        for idx, attr_name in enumerate(x_names):
            x[:, idx] = np.asarray(plydata.elements[0][attr_name])
        for idx, attr_name in enumerate(y_names):
            y[:, idx] = np.asarray(plydata.elements[0][attr_name])
        for idx, attr_name in enumerate(z_names):
            z[:, idx] = np.asarray(plydata.elements[0][attr_name])
        xyz = np.stack((x, y, z),  axis=-1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: (int(x.split('_')[-1]), int(x.split('_')[-2])))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rots = rots.reshape(xyz.shape[0], -1, 4)

        self._xyz = nn.Parameter(torch.tensor(xyz[:,:1,:], dtype=torch.float, device="cuda").requires_grad_(True))
        self._xyz_coefficient = nn.Parameter(torch.tensor(xyz[:,1:,:], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._xyz_coefficient=optimizable_tensors["xyz_coefficient"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_xyz_coefficient,new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "xyz_coefficient":new_xyz_coefficient,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._xyz_coefficient = optimizable_tensors["xyz_coefficient"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()### 因为前面是clone。所以这一步，只去前面的点，不取后面没有被克隆的点。
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) ## 大于一定的scale

        # print(grads.shape)
        # print(selected_pts_mask.shape)
        stds = self.get_scaling[selected_pts_mask].repeat(N*self.get_xyz.shape[1],1)## calculate stds even for coefficient? then sample new coefficient from the normal distribution?
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1,1).reshape(-1, 3, 3) 

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1).unsqueeze(1).reshape(-1, self.get_xyz.shape[1], 3) + self.get_xyz[selected_pts_mask].repeat(N, 1, 1)
        new_xyz=new_xyz[:, :1, :]
        new_xyz_coefficient= self.get_xyz[selected_pts_mask].repeat(N, 1, 1)[:, 1:, :]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_xyz_coefficient, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) ## 梯度大于阈值。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)## scale小于一定尺度。 ## 这里的dim1是什么？
        
        new_xyz = self._xyz[selected_pts_mask]
        new_xyz_coefficient = self._xyz_coefficient[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_xyz_coefficient,new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze() ## opacity 过小
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size ## 屏幕空间的点过大。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent ## 世界坐标系下面过大
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1



class SeperateRepreIsotropicGaussianModel:
    """different representation for xyz and xyz_coefficient, and different learning rate for them
        above this, We use isotropic gaussian model, which fix Rotation and Scaling.
    2024年1月10日14:03:23: 加入 global transform model和 local transfrom model。
    """
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, L: int, L_global : int=-1):
        self.has_global_feature=False
        self.active_sh_degree = 0
        self.L = L
        self.L_global = L_global
        if  self.L_global >0:
            self.global_xyz_coefficient = torch.empty(0)
            self.has_global_feature=True
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._xyz_coefficient = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._rotation_coefficient = torch.empty(0)
        self._rotation_identical=torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
    def get_localsmoothnessloss(self,time1,time2,neighbor_k=10,Sampe_query_pcd=False):
        
        kw = self._Gaussians_step(time1,time2)
        pcd = kw["pcd"]
        flow = kw["flow"]
        if not Sampe_query_pcd:
            query_pcd = pcd 
        else:
            raise NotImplementedError
        loss =localsmoothness_loss(query_pcd,pcd,flow,neighbor_k)
        return loss

    def _Gaussians_step(self,time1,time2):
        L = self.L
        basis = 2**torch.arange(0, L, device='cuda').repeat_interleave(2)*math.pi*time1
        basis[::2] = torch.sin(basis[::2])
        basis[1::2] = torch.cos(basis[1::2])
        basis_global=None
        if hasattr(self,"L_global") and self.L_global>0:
                basis_global = 2**torch.arange(0, self.L_global, device='cuda').repeat_interleave(2)*math.pi*time1
                basis_global[::2] = torch.sin(basis_global[::2])
                basis_global[1::2] = torch.cos(basis_global[1::2])
        means3D,_=self.step(basis,basis_global)
   
        assert time2>=0.0 and time2<=1.0,"time2 should be in [0,1]"
        basis2 = 2**torch.arange(0, L, device='cuda').repeat_interleave(2)*math.pi*time2
        basis2[::2] = torch.sin(basis2[::2])
        basis2[1::2] = torch.cos(basis2[1::2])
        basis_global2=None
        if hasattr(self,"L_global") and self.L_global>0:
                basis_global2 = 2**torch.arange(0, self.L_global, device='cuda').repeat_interleave(2)*math.pi*time2
                basis_global2[::2] = torch.sin(basis_global2[::2])
                basis_global2[1::2] = torch.cos(basis_global2[1::2])
        means3D2,_=self.step(basis2,basis_global2)

        # cur_means3d >??
        flow=means3D2-means3D



        # print(rendered_image.mean(dim=(1,2)))    
        return {"pcd":means3D,
                "flow": flow}
        
    def capture(self):
        return (
            self.global_xyz_coefficient if self.has_global_feature else None,
            self.active_sh_degree,
            self.get_xyz,
            self._features_dc,
            self._features_rest,
            self.get_scaling_noact,
            self._rotation_identical,
            self._rotation_coefficient,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        if self.has_global_feature:
            (self.global_xyz_coefficient,
            self.active_sh_degree, 
            xyz,
            self._features_dc, 
            self._features_rest,
            _scaling, 
            self._rotation_identical, 
            self._rotation_coefficient,
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
        else:
            (self.active_sh_degree, 
            xyz,
            self._features_dc, 
            self._features_rest,
            _scaling, 
            self._rotation_identical, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
        self._xyz=xyz[:,:1,:]
        self._xyz_coefficient=xyz[:,1:,:]
        time_length = self._xyz.shape[1]
        # self._xyz = nn.Parameter(model_args[1].repeat(1, time_length, 1).requires_grad_(True))
        # self._rotation = nn.Parameter(model_args[5].repeat(1, time_length, 1).requires_grad_(True))
        self._scaling=_scaling[:,:1] ## TODO: 这里的scaling是不是明明是一个值，但是我还是存了三个。
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling.repeat(1,3))
    @property
    def get_scaling_noact(self):
        return self._scaling.repeat(1,3)
    @property
    def get_rotation(self):## FIXME 
        # return self._rotation
        return self.rotation_activation(torch.cat([self._rotation_identical,self._rotation_coefficient],dim=1), dim=-1)## TODO：
    @property
    def get_rotation_noact(self):
        return torch.cat([self._rotation_identical,self._rotation_coefficient],dim=1) ## (N,1,4) and (N,1,4) --> N,2,4
    @property
    def get_xyz(self):
        return torch.cat([self._xyz,self._xyz_coefficient],dim=1)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def step(self,basis,global_basis=None):
    
        local_movement=(self.get_xyz[:, 1:2*self.L+1, :]*basis.unsqueeze(-1)).sum(1)
        next_pos = self.get_xyz[:, 0, :]+local_movement
        # movement=0.0
        movement=local_movement
        if global_basis is not None and self.L_global != -1:
            global_movement=(self.global_xyz_coefficient*global_basis.unsqueeze(-1)).sum(1)
            # print(global_movement)
            next_pos = next_pos+global_movement
            movement=movement+global_movement
        return next_pos,movement
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation_noact)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        
        fused_point_cloud_xyz = torch.tensor(np.asarray(pcd.points[:,:1,:])).float().cuda()
        # fused_point_cloud_xyz_coefficient = torch.tensor(np.asarray(pcd.points[:,1:,:])).float().cuda()
        ## make it adaptive for any number of coefficient level.
        fused_point_cloud_xyz_coefficient = torch.zeros((fused_point_cloud_xyz.shape[0], 2*self.L, 3)).float().cuda()
        fused_point_cloud = torch.cat([fused_point_cloud_xyz,fused_point_cloud_xyz_coefficient],dim=1)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud_xyz.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[:, 0, :])).float().cuda()), 0.0000001)
        ## for zju mocap only 
        ## FIXME :
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[:, 0, :])).float().cuda()), 0.0000003)
        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[:, 0, :])).float().cuda()), 0.0003) ## MonoDynerf
        
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 1)
        rots = torch.zeros((fused_point_cloud.shape[0], 1, 4), device="cuda")
        rots[:, 0, 0] = 1
        rots_coefficient = torch.zeros((fused_point_cloud.shape[0], fused_point_cloud.shape[1]-1, 4), device="cuda")
    
        # rots[:, 3, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        if self.L_global>0:
            self.global_xyz_coefficient=nn.Parameter(torch.zeros((1,2*self.L_global, 3), dtype=torch.float, device="cuda").requires_grad_(True))### positon global coefficient.

        self._xyz = nn.Parameter(fused_point_cloud_xyz.requires_grad_(True))
        self._xyz_coefficient = nn.Parameter(fused_point_cloud_xyz_coefficient.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation_identical = nn.Parameter(rots.requires_grad_(False))
        self._rotation_coefficient = nn.Parameter(rots_coefficient.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._xyz_coefficient], 'lr': training_args.position_coeff_lr_init, "name": "xyz_coefficient"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            {'params': [self._rotation_coefficient], 'lr': training_args.rotation_coeff_lr, "name": "rotation_coefficient"},
            # {'params': [self.global_xyz_coefficient], 'lr': training_args.global_coeff_lr, "name": "global_xyz_coefficient"}
        ]
        if self.L_global>0:
           
            l.append(  {'params': [self.global_xyz_coefficient], 'lr': training_args.global_coeff_lr, "name": "global_xyz_coefficient"}  )
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.xyz_coefficient_scheduler_args = get_expon_lr_func(lr_init=training_args.position_coeff_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_coeff_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_coeff_lr_delay_mult,
                                                    max_steps=training_args.position_coeff_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        current_lr={}
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                current_lr["xyz_lr"]=param_group['lr']
                param_group['lr'] = lr
                # return lr
            if param_group["name"] == "xyz_coefficient":
                lr = self.xyz_coefficient_scheduler_args(iteration)
                current_lr["xyz_coeff_lr"]=param_group['lr']
                param_group['lr'] = lr
                # return lr
        return  current_lr
    def construct_list_of_attributes(self):
        l = []
        for t in range(self.get_xyz.shape[1]):
            l.extend([f'x{t:03}', f'y{t:03}', f'z{t:03}'])
        l.extend(['nx', 'ny', 'nz'])
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.get_scaling_noact.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.get_rotation_noact.shape[-1]):
            for t in range(self.get_rotation_noact.shape[-2]):
                l.append(f'rot_{t:03}_{i}')
        return l
    def save_global_feature(self, path):
        assert self.has_global_feature and self.L_global>0,"no global feature"
        mkdir_p(os.path.dirname(path))
        global_xyz_coefficient = self.global_xyz_coefficient.detach().cpu().numpy()
        np.save(path,global_xyz_coefficient)
    def load_global_feature(self, path):
        saved_global_feature = np.load(path)  
        self.global_xyz_coefficient=nn.Parameter(torch.tensor(saved_global_feature, dtype=torch.float, device="cuda").requires_grad_(True))
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().flatten(start_dim=1).cpu().numpy()
        normals = np.zeros((xyz.shape[0], 3))
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self.get_scaling_noact.detach().cpu().numpy()
        rotation = self.get_rotation_noact.detach().flatten(start_dim=1).cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        # raise NotImplementedError
        plydata = PlyData.read(path)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        x_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("x")]
        y_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("y")]
        z_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("z")]
        x_names = sorted(x_names, key = lambda x: int(x.replace('x', '')))
        y_names = sorted(y_names, key = lambda y: int(y.replace('y', '')))
        z_names = sorted(z_names, key = lambda z: int(z.replace('z', '')))
        assert len(x_names) == len(y_names) == len(z_names)
        x = np.zeros((opacities.shape[0], len(x_names)))
        y = np.zeros((opacities.shape[0], len(y_names)))
        z = np.zeros((opacities.shape[0], len(z_names)))
        for idx, attr_name in enumerate(x_names):
            x[:, idx] = np.asarray(plydata.elements[0][attr_name])
        for idx, attr_name in enumerate(y_names):
            y[:, idx] = np.asarray(plydata.elements[0][attr_name])
        for idx, attr_name in enumerate(z_names):
            z[:, idx] = np.asarray(plydata.elements[0][attr_name])
        xyz = np.stack((x, y, z),  axis=-1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: (int(x.split('_')[-1]), int(x.split('_')[-2])))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rots = rots.reshape(xyz.shape[0], -1, 4)

        self._xyz = nn.Parameter(torch.tensor(xyz[:,:1,:], dtype=torch.float, device="cuda").requires_grad_(True))
        self._xyz_coefficient = nn.Parameter(torch.tensor(xyz[:,1:,:], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[:,:1], dtype=torch.float, device="cuda").requires_grad_(True)) #### in isotropic model, scale is 1D ,aussing they are the same.
        self._rotation_identical = nn.Parameter(torch.tensor(rots[:,:1,:], dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation_coefficient = nn.Parameter(torch.tensor(rots[:,1:,:], dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if group["name"] == "global_xyz_coefficient":
                continue
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._xyz_coefficient=optimizable_tensors["xyz_coefficient"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation_coefficient = optimizable_tensors["rotation_coefficient"]
        N=self._rotation_coefficient.shape[0]
        _rotation_identical=torch.zeros([N,1,4]).to(self._rotation_coefficient)
        _rotation_identical[:,0,0]=1
        self._rotation_identical=nn.Parameter(_rotation_identical).requires_grad_(False)
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == "global_xyz_coefficient":
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_xyz_coefficient,new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation_coefficient):
        d = {"xyz": new_xyz,
             "xyz_coefficient":new_xyz_coefficient,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation_coefficient" : new_rotation_coefficient}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._xyz_coefficient = optimizable_tensors["xyz_coefficient"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation_coefficient = optimizable_tensors["rotation_coefficient"]
        N=self._rotation_coefficient.shape[0]
        _rotation_identical=torch.zeros([N,1,4]).to(self._rotation_coefficient)
        _rotation_identical[:,0,0]=1
        self._rotation_identical=nn.Parameter(_rotation_identical).requires_grad_(False)
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()### 因为前面是clone。所以这一步，只去前面的点，不取后面没有被克隆的点。
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) ## 大于一定的scale

        # print(grads.shape)
        # print(selected_pts_mask.shape)
        stds = self.get_scaling[selected_pts_mask].repeat(N*self.get_xyz.shape[1],1) ## TODO：这里的代码有问题是吧？后面的coefficient也参与了计算？
        means = torch.zeros((stds.size(0), 3),device="cuda")    
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation_noact[selected_pts_mask]).repeat(N,1,1,1).reshape(-1, 3, 3)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1).unsqueeze(1).reshape(-1, self.get_xyz.shape[1], 3) + self.get_xyz[selected_pts_mask].repeat(N, 1, 1) 
        new_xyz=new_xyz[:, :1, :] ##TODO： 我这里似乎没有对new_coefficient采样。而是直接copy的以前的轨迹。
        new_xyz_coefficient= self.get_xyz[selected_pts_mask].repeat(N, 1, 1)[:, 1:, :]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))[:,:1] ## asuming they are the same
        new_rotation_coefficient = self._rotation_coefficient[selected_pts_mask].repeat(N,1,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_xyz_coefficient, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation_coefficient)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) ## 梯度大于阈值。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)## scale小于一定尺度。 ## 这里的dim1是什么？
        
        new_xyz = self._xyz[selected_pts_mask]
        new_xyz_coefficient = self._xyz_coefficient[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation_coefficient = self._rotation_coefficient[selected_pts_mask]

        self.densification_postfix(new_xyz, new_xyz_coefficient,new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation_coefficient)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze() ## opacity 过小
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size ## 屏幕空间的点过大。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent ## 世界坐标系下面过大
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    def prune(self,min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze() ## opacity 过小
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size ## 屏幕空间的点过大。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent ## 世界坐标系下面过大
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
