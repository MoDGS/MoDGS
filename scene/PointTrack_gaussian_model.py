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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from typing import Union
from dataclasses import dataclass
import json
@dataclass
class Gaussian:
    """_summary_
            ## copied from the gaussian-splatting-lightning repo
            https://github.com/yzslab/gaussian-splatting-lightning/blob/main/internal/models/gaussian_model_simplified.py
    Args:
        Returns:
            _type_: _description_
        """
    sh_degrees: int
    xyz: Union[np.ndarray, torch.Tensor]  # [n, 3]
    opacities: Union[np.ndarray, torch.Tensor]  # [n, 1]
    features_dc: Union[np.ndarray, torch.Tensor]  # [n, 3, 1], or [n, 1, 3]
    features_extra: Union[np.ndarray, torch.Tensor]  # [n, 3, 15], or [n, 15, 3]
    scales: Union[np.ndarray, torch.Tensor]  # [n, 3]
    rotations: Union[np.ndarray, torch.Tensor]  # [n, 4]

    @classmethod
    def load_from_ply(cls, path: str, sh_degrees: int):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (sh_degrees + 1) ** 2 - 3  # TODO: remove such a assertion
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_degrees + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return cls(
            sh_degrees=sh_degrees,
            xyz=xyz,
            opacities=opacities,
            features_dc=features_dc,
            features_extra=features_extra,
            scales=scales,
            rotations=rots,
        )

    @classmethod
    def load_from_state_dict(cls, sh_degrees: int, state_dict: dict, key_prefix: str = "gaussian_model._"):
        init_args = {
            "sh_degrees": sh_degrees,
        }
        for name_in_dict, name_in_dataclass in [
            ("xyz", "xyz"),
            ("features_dc", "features_dc"),
            ("features_rest", "features_extra"),
            ("scaling", "scales"),
            ("rotation", "rotations"),
            ("opacity", "opacities"),
        ]:
            init_args[name_in_dataclass] = state_dict["{}{}".format(key_prefix, name_in_dict)]

        return cls(**init_args)

    def to_parameter_structure(self):
        assert isinstance(self.xyz, np.ndarray) is True
        return Gaussian(
            sh_degrees=self.sh_degrees,
            xyz=torch.tensor(self.xyz, dtype=torch.float),
            opacities=torch.tensor(self.opacities, dtype=torch.float),
            features_dc=torch.tensor(self.features_dc, dtype=torch.float).transpose(1, 2),
            features_extra=torch.tensor(self.features_extra, dtype=torch.float).transpose(1, 2),
            scales=torch.tensor(self.scales, dtype=torch.float),
            rotations=torch.tensor(self.rotations, dtype=torch.float),
        )

    def to_ply_format(self):
        assert isinstance(self.xyz, torch.Tensor) is True
        return self.__class__(
            sh_degrees=self.sh_degrees,
            xyz=self.xyz.cpu().numpy(),
            opacities=self.opacities.cpu().numpy(),
            features_dc=self.features_dc.transpose(1, 2).cpu().numpy(),
            features_extra=self.features_extra.transpose(1, 2).cpu().numpy(),
            scales=self.scales.cpu().numpy(),
            rotations=self.rotations.cpu().numpy(),
        )

    def save_to_ply(self, path: str):
        assert isinstance(self.xyz, np.ndarray) is True

        gaussian = self

        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = gaussian.xyz
        normals = np.zeros_like(xyz)
        f_dc = gaussian.features_dc.reshape((gaussian.features_dc.shape[0], -1))
        # TODO: change sh degree
        if gaussian.sh_degrees > 0:
            f_rest = gaussian.features_extra.reshape((gaussian.features_extra.shape[0], -1))
        else:
            f_rest = np.zeros((f_dc.shape[0], 0))
        opacities = gaussian.opacities
        scale = gaussian.scales
        rotation = gaussian.rotations

        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the 3 DC
            for i in range(gaussian.features_dc.shape[1] * gaussian.features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            if gaussian.sh_degrees > 0:
                for i in range(gaussian.features_extra.shape[1] * gaussian.features_extra.shape[2]):
                    l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(gaussian.scales.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(gaussian.rotations.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
class GaussianModel_light:
    # class GaussianModelSimplified(nn.Module):
    """_summary_
            ## mainly brought from the gaussian-splatting-lightning repo
            https://github.com/yzslab/gaussian-splatting-lightning/blob/main/internal/models/gaussian_model_simplified.py
    Args:
        Returns:
            _type_: _description_
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
    def __init__(
            self,
            xyz: torch.Tensor,
            features_dc: torch.Tensor,
            features_rest: torch.Tensor,
            scaling: torch.Tensor,
            rotation: torch.Tensor,
            opacity: torch.Tensor,
            sh_degree: int,
            device=torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.setup_functions()
        self._xyz =nn.Parameter( xyz.to(device).requires_grad_(True))
        # self._features_dc = features_dc
        # self._features_rest = features_rest
        self._scaling = nn.Parameter(torch.exp(scaling).to(device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.nn.functional.normalize(rotation).to(device).requires_grad_(True))
        self._opacity = nn.Parameter(torch.sigmoid(opacity).to(device).requires_grad_(True))

        self._features = nn.Parameter(torch.cat([features_dc, features_rest], dim=1).to(device).contiguous().requires_grad_(True))

        self._opacity_origin = None

        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree

    def to_device(self, device):
        self._xyz = self._xyz.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        self._features = self._features.to(device)
        return self
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @classmethod
    def construct_from_state_dict(cls, state_dict, active_sh_degree, device):
        init_args = {
            "sh_degree": active_sh_degree,
            "device": device,
        }
        for i in state_dict:
            if i.startswith("gaussian_model._") is False:
                continue
            init_args[i[len("gaussian_model._"):]] = state_dict[i]
        return cls(**init_args)

    @classmethod
    def construct_from_ply(cls, ply_path: str, sh_degree, device):
        gaussians = Gaussian.load_from_ply(ply_path, sh_degree).to_parameter_structure()
        return cls(
            sh_degree=sh_degree,
            device=device,
            xyz=gaussians.xyz,
            opacity=gaussians.opacities,
            features_dc=gaussians.features_dc,
            features_rest=gaussians.features_extra,
            scaling=gaussians.scales,
            rotation=gaussians.rotations,
        )
    @property
    def get_scaling(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity

    def select(self, mask: torch.tensor):
        if self._opacity_origin is None:
            self._opacity_origin = torch.clone(self._opacity)  # make a backup
        else:
            self._opacity = torch.clone(self._opacity_origin)

        self._opacity[mask] = 0.


    def to_parameter_structure(self) -> Gaussian:
        xyz = self._xyz.cpu()
        features_dc = self._features[:, :1, :].cpu()
        features_rest = self._features[:, 1:, :].cpu()
        scaling = torch.log(self._scaling).cpu()
        rotation = self._rotation.cpu()
        opacity = inverse_sigmoid(self._opacity).cpu()

        return Gaussian(
            sh_degrees=self.max_sh_degree,
            xyz=xyz,
            opacities=opacity,
            features_dc=features_dc,
            features_extra=features_rest,
            scales=scaling,
            rotations=rotation,
        )

    def to_ply_structure(self) -> Gaussian:
        xyz = self._xyz.cpu().numpy()
        features_dc = self._features[:, :1, :].transpose(1, 2).cpu().numpy()
        features_rest = self._features[:, 1:, :].transpose(1, 2).cpu().numpy()
        scaling = torch.log(self._scaling).cpu().numpy()
        rotation = self._rotation.cpu().numpy()
        opacity = inverse_sigmoid(self._opacity).cpu().numpy()

        return Gaussian(
            sh_degrees=self.max_sh_degree,
            xyz=xyz,
            opacities=opacity,
            features_dc=features_dc,
            features_extra=features_rest,
            scales=scaling,
            rotations=rotation,
        )

class TimeTableGaussianModel:
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
    def check_inverse_scale(self):
        if os.path.exists(os.path.join(os.path.dirname(self.TimeTable_dir),"re_scale.json")):
            print(f"Found re_scale.json, will load it")
            with open(os.path.join(os.path.dirname(self.TimeTable_dir),"re_scale.json"), 'r') as json_file:
                dict_rescale = json.load(json_file)
                mean_xyz = torch.tensor(dict_rescale["mean_xyz"]).to(self.device)
                scale = dict_rescale["scale"]
                min_xyz = torch.tensor(dict_rescale["min_xyz"]).to(self.device)
                max_xyz = torch.tensor(dict_rescale["max_xyz"]).to(self.device)
            if (torch.isnan(self.time_pcd[:,:,:]).any(-1)).sum()==0:
                print("No nan value found in time_pcd, will rescale it")
                self.time_pcd[:,:,:3]=self.time_pcd[:,:,:3]/scale+ mean_xyz

            else:
                print("Found nan value in time_pcd, will rescale only non-nan value")
                pcd = self.time_pcd[(~torch.isnan(self.time_pcd[:,:,:]).any(-1))][:,:3]
                pcd=pcd/scale+mean_xyz
                new_time_pcd_xyz = torch.full_like(self.time_pcd[:,:,:3],fill_value=np.nan)
                new_time_pcd_xyz[(~torch.isnan(self.time_pcd[:,:,:]).any(-1))]=pcd
                new_time_pcd= torch.cat([new_time_pcd_xyz,self.time_pcd[:,:,3:]],-1)
                self.time_pcd = new_time_pcd

    def __init__(self,TimeTable_dir ,device="cpu",table_frame_interval=1) -> None:
        
        self.setup_functions()
        self.table_frame_interval = table_frame_interval
        self.device = torch.device(device)
        if os.path.exists(TimeTable_dir) is False:
            raise FileNotFoundError("time_pcd file not found")
        time_pcd= torch.Tensor(np.load(TimeTable_dir),device=self.device)
        self.TimeTable_dir = TimeTable_dir
        self.time_pcd = time_pcd[:,:,:6]
        
        self.check_inverse_scale()

        N,T=time_pcd.shape[:2]
        self.N = N
        self.T = T
        self.sh_degree = 0
        self.max_sh_degree = 3
        self.max_frames = T
    def set_max_frame(self, max_frame):
        self.max_frames = max_frame
    def get_max_frame(self):
        return self.max_frames
    def get_gaussians_at_timeX(self, t: int):
        if t > self.max_frames or t < 0:
            raise ValueError("Time out of range")
        if type(t) is not int:
            raise ValueError("Time must be an integer")
        t/=float(self.table_frame_interval)
        t= round(t)
        if t == self.T:
            t = self.T - 1
        gaussian_t=  self.time_pcd[:,t,:]
        not_nan_mask = ~torch.isnan(gaussian_t[:,:]).any(dim=1)
        xyz = gaussian_t[not_nan_mask, :3]
        rgb= gaussian_t[not_nan_mask, 3:6]
        # rgb=torch.ones_like(rgb)
        ## FIXME: 以后删除这一行
        LQM_DEBUG_SCALE_FACTOR=1
        xyz=xyz.contiguous().float().cuda()*LQM_DEBUG_SCALE_FACTOR
        rgb=rgb.contiguous().float().cuda()
        features = torch.zeros((rgb.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        fused_color=RGB2SH( rgb)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
        # dist2 = torch.clamp_min(distCUDA2(xyz), 0.000001)
        dist2 = torch.full((xyz.shape[0],), 0.0000001, device="cuda")
        
        scaling= torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3).float().cuda()
        rotation = torch.zeros((xyz.shape[0], 4), device="cuda").float().cuda()
        rotation[:, 0] = 1
        predefine_opacity_value =0.99999
        opacity = inverse_sigmoid(predefine_opacity_value * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        
        return GaussianModel_light( xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            self.sh_degree)
        

class Original_GaussianModel:
    """ original Gaussian model class copied from :https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py"""

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


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
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
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    @property    
    def get_normals(self,):
        #TODO 以后要改，现在直接返回【0,0,1】
        if not (hasattr(self,"normals") and self.normals.shape[0]==self._xyz.shape[0]):
            self.normals = torch.zeros_like(self._xyz)
            self.normals[:,2]=1
        return self.normals
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
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
            print("Active SH degree increased to ", self.active_sh_degree)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

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
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

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

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
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
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

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
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
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

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    def add_to_gaussians(self,cano_xyz,rgb):
        """add pnt to gaussian model

        Args:
            cano_xyz (_type_): _description_
            rgb (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert cano_xyz.shape[0] == rgb.shape[0]
        
        new_xyz = torch.Tensor(cano_xyz).cuda()
        # new_scaling = []
        new_rotation = torch.ones((new_xyz.shape[0], 4), device="cuda")
        new_rotation[:, 0] = 1
        new_fused_color = RGB2SH(torch.tensor(np.asarray(rgb)).float().cuda())
        features = torch.zeros((cano_xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = new_fused_color
        features[:, 3:, 1:] = 0.0
        new_features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        new_features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        dist2= torch.full((cano_xyz.shape[0],), 0.000005, device="cuda")
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) ## NOTE
        scales = torch.clamp(scales, -10, 1.0)
        new_scaling = scales 
        predefine_opacity_value =0.5
        new_opacities = inverse_sigmoid(predefine_opacity_value * torch.ones((cano_xyz.shape[0], 1), dtype=torch.float, device="cuda"))


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,new_rotation)
        return new_xyz.shape[0]

class PointTrackIsotropicGaussianModel(Original_GaussianModel):
    # raise NotImplementedError("Not implemented yet")

    """ Subclass inheriting from Original_GaussianModel to implement the isotropic gaussian model for point tracking"""
    
    def __init__(self, sh_degree : int):
        super().__init__(sh_degree)
        

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self.get_scaling_noact,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    def get_ones_xyz(self,):
        if hasattr(self,"ones_xyz") and self.ones_xyz.shape == self.get_xyz.shape:
            return self.ones_xyz
        else:
            self.ones_xyz = torch.ones_like(self.get_xyz)
            
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        _scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self._scaling=_scaling[:,:1] ## TODO: 这里的scaling是不是明明是一个值，但是我还是存了三个。
        
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        
        # self._xyz = nn.Parameter(model_args[1].repeat(1, time_length, 1).requires_grad_(True))
        # self._rotation = nn.Parameter(model_args[5].repeat(1, time_length, 1).requires_grad_(True))

        
        
    @property
    def get_scaling_noact(self):
        return self._scaling.repeat(1,3)
  
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling.repeat(1,3))
    
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        dist2= torch.full((fused_point_cloud.shape[0],), 0.000005, device="cuda")
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 1)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        predefine_opacity_value =0.999
        opacities = inverse_sigmoid(predefine_opacity_value * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        ## FIXME: 2024年5月18日15:09:51，为了dubug
        # try:
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)



    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.get_scaling_noact.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self.get_scaling_noact.detach().cpu().numpy() ## scale 只需要保存一个数值，但是还是保存了三次
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)



    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
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
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[:,:1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
        
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        # "rotation" : new_rotation
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]
        N=self._xyz.shape[0]
        _rotation=torch.zeros([N,4]).to(self._xyz.device)
        _rotation[:,0]=1
        self._rotation=nn.Parameter(_rotation).requires_grad_(False)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_scaling=new_scaling[:,:1]
        # new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]
        N=self._xyz.shape[0]
        _rotation=torch.zeros([N,4]).to(self._xyz.device)
        _rotation[:,0]=1
        self._rotation=nn.Parameter(_rotation).requires_grad_(False)
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
    def add_to_gaussians(self,cano_xyz,rgb):
        """add pnt to gaussian model

        Args:
            cano_xyz (_type_): _description_
            rgb (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert cano_xyz.shape[0] == rgb.shape[0]
        
        new_xyz = torch.Tensor(cano_xyz).cuda()
        # new_scaling = []
        new_rotation = torch.ones((new_xyz.shape[0], 4), device="cuda")
        new_rotation[:, 0] = 1
        new_fused_color = RGB2SH(torch.tensor(np.asarray(rgb)).float().cuda())
        features = torch.zeros((cano_xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = new_fused_color
        features[:, 3:, 1:] = 0.0
        new_features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        new_features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        dist2= torch.full((cano_xyz.shape[0],), 0.000005, device="cuda")
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 1) ## NOTE: IsotropicGaussian Model .
        scales = torch.clamp(scales, -10, 1.0)
        new_scaling = scales 
        predefine_opacity_value =0.6
        new_opacities = inverse_sigmoid(predefine_opacity_value * torch.ones((cano_xyz.shape[0], 1), dtype=torch.float, device="cuda"))


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling)
        return new_xyz.shape[0]

        
        return  



