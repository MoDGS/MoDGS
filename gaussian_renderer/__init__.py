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


from diff_gaussian_rasterization import GaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer 

from scene.gaussian_model import BasisGaussianModel
from scene.original_gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import numpy as np
import random
from tqdm import tqdm

def original_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,specified_xyz=None):
    """ from 
     https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        # prefiltered=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if specified_xyz is not None:
        if specified_xyz.dim()==3:
            specified_xyz = specified_xyz.squeeze(0)
        means3D = specified_xyz
    else:
        means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            # FIXME: THIS is the Origianl Verison:shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    details_dict=None
    res  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    if len(res)==4:
        rendered_image, radii, depth, alpha = res
    elif len(res)==2:
        rendered_image, radii = res
        depth=None
        alpha=None
    elif len(res)==10:
        rendered_image, radii, depth, alpha, proj_means_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu= res
        details_dict = {"rendered_depth":depth,"rendered_alpha":alpha,"proj_means_2D":proj_means_2D,"conic_2D":conic_2D,"conic_2D_inv":conic_2D_inv,"gs_per_pixel":gs_per_pixel,"weight_per_gs_pixel":weight_per_gs_pixel,"x_mu":x_mu}    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "alpha":alpha,
            "render_details":details_dict}
    
    #invisible_pcd= means3D[~(radii>0)]
    # invisible_pcd_homo =torch.cat([invisible_pcd,torch.ones([invisible_pcd.shape[0],1],device="cuda")],dim=1)
    #w2c= viewpoint_camera.world_view_transform.cuda()
    # invisible_pcd_cam = (invisible_pcd_homo@w2c.cuda())[:,:3]
    #np.savetxt("./Jupyter_test_exported/papervindmin_invisible_pcd_cam.txt",invisible_pcd_cam.cpu().numpy(),delimiter=" ")
    #pro_pcd_homo =invisible_pcd_homo@viewpoint_camera.full_proj_transform.cuda()

def render_depth_normal(viewpoint_camera, pc : GaussianModel, 
                            pipe, bg_color : torch.Tensor, 
                            # sh_degree,
                            scaling_modifier = 1.0, 
                            override_color = None,

                            # ray-plane args
                            use_depth_rayplane_intersect = False,
                            img_ray_dir_cam = torch.tensor([]), ## 
                            normals_gauss_cam = torch.tensor([]),

                            # diff args
                            is_render_depth_diff = False,
                            img_depth=torch.tensor([]),
                            is_render_normal_diff = False,
                            img_normal=torch.tensor([]),
                            
                            check_normal_dir=False,
                            specified_xyz=None,
                            ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # use_depth_rayplane_intersect = True
    # normals_gauss_cam = override_normal_cam
    # TODO: check correpondence of variables
    raster_settings = JiePengGaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree, #sh_degree=sh_degree, # 
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        check_normal_dir=check_normal_dir,
        img_normal_prior=img_normal,
        use_depth_rayplane_intersect=use_depth_rayplane_intersect,
        img_ray_dir=img_ray_dir_cam,
        is_render_depth_diff=is_render_depth_diff,
        img_depth=img_depth,
        is_render_normal_diff=is_render_normal_diff,
        img_normal=img_normal
        # normals_cam=normals_gauss_cam
    )
    # print(f'[here] use_depth_rayplane_intersect')

    rasterizer = JiePengGaussianRasterizer(raster_settings=raster_settings)

    if specified_xyz is not None:
        if specified_xyz.dim()==3:
            specified_xyz = specified_xyz.squeeze(0)
        means3D = specified_xyz
    else:
        means3D = pc.get_xyz
    # means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        shs = None
    # if use_depth_rayplane_intersect:
    #     colors_precomp = means3D
    use_detached_gauss_prop =use_depth_rayplane_intersect
    if use_detached_gauss_prop:
        means3D = means3D.clone().detach()
        means2D = means2D.clone().detach()
        opacity = opacity.clone().detach()
        scales = scales.clone().detach()
        rotations = rotations.clone().detach()
    
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii = rasterizer(
    res = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        normals_cam = normals_gauss_cam)
    
    if len(res)==4:
        rendered_image, radii, depth, alpha = res
    elif len(res)==2:
        rendered_image, radii = res
        depth=None
        alpha=None
    elif len(res)==10:
        rendered_image, radii, rendered_depth, rendered_alpha, proj_means_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu= res
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "alpha":alpha}   
    
    
    

    

    
############################################################################################
################# Code below are old  before 2024年3月1日21:54:13，Basis Gaussian的模型。#####
############################################################################################
def render_video(scene,gaussians,pipe,background):
    # print("Rendering video...")
    views = scene.getVisCameras()
    rendering_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering Video progress")):
        if type(view) is list:
            view = view[0]
        rendering = render(view, gaussians, pipe, background)["render"]
        rendering_list.append((torch.clip(rendering.detach(),0,1)))
    return torch.stack(rendering_list,dim=0)
    ## shape=(num_frames,3,im_height,im_width)
    
    pass

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, itr=-1,static_util_iter=3000,time_noise=False,time_interval=0,smooth_term=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    time = viewpoint_camera.time
    idx1, idx2, idx3 = 0, 1, 2
    # mask = torch.logical_and(pc.get_xyz[:, 3, 0] <= time, time <= pc.get_xyz[:, 4, 0])
    # print(pc.get_xyz[:, 3, 0], pc.get_xyz[:, 4, 0])
    # if time >= 0.5:
        # idx1, idx2, idx3 = 3, 4, 5        
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz[:, 0, :], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # print(viewpoint_camera.R, viewpoint_camera.T, viewpoint_camera.FoVx, viewpoint_camera.FoVy)

    # static_util_iter = 3000
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    # print(pc.get_xyz[:, 0, :].mean(axis=0).data, pc.get_opacity.mean().item(), pc.get_rotation[:, 0, :].mean(axis=0).data, pc.get_scaling.mean().item()) # , pc.get_features.mean(axis=0))
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if not time_noise:
        ast_noise = 0
    else:
        ast_noise = random.gauss(0, 1)*time_interval * smooth_term(itr)
    time =time+ast_noise
    L = pc.L
    basis = 2**torch.arange(0, L, device='cuda').repeat_interleave(2)*math.pi*time
    basis[::2] = torch.sin(basis[::2])
    basis[1::2] = torch.cos(basis[1::2])
    basis_global=None
    if hasattr(pc,"L_global") and pc.L_global>0:
            basis_global = 2**torch.arange(0, pc.L_global, device='cuda').repeat_interleave(2)*math.pi*time
            basis_global[::2] = torch.sin(basis_global[::2])
            basis_global[1::2] = torch.cos(basis_global[1::2])
    movement=None       
    if itr != -1 and itr <= static_util_iter:
        means3D = pc.get_xyz[:, 0, :]
    else:
        # means3D = pc.get_xyz[:, 0, :] + (pc.get_xyz[:, 1:2*L+1, :]*basis.unsqueeze(-1)).sum(1)
        means3D,movement=pc.step(basis,basis_global)

    means2D = screenspace_points[:]
    opacity = pc.get_opacity[:]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[:]

        if L == 0:
            rontations = pc.get_rotation[:, 0, :]
        else:
            if itr != -1 and itr <= static_util_iter:
                rotations = pc.get_rotation[:, idx1, :]
            else:
                rotations = pc.get_rotation[:, idx1, :] + pc.get_rotation[:, idx2, :]*time

        # rotations = pc.get_rotation[:, 0, :] + pc.get_rotation[:, 1, :]*torch.sin(torch.tensor(math.pi*time)) + pc.get_rotation[:, 2, :]*torch.cos(torch.tensor(math.pi*time)) + pc.get_rotation[:, 3, :]*torch.sin(2*torch.tensor(math.pi*time)) + pc.get_rotation[:, 4, :]*torch.cos(2*torch.tensor(math.pi*time)) + pc.get_rotation[:, 5, :]*torch.sin(4*torch.tensor(math.pi*time)) + pc.get_rotation[:, 6, :]*torch.cos(4*torch.tensor(math.pi*time)) + pc.get_rotation[:, 7, :]*torch.sin(8*torch.tensor(math.pi*time)) + pc.get_rotation[:, 8, :]*torch.cos(4*torch.tensor(math.pi*time))

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            if itr != -1 and itr <= static_util_iter:
                dir_pp = (pc.get_xyz[:, 0, :] - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            else:
                dir_pp = (pc.get_xyz[:, idx1, :] + (pc.get_xyz[:, 1:2*L+1, :]*basis.unsqueeze(-1)).sum(1) - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))

            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii , depth, alpha= rasterizer(
        means3D = means3D,##torch.Size([214475, 3])
        means2D = means2D,##torch.Size([214475, 3])
        shs = shs,##torch.Size([214475, 16, 3])
        colors_precomp = colors_precomp,
        opacities = opacity,##torch.Size([214475, 1])
        scales = scales,##torch.Size([214475, 3])
        rotations = rotations,##torch.Size([214475, 4])
        cov3D_precomp = cov3D_precomp)
    # if itr % 400 == 0:
        # print(sum(radii > 0)/len(radii), len(radii))
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "alpha":alpha,
            "movement":movement}



