import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
# from scene import PointTrackScene
import os 
from argparse import ArgumentParser, Namespace
import uuid
import numpy as np
from utils.image_utils import psnr, psnr_masked
from utils.loss_utils import l1_loss,ssimmap,ssim
from utils import flow_viz 
from pytorch3d.ops import knn_gather, knn_points,ball_query
import PIL.Image as Image
import imageio
from tqdm import tqdm
from utils.system_utils import check_exist
from gaussian_renderer import render_depth_normal
import subprocess
# import moviepy.editor as mpy
import av
import torchvision

import utils.flow_viz as flow_viz
from utils.pearson_coeff import pearson_corrcoef
import torchvision
from torchvision.utils import  save_image
from torch import nn
from utils.system_utils import check_exist
from kornia.filters import median_blur 
def tob8(img):
    if torch.is_tensor(img):
        return (255*np.clip(img.cpu().squeeze().numpy(),0,1)).astype(np.uint8)
    else:
        return (255*np.clip(img.squeeze(),0,1)).astype(np.uint8)
def show_img(img):
    return Image.fromarray((img*255).astype(np.uint8))
def minmax_norm(img,img_min=None,img_max=None):
    if img_min is None:
        img_min = img.min()
    if img_max is None:
        img_max = img.max()
    norm = (img-img_min)/(img_max-img_min)
    return norm
def fun():
    pass
def construct_2d_matrix(x,y,z):
    """
    Construct a 2D matrix from 3 1D vectors
    """
    K,H,W =x.shape
    mat = torch.full((2,2,K,H,W),torch.nan,device=x.device)
    mat[0,0] = x
    mat[0,1] = y
    mat[1,0] = y
    mat[1,1] = z
    return mat
def cumulativeWeight2weight(cum_weight_per_gs_pixel):
    """weight_per_gs_pixel: 20*H*W"""
    weight_per_gs_pixel = torch.zeros_like(cum_weight_per_gs_pixel)
    weight_per_gs_pixel[1:,:,:]=cum_weight_per_gs_pixel[1:,:,:]-cum_weight_per_gs_pixel[:-1,:,:]
    weight_per_gs_pixel[0,:,:] = cum_weight_per_gs_pixel[0,:,:]
    weight_per_gs_pixel[weight_per_gs_pixel<0] = 0

    return weight_per_gs_pixel


def get_gs_2d_flow(render_details,render_details_target):
    """ GAO QUANKAI
    get the 2D flow from the rendered details , currently we only consider the situation of isotropic gaussian(Ignore the Covariance of gaussians,
    as mentioned in the paper, flow = Sum(wi*(u_it-u_it+1))
    )
    Args:
        render_details (dict): dict contains the rendered from diff-rasterizer
        render_details_target (dict): dict contains the rendered from diff-rasterizer
        
    Returns:
        _type_: 2d estimated flow 
    """
    proj_means_2D = render_details["proj_means_2D"] # torch.Size([300000, 2])
    conic_2D = render_details["conic_2D"] ## conic_2D.shape torch.Size([300000, 3])
    conic_2D_inv = render_details["conic_2D_inv"]## conic_2D_inv.shape torch.Size([300000, 3])
    gs_per_pixel = render_details["gs_per_pixel"].long()## gs_per_pixel.shape torch.Size([20, 540, 960])
    weight_per_gs_pixel = render_details["weight_per_gs_pixel"] ## weight_per_gs_pixel.shape torch.Size([20, 540, 960])
    x_mu = render_details["x_mu"] #x_mu.shape torch.Size([20, 2, 540, 960])
    proj_means_2D_target = render_details_target["proj_means_2D"]
    conic_2D_target = render_details_target["conic_2D"]
    conic_2D_inv_target = render_details_target["conic_2D_inv"]
    # gs_per_pixel_target = render_details_target["gs_per_pixel"]
    weight_per_gs_pixel_target = render_details_target["weight_per_gs_pixel"]
    x_mu_target = render_details_target["x_mu"]
    weights = (weight_per_gs_pixel[:,:,:,None]) / (weight_per_gs_pixel[:,:,:,None].sum(0) + 1e-6)
    flow = (proj_means_2D_target[gs_per_pixel] - proj_means_2D[gs_per_pixel].detach()) * weights.detach()
    flow = flow.sum(0).permute(2,0,1)
    return flow
def get_flow_loss(render_flow,gt_flow,mask,loss_method="mae"):
    """_summary_

    Args:
        render_flow (tensor):2*H*W
        gt_flow (tensor): 2*H*W
        mask (tensor): H*W
        loss_method (str, optional): _description_. Defaults to "mae".
    Raises:
        NotImplementedError: _description_

    Returns:
       flow loss  _type_: tensor 
    """
    assert render_flow.shape == gt_flow.shape and render_flow.shape[0]==2 ## 2*H*W,
    if loss_method =="mae":
        return torch.sum(torch.abs(render_flow-gt_flow)*mask)/(torch.sum(mask)+1e-8)
    elif loss_method == "normalized_mae":
        # return torch.mean(torch.abs(render_flow-gt_flow))
                    # # ##1. 
        gt_flow = gt_flow / (torch.max(torch.sqrt(torch.square(gt_flow).sum(0))) + 1e-5)
        # fwd_flow = fwd_flow / (torch.max(torch.sqrt(torch.square(fwd_flow).sum(-1))) + 1e-5)
        render_flow = render_flow / (torch.max(torch.sqrt(torch.square(render_flow).sum(0))) + 1e-5)
        # render_flow_fwd = render_flow_fwd / (torch.max(torch.sqrt(torch.square(render_flow_fwd).sum(-1))) + 1e-5)
        M = mask.unsqueeze(0)
        flow_loss = torch.sum(torch.abs(gt_flow - render_flow) * M) / (torch.sum(M) + 1e-8) / gt_flow.shape[0]
        return flow_loss 
    elif loss_method == "masked_pearson":
        if mask.dim()==3:
            mask = mask.squeeze()
        flow_loss_u = 1 - pearson_corrcoef( render_flow[0,mask>0], gt_flow[0,mask>0]) ## channel 0,u
        flow_loss_v = 1 - pearson_corrcoef( render_flow[1,mask>0], gt_flow[1,mask>0])## channel 1,v
        return 0.5*(flow_loss_u+flow_loss_v)
    elif loss_method == "masked_pearson_Rad_Direction":
        if mask.dim()==3:
            mask = mask.squeeze()
        render_R2= torch.square(render_flow).sum(0)
        gt_flow_R2 = torch.square(gt_flow).sum(0)
        flow_loss_u = 1 - pearson_corrcoef( render_flow[0,mask>0], gt_flow[0,mask>0]) ## channel 0,u
        flow_loss_v = 1 - pearson_corrcoef( render_flow[1,mask>0], gt_flow[1,mask>0])## channel 1,v
        return 0.5*(flow_loss_u+flow_loss_v)
    elif loss_method == "l2norm":
        assert mask.dim()==2
        Lflow = torch.norm((render_flow - gt_flow)[:,mask], p=2, dim=0).mean()
        return Lflow
    else:
        raise NotImplementedError(loss_method)
    # return torch.sum(torch.abs(render_flow-gt_flow)*mask)/(torch.sum(mask)+1e-8)

def get_depth_order_loss(render_depth,gt_depth,mask,method_name="sign_loss",pair_num = 200000,alpha=100,):
    """_summary_

    Args:
        render_depth (_type_): 1,H,W,
        gt_depth (_type_): H,W,
        mask (_type_): H,W,
        method_name (str, optional): _description_. Defaults to "pearson".

    Returns:
        _type_: _description_
    """
    if method_name=="vallina":
        
        raise NotImplementedError


    elif method_name=="SingleImageDepthPerceptionintheWild" or method_name=="DepthRanking": ## https://arxiv.org/pdf/1604.03901.pdf
        loss = 0.0
        threshold = (gt_depth.max()-gt_depth.min())/100 ## 分成 10 个区间，大于区间才考虑
        gt_depth=gt_depth[mask>0]## N,1
        render_depth=render_depth.squeeze(0)[mask>0]## N,1
        index1 = torch.randperm(gt_depth.shape[0])[:pair_num,]
        index2 = torch.randperm(gt_depth.shape[0])[:pair_num,]
        gt_depth_diff = gt_depth[index1]-gt_depth[index2]
        gt_oder = torch.sign(gt_depth_diff) ## 1 or -1
        gt_oder[gt_depth_diff.abs()<threshold]=0
        selected_i = render_depth[index1]
        selected_j = render_depth[index2]
        
        # loss += torch.mean(torch.clamp((selected_i-selected_j)[gt_oder==1],min=None,max=0))
        # loss += torch.mean(torch.clamp((selected_j-selected_i)[gt_oder==-1],min=None,max=0))
        pred_depth = render_depth[index1]-render_depth[index2]
        log_loss = torch.mean(torch.log(1 + torch.exp(-gt_oder[gt_oder != 0] * pred_depth[gt_oder != 0])))
        square_loss = torch.mean(torch.square(pred_depth[gt_oder == 0]))
        return log_loss+square_loss
        

    elif method_name=="tanh_threshold":
        # alpha = 100
        ### 用 Tanh 来近似这个符号函数。
        gt_depth=gt_depth[mask>0]## N,1
        depthmax=gt_depth.max()
        depthmin=gt_depth.min()
        interval = (depthmax-depthmin)/10
        # interval = (depthmax-depthmin)/20
    
        render_depth=render_depth.squeeze(0)[mask>0]## N,1
        index1 = torch.randperm(gt_depth.shape[0])[:pair_num,]
        index2 = torch.randperm(gt_depth.shape[0])[:pair_num,]
        threshold_msk  = (torch.abs(gt_depth[index1]-gt_depth[index2])>=interval)
        index1 = index1[threshold_msk]
        index2 = index2[threshold_msk]
        
        gt_oder = torch.sign(gt_depth[index1]-gt_depth[index2])
        # render_oder = torch.sign(render_depth[index1]-render_depth[index2])
        render_diff = render_depth[index1]-render_depth[index2]

        loss = torch.mean(torch.abs(torch.tanh(alpha*render_diff)-gt_oder))
        return loss

        
    else:
        raise NotImplementedError(method_name)

    return loss
def localsmoothness_loss(query_pcd,pcd, flow, neighbor_K=10):
        pairwise_dist = knn_points(query_pcd.unsqueeze(0), pcd.unsqueeze(0), K=neighbor_K, return_sorted=False)

        # Gather the flow of the k nearest neighbors for each point
        neighbor_flows = knn_gather(flow.unsqueeze(0), pairwise_dist.idx, )#neighbor_K)
        neighbor_flows=neighbor_flows[:,:,1:,:] ## remove the first point which is the point itself
        # Compute the mean flow of the k nearest neighbors for each point
        # mean_flow = neighbor_flows.mean(dim=2)

        # Compute the difference between each point's flow and the mean flow of its neighbors
        loss = torch.mean(torch.square(flow.unsqueeze(0).unsqueeze(2)-neighbor_flows))


        return loss
    

    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(os.path.join(args.model_path,args.timestamp)))
    os.makedirs(os.path.join(args.model_path,args.timestamp), exist_ok = True)
    with open(os.path.join(args.model_path,args.timestamp, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(args.model_path,args.timestamp))
        print("Tensor board Directory:",os.path.join(args.model_path,args.timestamp))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
def adjust_time():
    
    pass
# torch.norm()

def cotraining_report(tb_writer, iteration:int,loss_dict, elapsed, testing_iterations, scene , renderFunc, renderArgs,):
    # raise NotImplementedError("This function is not implemented")
    # light_writer = True
    light_writer = False
    if tb_writer:        
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for k,v in loss_dict.items():
            # if v is not None and v.item() >1e-10:
            if v is not None :
                try:
                    tb_writer.add_scalar('train_loss_patches/'+k, v.item(), iteration)
                except:
                    tb_writer.add_scalar('train_loss_patches/'+k, v, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        (pipe, background) =renderArgs
        black_bg =  torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        save_dir=check_exist(os.path.join(tb_writer.log_dir,"render_video"))
        if not light_writer:
            render_spiral_video(tb_writer,scene,scene.gaussians,renderFunc,renderArgs,iteration)
        torch.cuda.empty_cache()
        CoTraining_Cams= scene.getCoTrainingCameras()
        train_cams = [CoTraining_Cams[idx % len(CoTraining_Cams)] for idx in range(0, len(CoTraining_Cams), 4)]
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getCoTestingCameras()[::30]},  ## LQM: TODO: 为了测试，先注释掉,for ICLR2025 rebuttal 
                              {'name': 'train', 'cameras' : train_cams })
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                test_view_list=None
                flow_view_list=None
                l1_test = 0.0
                psnr_test = 0.0
                psnr_test_masked=0.0
                flow_loss = 0.0
                if config['name'] == 'train':
                    flow_view_list = []
                if config['name'] == 'test':
                    test_view_list = []
                for idx, data in tqdm(enumerate(config['cameras'])):
                    # if idx <1 or idx>len(config['cameras'])-2:
                    #     continue
                    
                    viewpoint,pcd_pair = data
                    #### Predict Gaussian position. 
                    xyz= scene.gaussians.get_xyz
                    # time = pcd_pair["time"]
                    # time = viewpoint.time.unsqueeze(0)*428.0/(len(scene.timePcd_dataset))/scene.timePcd_dataset.PCD_INTERVAL
                    time = scene.rescale_time(viewpoint.time, pcd_pair["time"])
                    # time = scene.rescale_time(viewpoint)
                    predicted_xyz= scene.net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
                    ###
                    rendered_pkg=renderFunc(viewpoint, scene.gaussians, *renderArgs,specified_xyz = predicted_xyz)
                    rendered_img,rendered_depth,rendered_alpha=rendered_pkg["render"],rendered_pkg["depth"],rendered_pkg["alpha"]        
                    image = torch.clamp(rendered_img, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # if loss_dict["L2d_render_flow"] is not None and  config['name'] == 'train':  ### TODO： 现在是所有image 都加入训练，flow 有一点问题。flow是有间隔的。

                    if config['name'] == 'test':
                        test_view_list.append(image)
                        if viewpoint.mask is not None :
                            mask = viewpoint.mask[None].to(gt_image)
                            gt_image = gt_image * mask
                            image = image * mask
                    if iteration in [1,30000]: ## saving  images
                    # if iteration in [1,15000,30000]: ## saving  images
                        # if config['name'] == 'test':
                        check_exist(os.path.join(save_dir,config['name'],f"view_iteration_{iteration}"))
                        save_image(image,os.path.join(save_dir,config['name'],f"view_iteration_{iteration}",f"{viewpoint.image_name}.png"))
                        if rendered_depth is not None:
                            rendered_depth=  torch.clamp(minmax_norm(rendered_depth,scene.near,scene.far),0,1)
                            save_image(rendered_depth,os.path.join(save_dir,config['name'],f"view_iteration_{iteration}",f"{viewpoint.image_name}_depth.png"))
                        if rendered_alpha is not None:
                            save_image(rendered_alpha,os.path.join(save_dir,config['name'],f"view_iteration_{iteration}",f"{viewpoint.image_name}_alpha.png"))         
                    
                    # if tb_writer and (idx % 10):
                    if tb_writer:
                    # if tb_writer  and not light_writer:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_errormap".format(viewpoint.image_name), torch.abs(image[None]-gt_image[None]), global_step=iteration)
                        if rendered_depth is not None:
                            rendered_depth= torch.clamp(minmax_norm(rendered_depth,scene.near,scene.far),0,1)
                            tb_writer.add_images(config['name'] + "_view_{}/render_depth".format(viewpoint.image_name), rendered_depth[None], global_step=iteration)
                        if rendered_alpha is not None:
                            tb_writer.add_images(config['name'] + "_view_{}/render_alpha".format(viewpoint.image_name), rendered_alpha[None], global_step=iteration)
                          
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if viewpoint.mask is not None:  
                                tb_writer.add_images(config['name'] + "_view_{}/gt_mask".format(viewpoint.image_name), viewpoint.mask[None][None], global_step=iteration)
                            if viewpoint.depth is not None:
                                depth=viewpoint.depth
                                # depth= (depth-depth.min())/(depth.max()-depth.min())
                                depth= torch.clamp(minmax_norm(depth,scene.near,scene.far),0,1)
                                tb_writer.add_images(config['name'] + "_view_{}/gt_depth".format(viewpoint.image_name), depth[None][None], global_step=iteration)
 

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if viewpoint.mask is not None and config['name'] == 'test':
                        psnr_test_masked += psnr_masked(image, gt_image,mask).double()

                    
                psnr_test /= len(config['cameras'])
                if viewpoint.mask is not None and config['name'] == 'test':
                    psnr_test_masked /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                if not light_writer and test_view_list is not None:
                    img_tensor =torch.stack(test_view_list)
                    tensor_to_video(img_tensor.permute(0,2,3,1), os.path.join(save_dir,config['name']+f"view_iteration_{iteration}_rgb.mp4"), fps=10)
                    add_video_to_tensorboard(os.path.join(save_dir,config['name']+f"view_iteration_{iteration}_rgb.mp4"),tb_writer, config['name']+"_video", iteration)
                if not light_writer and flow_view_list is not None and len(flow_view_list)>0:
                    img_tensor =torch.stack(flow_view_list)
                    tensor_to_video(img_tensor.permute(0,2,3,1), os.path.join(save_dir,config['name']+f"view_iteration_{iteration}_rgb.mp4"), fps=10)
                    add_video_to_tensorboard(os.path.join(save_dir,config['name']+f"view_iteration_{iteration}_rgb.mp4"),tb_writer, config['name']+"_video", iteration)

                if flow_loss>0:
                    flow_loss/=len(config['cameras'])
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - flow_loss', flow_loss, iteration)    

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if viewpoint.mask is not None and config['name'] == 'test':
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_masked',psnr_test_masked, iteration)


        
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
# render_spiral_video(writer,scene,gaussians,RenderFunc,renderArgs)   
def render_spiral_video(writer ,scene ,gaussians,RenderFunc,renderArgs,iteration):
    print("Rendering Spiral Video")
    for key in tqdm(list(scene.spiral_cameras.keys())):
            # scene.spiral_cameras[key].original_image=scene.spiral_cameras[key].original_image.cuda()
        Spiral_views = scene.getSpiralCameras(key)
        img_list=[]
        depth_list=[]
        alpha_list=[]
        for viewpoint in  tqdm(Spiral_views):
            time = torch.Tensor([viewpoint.time]).to("cuda")
            # time = scene.rescale_time(torch.tensor([viewpoint.time]).cuda(),None)
            xyz= gaussians.get_xyz
            predicted_xyz= scene.net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
            rendered_pkg=RenderFunc(viewpoint, scene.gaussians, *renderArgs,specified_xyz = predicted_xyz)
            rendered_img,rendered_depth,rendered_alpha=rendered_pkg["render"],rendered_pkg["depth"],rendered_pkg["alpha"]     
            if RenderFunc.__name__ == "render_depth_normal":
                # pass 
                img_ray_dir_cam=scene.cam_direction
                normals_gauss_cam=gaussians.get_normals
                cam_xyz = predicted_xyz.squeeze(0)@viewpoint.world_view_transform[:3,:3].cuda() +viewpoint.world_view_transform[3,:3].cuda()
                override_color = cam_xyz
                rendered_depth = RenderFunc(viewpoint, scene.gaussians, *renderArgs,override_color=override_color,
                                            
                            specified_xyz = predicted_xyz.squeeze(0), 
                            use_depth_rayplane_intersect = True,
                            img_ray_dir_cam =img_ray_dir_cam, ## 
                            normals_gauss_cam = normals_gauss_cam,
                            
                            is_render_depth_diff = False,
                            img_depth=torch.tensor([]), ## HW3
                            is_render_normal_diff = False,
                            img_normal=torch.tensor([]),
                            
                            check_normal_dir=False,
                            
                            
                            )["render"][2:,...]
                rendered_alpha = RenderFunc(viewpoint, scene.gaussians, *renderArgs,override_color=scene.gaussians.get_ones_xyz(),
                                            specified_xyz = predicted_xyz.squeeze(0), 
                            )["render"][2:,...]

            elif  RenderFunc.__name__ == "original_render":
                pass
            else:
                raise NotImplementedError("RenderFunc.__name__")
            
            if rendered_alpha is not None:   
                alpha = rendered_alpha.cpu().detach()
                alpha_list.append((minmax_norm(alpha)))
                
            if rendered_depth  is not  None:
                depth= rendered_depth.cpu().detach()
                depth_list.append(minmax_norm(depth,scene.near,scene.far))
                
            rendering_img = rendered_pkg["render"].cpu().detach().clamp(0,1)
            # rendering_img = (255*np.clip(rendering_img.cpu().numpy(),0,1)).astype(np.uint8).transpose(1, 2, 0)
        # if output_video==True:
            img_list.append(rendering_img)


        img_tensor = torch.stack(img_list)
        video_path = os.path.join(writer.log_dir,"render_video", f"step{iteration}_spiral_rgb_time_{key}.mp4")
        tensor_to_video((img_tensor.permute(0,2,3,1)*255).byte(), video_path, fps=10)
        add_video_to_tensorboard(video_path,writer, f"spiral_rgb_time_{key}", iteration)
        # writer.add_video(f"spiral_rgb_time_{key}",img_tensor[None],fps=20,global_step=iteration)
        
        if depth_list:
            
            depth_tensor = torch.stack( depth_list)
            video_path = os.path.join(writer.log_dir,"render_video", f"step{iteration}_spiral_depth_time_{key}.mp4")
            tensor_to_video((depth_tensor.permute(0,2,3,1)*255).byte(), video_path, fps=10)
            add_video_to_tensorboard(video_path,writer, f"spiral_depth_time_{key}", iteration)
            
            # writer.add_video(f"spiral_depth_time_new_{key}",torch.stack(depth_list)[None],fps=20,global_step=iteration)
        if alpha_list :   
            writer.add_video(f"spiral_alpha_time_{key}",torch.stack(alpha_list)[None],fps=10,global_step=iteration)
    



def tensor_to_video(tensor, video_path, fps=30):
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()

    N, H, W, C = tensor.shape
    if not tensor.dtype == torch.uint8:
        tensor = (tensor * 255).byte()
    if C==1:
        tensor = tensor.repeat(1, 1, 1, 3)

    container = av.open(video_path, mode='w')
    stream = container.add_stream('mpeg4', rate=fps)
    stream.width = W
    stream.height = H
  
    stream.options = {'b:v': '1.0M'}  


    for i in range(N):
        frame = av.VideoFrame.from_ndarray(tensor[i].numpy(), format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)


    container.close()

import torch
import torchvision

def add_video_to_tensorboard(video_path, writer, tag, iteration):

    if not os.path.exists(video_path):
        print("video_path not exists")
    video, audio, info = torchvision.io.read_video(video_path, pts_unit='sec')
    video = video.permute(0, 3, 1, 2)  # N*H*W*3 -> N*3*H*W

    writer.add_video(tag, video.unsqueeze(0), iteration, fps=info["video_fps"])

def add_flow_to_tensorboard(flow_dict,writer,iteration,config_name,viewpoint_name,add_gt=False):
    trans_=torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]])).float().T.cuda()
    for k,v in flow_dict.items():
        if v is not None:
            if not add_gt and "gt" in k:
                continue
            if  not "mask" in k:

                flow = v.permute(1, 2, 0)  ## TODO: 变不变换的区别是什么。
                flow = flow.cpu().numpy()
                flow_img = np.transpose(flow_viz.flow_to_image(flow),(2,0,1))
                writer.add_images(config_name + "_view_{}/{}_flow".format(viewpoint_name,k), flow_img[None], global_step=iteration)
            else:
                msk_img = v
                writer.add_images(config_name + "_view_{}/{}_flow".format(viewpoint_name,k), msk_img[None][None], global_step=iteration)
                

