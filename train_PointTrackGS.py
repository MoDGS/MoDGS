import os
import torch
from torch.utils.data import DataLoader
from random import randint
from utils.loss_utils import l1_loss, ssim,mask_l1_loss,mask_ssim,l2_loss
import sys
from scene import PointTrackScene,GaussianModelTypes
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import PointTrackModelParams, PipelineParams, PointTrackOptimizationParams,MLP_Params,NeuralInverseTrajectory_Params
import copy
from datetime import datetime
# from utils.flow_viz import flow_to_image
import utils.flow_viz as flow_viz
import imageio
import torch.nn as nn
import lpips
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import random
from PIL import Image
import numpy as np
from utils.system_utils import make_source_code_snapshot,get_timestamp,check_exist
from model.neuralsceneflowprior import Neural_Flow
# from model.RealNVP import RealNVP
from model.mfn import GaborNet
from model import get_embedder
from dataloader.timePcdTable_dataset import TimePCDTable
from dataloader import get_dataset_by_table_dir
from utils.system_utils import save_cfg
from glob import glob
from gaussian_renderer import network_gui
import torch.nn.functional as F
from utils.loss_utils import get_depthloss
from utils.gaussian_training_utils import  show_img,minmax_norm,tob8
def run_network(xyz_inputs, time_inputs,embed_fn_xyz, embed_fn_time, network_func):
    """Prepares inputs and applies network 'fn'.
    """
    if xyz_inputs.dim()==2:
        xyz_inputs = xyz_inputs.unsqueeze(0)
    xyz_embeded= embed_fn_xyz(xyz_inputs)

    times= time_inputs.unsqueeze(0).reshape(xyz_inputs.shape[0],xyz_inputs.shape[1],1)
    time_embeded = embed_fn_time(times)
    input = torch.cat([xyz_embeded,time_embeded],dim=-1)
    outs= network_func(input)
    
    
    return outs

def test_table_completion(args):
    # def searchForMaxIteration(folder):
    #     saved_iters = [int(fname.split("_")[-1]) for fname in glob(os.path.join(folder,"*.pth"))]
    #     return max(saved_iters)
    from pointTrack_tools import get_start_end,table_completion
    
    ckpt_path = args.ckpt_path
    embed_fn_xyz, input_ch_xyz = get_embedder(args.multires_xyz, input_dim=3)
    embed_fn_time, input_ch_time = get_embedder(args.multires_time, input_dim=1)
    
    
    
    model = Neural_Flow(input_dim=input_ch_xyz+input_ch_time ,output_dim=args.output_dim,filter_size= args.filter_size,act_fn= args.act_fn,net_depth= args.net_depth).cuda()

    model.load_state_dict(torch.load(os.path.join(ckpt_path)))
    model.eval()
    network_query_fn = lambda inputs, times : run_network(inputs, times,  embed_fn_xyz,
                                                                embed_fn_time,
                                                                model)
    # outdir= check_exist(os.path.join(args.stage1_model_path,args.timestamp))
    outdir =os.path.join(args.stage1_model_path,args.timestamp)
    print(f"completion outdir:{outdir}")
    args.outdir = outdir
    
    # timePcddataset = TimePCDTable(args,keeprgb=True,device="cuda")
    timePcddataset = TimePCDTable(args,keeprgb=True, do_sale = True,device="cpu")
    # timePcddataset.filter_firstN(args.filterfirst_N)
    # dataloader = DataLoader(timePcddataset, batch_size=1, num_workers=0,shuffle=True)
    # iteration=0  tem maks =(N_se[:,2]-N_se[:,1])==10 and N_se[:,1]==10
    valid_mask  =timePcddataset.get_valid_mask()
    time_pcd = timePcddataset.get_time_pcd().clone()
    N_se  =get_start_end(valid_mask)
    with torch.no_grad():
        completed_table = table_completion(time_pcd,valid_mask,N_se,network_query_fn)
    
    
    np.save(os.path.join(outdir,"completed_table.npy"),completed_table.cpu().numpy())
    return completed_table

def validate_table_completion(model,dataset,iteration,writer):
    model.eval()
    val_dataset= dataset.get_val_dataset()
    # with torch.no_grad():
    embed_fn_xyz, input_ch_xyz = get_embedder(args.multires_xyz, input_dim=3)
    embed_fn_time, input_ch_time = get_embedder(args.multires_time, input_dim=1)
    val_l2loss = 0.0
    dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0,shuffle=False)
    with torch.no_grad():
        for idx,data in enumerate(dataloader):
            xyz= data["valid_xyz"]
            time= data["time"]
            time = time.expand(xyz.shape[0],xyz.shape[1],-1)
            xyz_embeded= embed_fn_xyz(xyz)
            time_embeded = embed_fn_time(time)
            input = torch.cat([xyz_embeded,time_embeded],dim=-1)
            flow= model(input)
            bwd_flow,fwd_flow= flow[...,:3],flow[...,3:]
            # loss_bwd=0.0
            # loss_fwd=0.0
            if "fwd_gt" in data:
                next_xyz = data["fwd_gt"]["fwd_valid_gt"]
                next_mask = data["fwd_gt"]["fwd_mask"]
                xyz_next_predicted= xyz +fwd_flow
                val_l2loss += l2_loss(xyz_next_predicted[next_mask],next_xyz[next_mask])
                

            if "bwd_gt" in data:
                pre_xyz = data["bwd_gt"]["bwd_valid_gt"]
                pre_mask = data["bwd_gt"]["bwd_mask"]
                xyz_pre_predicted= xyz +bwd_flow
                val_l2loss += l2_loss(xyz_pre_predicted[pre_mask],pre_xyz[pre_mask])
        val_l2loss/=len(val_dataset)
        writer.add_scalar('val_loss_k2', val_l2loss, global_step=iteration)
    model.train()



def train_table_completion_NIT(args):
    """base on discussion with LIU YUAN 2024年3月24日20

    Args:
        args (_type_): _description_
    """
    from model.neuralsceneflowprior import Neural_InverseTrajectory_Trainer
       # from pytoch_lightning import LightningModel
    

    outdir= check_exist(os.path.join(args.stage1_model_path,args.timestamp))
    
    args.outdir = outdir
    save_cfg(args,outdir,args.timestamp)

    if TENSORBOARD_FOUND:
        writer = SummaryWriter(log_dir=outdir)
    else:
        raise NotImplementedError("TENSORBOARD_FOUND is False")

    trainer = Neural_InverseTrajectory_Trainer(args)
    trainer.training_setup(args)
    
    print(f"Optimizing outdir:{outdir}")
    trainer.to_trainning()
    TableDatasetClass= get_dataset_by_table_dir(args.TimePcd_dir,args.exhaustive_training)
    if args.exhaustive_training:
        table_device="cpu"
    else:
        table_device= "cuda"
    dataset = TableDatasetClass(args,max_points=int(args.max_points_perstep),  do_sale = True,sampling_strategy="random",reserve_for_validation_rate=0.01,device=table_device)
    dataset.filter_firstN(args.filterfirst_N)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0,shuffle=True)
    iteration=trainer.resume_step
    DO_TRAIN = True
    with tqdm(total=args.stage1_max_steps,desc="progress_stage1") as pbar:
        while DO_TRAIN:
            for idx,data in enumerate(dataloader):
                
                if args.exhaustive_training:
                    loss = trainer.train_exhautive_one_step(iteration,data)
                else:
                    loss=trainer.train_one_step(iteration,data)
                trainer.log(iteration,writer)
                
                
                
                if iteration>0 and iteration%args.stage1_validation_step_interval==0:
                    if args.exhaustive_training:
                        trainer.validate_exhaustive_table_completion(dataset,iteration,writer)
                        
                    else:
                        trainer.validate_table_completion(dataset,iteration,writer)    ##  '240326_105658' iteration:254,valloss:0.000409784236884055 
                        ##  '240326_111235' iteration:254,valloss:0.000409784236884055 
        
                
        
                # writer.add_scalar('loss', loss, global_step=iteration)
                # if iteration%100==0:
                #     print(f"iteration:{iteration}, loss:{loss}")
                if iteration>args.stage1_max_steps:
                    DO_TRAIN=False
                    break
                if iteration%args.stage1_save_interval==0:
                    trainer.save_model(iteration)
                    # save_path = os.path.join(outdir, f"Stage1_Model_{timestamp}_step_{iteration}.pth")
                    # torch.save(model.state_dict(), save_path)
                iteration+=1
            pbar.update(len(dataset))
            pbar.set_postfix({ "Flow loss": f"{loss:.{7}f}"})
                
    return trainer 
    
    pass
def train_stage1and2_jointly(args,dataset_args:PointTrackModelParams,opt:PointTrackOptimizationParams, pipe : PipelineParams,net_args:NeuralInverseTrajectory_Params,  testing_iterations, saving_iterations,checkpoint_iterations,checkpoint=None):
    """_summary_

    Args:
        args (_type_): _description_
    """
    from model.neuralsceneflowprior import Neural_InverseTrajectory_Trainer
    from scene import PointTrackScene
    from pointTrack_tools import get_gaussians_init_pcd
    from gaussian_renderer import original_render,render_depth_normal
    from utils.gaussian_training_utils import cotraining_report,prepare_output_and_logger,get_depth_order_loss
    if args.lambda_depth_plane>0:
        print("lambda depth plane loss >0,using Jiepeng Rasterizer.")
        renderFunc = render_depth_normal
    else:
        renderFunc = original_render
    # renderFunc = render_depth_normal
    dataset_args.timestamp= args.timestamp
    dataset_args.gs_model_version= args.gs_model_version
    outdir= check_exist(os.path.join(args.model_path,args.timestamp))
    
    args.outdir = outdir
    save_cfg(args,outdir,args.timestamp)
    writer = prepare_output_and_logger(dataset_args)


    ######
    ###### 
    net_args.timestamp=args.timestamp
    net_args.outdir=args.outdir
    dataset_args.timestamp=args.timestamp
    dataset_args.gs_model_version=args.gs_model_version
    ModelClass=GaussianModelTypes[args.gs_model_version]

    if args.gs_model_version=="TimeTable_GaussianModel":
        
        gaussians =ModelClass(args.TimePcd_dir,table_frame_interval=args.PointTrack_frame_interval)
    elif args.gs_model_version=="PointTrackIsotropicGaussianModel":
        gaussians =ModelClass(dataset_args.sh_degree)
    elif args.gs_model_version=="Original_GaussianModel":
        gaussians =ModelClass(dataset_args.sh_degree)
    else:
        raise NotImplementedError("Not implemented yet")
    ## load time pcd 
    TableDatasetClass= get_dataset_by_table_dir(args.TimePcd_dir,args.exhaustive_training)
    if args.exhaustive_training:
        table_device="cpu"
    else:
        table_device= "cuda"
    timePcddataset = TableDatasetClass(args,keeprgb=True,max_points=int(args.max_points_perstep),  do_sale = True,reserve_for_validation_rate=0.01,device=table_device)
    timePcddataset.filter_firstN(args.filterfirst_N)
    
    dataset_args.exhaustive_training=args.exhaustive_training
    net_trainer = Neural_InverseTrajectory_Trainer(net_args)
    net_trainer.training_setup()
    scene= PointTrackScene(dataset_args, gaussians,timePcd_dataset=timePcddataset,net_trainer=net_trainer,shuffle=False)
    
    if not args.use_Global_NearFar:
        print("near far ",scene.near,scene.far)
        print("set near and far to None")
        scene.near = None
        scene.far = None
    #######
    #######
    print(f"Optimizing outdir:{outdir}")
    ####### initializing GaussianModel Using Canonical Space  InverseMLP
    



    init_pcd = get_gaussians_init_pcd(timePcddataset,net_trainer,number_init_pnts=300000)

    gaussians.create_from_pcd(init_pcd, scene.cameras_extent)
    gaussians.training_setup(opt)
    ## 



    resume_step =1#testing_iterations.remove(15000)

    timePcddataset.clean()
    del init_pcd,
    torch.cuda.empty_cache()
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    black_bg =  torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    net_trainer.to_trainning()
    
    idx_stack = None
    progress_bar = tqdm(range(resume_step, args.stageCoTrain_max_steps), desc="Training progress")
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    ema_loss_for_log = 0.0
    for iteration in range(resume_step,args.stageCoTrain_max_steps+1,1):
        # for idx,data in enumerate(dataloader):
        iter_start.record()    
        # if (iteration+1) % 3000==0 and args.all_SH:
        #     print("SHdegree up")
        #     gaussians.oneupSHdegree()
        # ## NOTE: Canceling SHdegree up
        
        if not idx_stack:
            # viewpoint_stack = scene.getTrainCameras().copy()
            # viewpoint_PCD_stack = copy.deepcopy(scene.getCoTrainingCameras())
            viewpoint_PCD_stack = scene.getCoTrainingCameras() ## FIXME :canceled copy.deepcopy exhaustive pairs occupy too much memory
            if scene.is_overfit_aftergrowth:
                
                idx_stack = scene.getOverfitIdxStack(iteration=iteration)
            else:
                idx_stack = torch.randperm(len(viewpoint_PCD_stack)).tolist()

        idx = idx_stack.pop()
        viewpoint,pcd_pair  = viewpoint_PCD_stack[idx]
        
        #### Predict Gaussian position. 
        xyz= gaussians.get_xyz
        # time = viewpoint_time.unsqueeze(0)(viewpoint.time, pcd_pair["time"])
        time = viewpoint.time.unsqueeze(0)
        
        predicted_xyz= net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
        ###

        ## Render using Gaussian
        bg = torch.rand((3), device="cuda") if args.random_background else background
        render_pkg = renderFunc(viewpoint, gaussians, pipe, bg, override_color = None, specified_xyz=predicted_xyz,
                            )
        
        # render_pkg = original_render(viewpoint, gaussians, pipe, background, override_color = None,specified_xyz=predicted_xyz)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[ "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        render_depth,render_alpha = render_pkg["depth"],render_pkg["alpha"]
        
        depth_mask = viewpoint.depth > 0
        mask = viewpoint.mask
        if mask is not None:
            mask = mask[None].to(image) ##(3,H,w)-->(1,H,w)
        ### Calculate Loss
        gt_image = viewpoint.original_image.cuda()
        Ll1 = opt.lambda_recon*l1_loss(image, gt_image)
        Lssim = opt.lambda_dssim*(1.0 - ssim(image, gt_image))
        loss = Ll1 + Lssim  


        L2d_render_flow= None
            # L2d_render_flow= torch.Tensor([0.0]).cuda()


            

        if opt.lambda_depthOderLoss>0:    
            # depth_mask = viewpoint.depth > 0
            

            LdepthOrder = get_depth_order_loss(render_depth,viewpoint.depth,depth_mask,method_name=args.depth_order_loss_type
                                        ,alpha=args.Alpha_tanh
                                        )
        
            loss += opt.lambda_depthOderLoss*LdepthOrder   
        else:
            LdepthOrder= None

        
        # loss = Ll1 + Lssim + opt.lambda_gs_approx_flow*Lgs_flow +opt.lambda_pcd_flow*Lpcd_flow +\
        #     opt.lambda_depth_plane*Ldepthplane+ opt.lambda_opacity_sparse*LsparseOpacity + opt.lambda_depthloss*Ldepth +\
        #         opt.lambda_2dflowloss*L2d_render_flow
        ### Calculate Loss
        loss.backward()
        
        
        
        iter_end.record()
        loss_dict= {"Ll1":Ll1,"Lssim":Lssim,
                    "LdepthOrder":LdepthOrder,
                    "loss_total":loss}
        
        ## record error information
        if iteration > opt.custom_densification_start and \
            iteration < opt.custom_densification_end:
            info_dict = {"render":image.detach(),"render_depth":render_depth.detach(),"render_alpha":render_alpha.detach(),}
        
        
        with torch.no_grad():

               
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == args.stageCoTrain_max_steps:
                progress_bar.close()
                
            net_trainer.log(iteration,writer) ## log lr of deform and feature mlp   
            # training_report(writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, original_render, (pipe, background))  
            cotraining_report(writer, iteration, loss_dict, iter_start.elapsed_time(iter_end), testing_iterations, scene,renderFunc, (pipe, background))
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset_args.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            ## custom Densification for Mono3D GS region that are underconstructed.

            # Optimizer iteration
            if iteration < args.stageCoTrain_max_steps: ## FUCK YOU 
                # Optimizer step
                # if iteration < opt.iterations:
                #step
                gaussians.optimizer.step()
                net_trainer.optimizer.step()
                ## zero grad
                gaussians.optimizer.zero_grad(set_to_none=True)
                net_trainer.optimizer.zero_grad()
                ## update lr rate
                net_trainer.scheduler.step()
                # net_trainer.update_learning_rate(iteration) TODO: update learning rate   
                gaussians.update_learning_rate(iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path,args.timestamp) + "/chkpnt" + str(iteration) + ".pth")
                net_trainer.save_model(iteration)
            
            
        iteration+=1
    # return trainer
    try: 
        evaluation_on_metricCam( scene,net_trainer,gaussians,args,pipe,renderFunc,black_bg)
    except Exception  as e:
        pass
    pass 
   
def  renderview_stage1and2(args,dataset_args:PointTrackModelParams,opt:PointTrackOptimizationParams, pipe : PipelineParams,net_args:NeuralInverseTrajectory_Params,checkpoint=None):
    """_summary_

    Args:
        args (_type_): _description_
    """
    from model.neuralsceneflowprior import Neural_InverseTrajectory_Trainer
    from scene import PointTrackScene
    from pointTrack_tools import get_gaussians_init_pcd
    from gaussian_renderer import original_render,render_depth_normal

    from utils.graphics_utils import get_intrinsic_matrix,fov2focal
    
    import kornia
    if args.lambda_depth_plane>0:
        print("lambda depth plane loss >0,using Jiepeng Rasterizer.")
        renderFunc = render_depth_normal
    else:
        renderFunc = original_render
    # renderFunc = render_depth_normal
    
    args.outdir = os.path.dirname(checkpoint)
    outdir= args.outdir
    dataset_args.Factor_ColmapDepthAlign =args.Factor_ColmapDepthAlign  ## 2024年5月12日完与LIU YUAN讨论之后的结果。
    ######
    ###### 
    # net_args.timestamp=args.timestamp
    net_args.outdir=args.outdir
    dataset_args.timestamp=args.timestamp
    dataset_args.gs_model_version=args.gs_model_version
    ModelClass=GaussianModelTypes[args.gs_model_version]

    if args.gs_model_version=="TimeTable_GaussianModel":
        
        gaussians =ModelClass(args.TimePcd_dir,table_frame_interval=args.PointTrack_frame_interval)
    elif args.gs_model_version=="PointTrackIsotropicGaussianModel":
        gaussians =ModelClass(dataset_args.sh_degree)
    elif args.gs_model_version=="Original_GaussianModel":
        gaussians =ModelClass(dataset_args.sh_degree)
    else:
        raise NotImplementedError("Not implemented yet")

    dataset_args.exhaustive_training=False ## only for test
    net_trainer = Neural_InverseTrajectory_Trainer(net_args)
    net_trainer.training_setup()
    # scene= PointTrackScene(dataset_args, gaussians,timePcd_dataset=timePcddataset,net_trainer=net_trainer,shuffle=False)
    scene= PointTrackScene(dataset_args, gaussians,timePcd_dataset=None,net_trainer=net_trainer,shuffle=False)
    
    if not args.use_Global_NearFar:
        print("near far ",scene.near,scene.far)
        print("set near and far to None")
        scene.near = None
        scene.far = None
    #######
    #######
    if checkpoint: 
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        max_iter_path,iter = net_trainer.find_max_iter_ckpt(os.path.dirname(checkpoint))
        net_trainer.load_model(ckpt_path=max_iter_path,load_optimizer=False)

    # timePcddataset.clean()
    torch.cuda.empty_cache()
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    black_bg =  torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    net_trainer.to_eval()
    train_cams = scene.getTrainCameras()
    test_cams = scene.getMetricTestCameras()
    
    scene_name = args.source_path.rstrip("/").split("/")[-2:-1]
    import pickle
    

    bg = torch.ones_like(black_bg)

    ### get intrinsic matrix
    width = train_cams[0].image_width
    height = train_cams[0].image_height
    focal_lengthX = fov2focal(train_cams[0].FoVx,width)
    focal_lengthY = fov2focal(train_cams[0].FoVy,height)
    print("Focals:",focal_lengthX,focal_lengthY)
    print("Width Height:",width,height)
    intrinsic=get_intrinsic_matrix(width=width, height= height, focal_length=focal_lengthX )
    intrinsic = torch.Tensor(intrinsic).to("cuda")
    if not os.path.exists(os.path.join(outdir,"metric_test_res")):
        os.makedirs(os.path.join(outdir,"metric_test_res"))
    print("out_dir:",outdir)
    postfix = ".jpg" if "nvidia" in args.source_path else ".png"
    
    ## FIX bugs
    big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * 1
    prune_mask =  big_points_ws
    gaussians.prune_points(prune_mask)
    print("==============Prune mask:",prune_mask.sum())
    ## FIX bugs
    
    
    with torch.no_grad():
        bg_pcd_list = []

            
        img_list = []
        depth_list = []
        for  cam  in tqdm(train_cams,desc="testing training cams"):                
            #### Predict Gaussian position. 
            xyz= gaussians.get_xyz
            # time = pcd_pair["time"]
            # time = viewpoint_time.unsqueeze(0)(cam.time,None)
            time = cam.time.unsqueeze(0)
            predicted_xyz= net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
            ###

        

            render_pkg = renderFunc(cam, gaussians, pipe, bg, \
                                    override_color = None, specified_xyz=predicted_xyz,)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[ "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            render_depth,render_alpha = render_pkg["depth"],render_pkg["alpha"]
            image_name = cam.image_name
            print("depth",render_depth.max()," ",render_depth.min())
            

            depth = torch.clamp(minmax_norm(render_depth,render_depth.quantile(0.01),render_depth.quantile(0.98)),0,1)
            
            img_list.append(tob8(torch.clamp(image.permute(1,2,0),0,1).cpu().numpy()))
            depth_list.append(tob8(depth.cpu().numpy()))
            alpha_msk = render_alpha>0.5
            time_folder = "%05d"%int(image_name[-4:])
            if not os.path.exists(os.path.join(outdir,f"trainview_len{len(train_cams)}")):
                os.makedirs(os.path.join(outdir,f"trainview_len{len(train_cams)}"))
            Image.fromarray(tob8(torch.clamp(image.permute(1,2,0),0,1).cpu().numpy())).save(os.path.join(outdir,f"trainview_len{len(train_cams)}",f"{time_folder}"+postfix))
            Image.fromarray(tob8(torch.clamp(depth,0,1).cpu().numpy())).save(os.path.join(outdir,f"trainview_len{len(train_cams)}",f"{time_folder}_depth"+postfix))
            Image.fromarray(tob8(torch.clamp(alpha_msk.to(torch.float32)[0],0,1).cpu().numpy())).save(os.path.join(outdir,f"trainview_len{len(train_cams)}",f"{time_folder}_alphaMsk"+postfix))
        
        
        imageio.mimsave(os.path.join(outdir,f"trainview_len{len(train_cams)}",f"render_depth.mp4"), depth_list,fps=10)
        print("save depth video to:",os.path.join(outdir,f"trainview_len{len(train_cams)}",f"render_depth.mp4"))
            # os.mkdir()
    
    

            
            

def evaluation_on_metricCam( scene,net_trainer,gaussians,args,pipe,renderFunc,black_bg):
    outdir= args.outdir
    
    net_trainer.to_eval()
    train_cams = scene.getTrainCameras()
    test_cams = scene.getMetricTestCameras()
    
    ### get intrinsic matrix
    width = train_cams[0].image_width
    height = train_cams[0].image_height
    # focal_lengthX = fov2focal(train_cams[0].FoVx,width)
    # focal_lengthY = fov2focal(train_cams[0].FoVy,height)
    # print("Focals:",focal_lengthX,focal_lengthY)
    # print("Width Height:",width,height)
    # intrinsic=get_intrinsic_matrix(width=width, height= height, focal_length=focal_lengthX )
    # intrinsic = torch.Tensor(intrinsic).to("cuda")
    if not os.path.exists(os.path.join(outdir,"metric_test_res")):
        os.makedirs(os.path.join(outdir,"metric_test_res"))
    print("out_dir:",outdir)
    postfix = ".jpg" if "nvidia" in args.source_path else ".png"
    
            ## FIX bugs
    big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * 1
    prune_mask =  big_points_ws
    gaussians.prune_points(prune_mask)
    print("==============Prune mask:",prune_mask.sum())
    ## FIX bugs
    with torch.no_grad():
        bg_pcd_list = []
        for key in list(test_cams.keys()):
            test_cams_x = test_cams[key] 
            for  cam  in tqdm(test_cams_x,desc="testing metric cams"):                
                #### Predict Gaussian position. 
                xyz= gaussians.get_xyz
                # time = pcd_pair["time"]
                time = cam.time.unsqueeze(0)(cam.time,None)
                predicted_xyz= net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
                ###

                ## Render using Gaussian
                bg = black_bg
                render_pkg = renderFunc(cam, gaussians, pipe, bg, \
                                        override_color = None, specified_xyz=predicted_xyz,)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[ "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                render_depth,render_alpha = render_pkg["depth"],render_pkg["alpha"]
                image_name = cam.image_name
                
                alpha_msk = render_alpha>0.5
                # image =image*alpha_msk.to(torch.float32)
                time_folder = "%05d"%int(image_name[-4:])
                if not os.path.exists(os.path.join(outdir,f"metric_test_res_factor{args.Factor_ColmapDepthAlign}_len{len(test_cams.keys())}",time_folder)):
                    os.makedirs(os.path.join(outdir,f"metric_test_res_factor{args.Factor_ColmapDepthAlign}_len{len(test_cams.keys())}",time_folder))
                Image.fromarray(tob8(image.permute(1,2,0).cpu().numpy())).save(os.path.join(outdir,f"metric_test_res_factor{args.Factor_ColmapDepthAlign}_len{len(test_cams.keys())}",time_folder,f"{key}"+postfix))
                Image.fromarray(tob8(torch.clamp(alpha_msk.to(torch.float32)[0],0,1).cpu().numpy())).save(os.path.join(outdir,f"metric_test_res_factor{args.Factor_ColmapDepthAlign}_len{len(test_cams.keys())}",time_folder,f"alphaMsk_{key}"+postfix))

            # os.mkdir()
    os.system("/data/qingmingliu/Software/anaconda3/envs/copy0422_d3dgs2309/bin/python evaluate_metric.py --pre_dir {} --target_dir {}".format(os.path.join(outdir,f"metric_test_res_factor{args.Factor_ColmapDepthAlign}_len{len(test_cams.keys())}"),os.path.join(args.source_path,"mv_images")))

    
    pass
def test_stage1and2_Metric(args,dataset_args:PointTrackModelParams,opt:PointTrackOptimizationParams, pipe : PipelineParams,net_args:NeuralInverseTrajectory_Params,checkpoint=None):
    """_summary_

    Args:
        args (_type_): _description_
    """
    from model.neuralsceneflowprior import Neural_InverseTrajectory_Trainer
    from scene import PointTrackScene
    from gaussian_renderer import original_render,render_depth_normal
    from utils.graphics_utils import get_intrinsic_matrix,fov2focal
    
    import kornia
    if args.lambda_depth_plane>0:
        print("lambda depth plane loss >0,using Jiepeng Rasterizer.")
        renderFunc = render_depth_normal
    else:
        renderFunc = original_render
    # renderFunc = render_depth_normal
    
    args.outdir = os.path.dirname(checkpoint)
    outdir= args.outdir
    dataset_args.Factor_ColmapDepthAlign =args.Factor_ColmapDepthAlign  ## 2024年5月12日完与LIU YUAN讨论之后的结果。
    ######
    ###### 
    # net_args.timestamp=args.timestamp
    net_args.outdir=args.outdir
    dataset_args.timestamp=args.timestamp
    dataset_args.gs_model_version=args.gs_model_version
    ModelClass=GaussianModelTypes[args.gs_model_version]

    if args.gs_model_version=="TimeTable_GaussianModel":
        
        gaussians =ModelClass(args.TimePcd_dir,table_frame_interval=args.PointTrack_frame_interval)
    elif args.gs_model_version=="PointTrackIsotropicGaussianModel":
        gaussians =ModelClass(dataset_args.sh_degree)
    elif args.gs_model_version=="Original_GaussianModel":
        gaussians =ModelClass(dataset_args.sh_degree)
    else:
        raise NotImplementedError("Not implemented yet")
    ## load time pcd 
    # TableDatasetClass= get_dataset_by_table_dir(args.TimePcd_dir,args.exhaustive_training)
    # if args.exhaustive_training:
    #     table_device="cpu"
    # else:
    #     table_device= "cuda"
    # timePcddataset = TableDatasetClass(args,keeprgb=True,max_points=int(args.max_points_perstep),  do_sale = True,reserve_for_validation_rate=0.01,device=table_device)
    # timePcddataset.filter_firstN(args.filterfirst_N)
    
    dataset_args.exhaustive_training=False ## only for test
    net_trainer = Neural_InverseTrajectory_Trainer(net_args)
    net_trainer.training_setup()
    # scene= PointTrackScene(dataset_args, gaussians,timePcd_dataset=timePcddataset,net_trainer=net_trainer,shuffle=False)
    scene= PointTrackScene(dataset_args, gaussians,timePcd_dataset=None,net_trainer=net_trainer,shuffle=False)
    
    if not args.use_Global_NearFar:
        print("near far ",scene.near,scene.far)
        print("set near and far to None")
        scene.near = None
        scene.far = None
    #######
    #######
    if checkpoint: 
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        max_iter_path,iter = net_trainer.find_max_iter_ckpt(os.path.dirname(checkpoint))
        net_trainer.load_model(ckpt_path=max_iter_path,load_optimizer=False)

    # timePcddataset.clean()
    torch.cuda.empty_cache()
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    black_bg =  torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    net_trainer.to_eval()
    train_cams = scene.getTrainCameras()
    test_cams = scene.getMetricTestCameras()
    
    scene_name = args.source_path.rstrip("/").split("/")[-2:-1]
    import pickle
    
    # Export metric cam pkl file
    # print(train_cams[0].camera_center)
    # with open(os.path.join(args.source_path,f"A_metric_cam_.pkl"), 'wb') as f:
    #     pickle.dump(test_cams, f)    
    # return 
    ## FIX bugs
    big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * 1
    prune_mask =  big_points_ws
    gaussians.prune_points(prune_mask)
    print("==============Prune mask:",prune_mask.sum())
    ## FIX bugs
    # bg = torch.ones_like(black_bg)
    bg = torch.zeros_like(black_bg)

    ### get intrinsic matrix
    width = train_cams[0].image_width
    height = train_cams[0].image_height
    focal_lengthX = fov2focal(train_cams[0].FoVx,width)
    focal_lengthY = fov2focal(train_cams[0].FoVy,height)
    print("Focals:",focal_lengthX,focal_lengthY)
    print("Width Height:",width,height)
    intrinsic=get_intrinsic_matrix(width=width, height= height, focal_length=focal_lengthX )
    intrinsic = torch.Tensor(intrinsic).to("cuda")
    if not os.path.exists(os.path.join(outdir,"metric_test_res")):
        os.makedirs(os.path.join(outdir,"metric_test_res"))
    print("out_dir:",outdir)
    postfix = ".jpg" if "nvidia" in args.source_path else ".png"
    with torch.no_grad():
        bg_pcd_list = []
        for key in list(test_cams.keys()):
            test_cams_x = test_cams[key] 
            img_list = []
            depth_list = []
            for  cam  in tqdm(test_cams_x,desc="testing metric cams"):                
                #### Predict Gaussian position. 
                xyz= gaussians.get_xyz
                # time = pcd_pair["time"]
                time = cam.time.unsqueeze(0)(cam.time,None)
                predicted_xyz= net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
                ###

            

                render_pkg = renderFunc(cam, gaussians, pipe, bg, \
                                        override_color = None, specified_xyz=predicted_xyz,)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[ "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                render_depth,render_alpha = render_pkg["depth"],render_pkg["alpha"]
                
                print(render_depth.max()," =====",render_depth.min())
                depth = torch.clamp(minmax_norm(render_depth,torch.quantile(render_depth,0.01),1.2*torch.quantile(render_depth,0.98)),0,1) ###TODO: 为了ablation visualization
                
                image_name = cam.image_name
                img_list.append(tob8(torch.clamp(image.permute(1,2,0),0,1).cpu().numpy()))
                depth_list.append(tob8(torch.clamp(depth[0],0,1).cpu().numpy()))
                alpha_msk = render_alpha>0.5
                # image =image*alpha_msk.to(torch.float32)
                time_folder = "%05d"%int(image_name[-4:])
                if not os.path.exists(os.path.join(outdir,f"metric_test_res_factor{args.Factor_ColmapDepthAlign}_len{len(test_cams.keys())}",time_folder)):
                    os.makedirs(os.path.join(outdir,f"metric_test_res_factor{args.Factor_ColmapDepthAlign}_len{len(test_cams.keys())}",time_folder))
                Image.fromarray(tob8(torch.clamp(image.permute(1,2,0),0,1).cpu().numpy())).save(os.path.join(outdir,f"metric_test_res_factor{args.Factor_ColmapDepthAlign}_len{len(test_cams.keys())}",time_folder,f"{key}"+postfix))
                Image.fromarray(tob8(torch.clamp(alpha_msk.to(torch.float32)[0],0,1).cpu().numpy())).save(os.path.join(outdir,f"metric_test_res_factor{args.Factor_ColmapDepthAlign}_len{len(test_cams.keys())}",time_folder,f"alphaMsk_{key}"+postfix))
            
            

    
    

        

    
    pass 
if __name__=="__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_start_method('spawn')
    import configargparse
    parser = configargparse.ArgumentParser(description="Training script parameters")## LQM

    # mlp = MLP_Params(parser)

    parser.add_argument('--comments', type=str, default="None")
    parser.add_argument('--stage', type=str, default="stage1",choices=["stage1","stage2","stage12_cotraining"])
    parser.add_argument('--mode', type=str, default="Train",choices=["train","test","test_metric_cams","render_train"],help="train or test")
    parser.add_argument('--timestamp', type=str, default="None",help="given time stamp to load ckpt when test")
    parser.add_argument('--stage1_model', type=str, default="NeuralInverseTrajectory",help="Stage1 Table Completion Model Type")
    # parser.add_argument('--stage2_model', type=str, default="PointTackGaussian",help="Stage2 Gaussian model  Model Type")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6011)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        # default=[1,5000 ,20000,30000] )
                        default=[1,100,3000,5000, 7_000,15000,20000,25000,30000] )
                        # default=[1,100,1000,3000,5000, 6000, 7_000,15000,25000,30000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 20_000, 30_000, 40000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000,30000,50000,100000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint",type=str,default=None)
    parser.add_argument("--Factor_ColmapDepthAlign",type=float,default=1.0)
    args = parser.parse_args(sys.argv[1:7]) ## NOTE: load stage1_mode sys.argv[3:] is used to avoid the conflict 
    if args.stage1_model =="NSFF":
        mlp = MLP_Params(parser)
    elif args.stage1_model =="NeuralInverseTrajectory":
        mlp= NeuralInverseTrajectory_Params(parser)
    if args.stage == "stage2" or  args.stage == "stage12_cotraining" :
        parser.add_argument('--stageCoTrain_max_steps', type=int, default=30000)
        lp = PointTrackModelParams(parser)
        op = PointTrackOptimizationParams(parser)
        pp = PipelineParams(parser)
    parser.add_argument('--config', is_config_file_arg=True, help='config file path')
    ## paser again
    args = parser.parse_args(sys.argv[1:]) ## NOTE: load all config
    
    if args.timestamp=="None":
        timestamp=get_timestamp()
        args.timestamp=timestamp
    
    safe_state(False,1234) ## set randomseed to 1234
    ## stage1 
    if args.mode=="train":
        if not hasattr(args,"model_path"):
            args.model_path = os.path.join(args.stage1_model_path,args.timestamp)
        make_source_code_snapshot(os.path.join(args.model_path,args.timestamp))
        # network_gui.init(args.ip, args.port)
        torch.autograd.set_detect_anomaly(args.detect_anomaly)    
    if args.stage == "stage1" :

        if args.mode=="train":
            if args.stage1_model =="NeuralInverseTrajectory":
                train_table_completion_NIT(args)
            else:
                raise NotImplementedError(f"{args.stage1_model} is not implemented yet")
            
            

    elif args.stage == "stage12_cotraining" :
        if args.mode=="train":
            train_stage1and2_jointly(args,lp.extract(args), op.extract(args), pp.extract(args),mlp.extract(args),  testing_iterations=args.test_iterations, saving_iterations=args.save_iterations,checkpoint_iterations=args.checkpoint_iterations,checkpoint=args.checkpoint)
        elif args.mode=="test_metric_cams":
            test_stage1and2_Metric(args,lp.extract(args), op.extract(args), pp.extract(args),mlp.extract(args),checkpoint=args.checkpoint)
        elif args.mode=="render_train":
            renderview_stage1and2(args,lp.extract(args), op.extract(args), pp.extract(args),mlp.extract(args),checkpoint=args.checkpoint)
        # raise NotImplementedError("stage3 is not implemented yet")
    