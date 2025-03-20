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



            
            
            

def Render_vis_cams(args,dataset_args:PointTrackModelParams,opt:PointTrackOptimizationParams, pipe : PipelineParams,net_args:NeuralInverseTrajectory_Params,checkpoint=None):
    """_summary_

    Args:
        args (_type_): _description_
    """
    from model.neuralsceneflowprior import Neural_InverseTrajectory_Trainer
    from scene import PointTrackScene
    from gaussian_renderer import original_render,render_depth_normal
    from utils.graphics_utils import get_intrinsic_matrix,unproject_from_depthmap_torch,project_from_rgbpcd,fov2focal
    
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
    test_cams = scene.getVisCameras()
    # test_cams = scene.getTrainCameras()
    
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
        alpha_list = []
        for  idx,cam  in tqdm(enumerate(test_cams),desc="render vis cams"):                
            #### Predict Gaussian position. 
            xyz= gaussians.get_xyz
            # time = pcd_pair["time"]
            # print("cam.time",cam.time)
            if isinstance(cam.time,torch.Tensor):
                pass 
            else:
                cam.time = torch.tensor(cam.time).to("cuda")
            # time = scene.rescale_time(cam.time,None)
            time = cam.time.unsqueeze(0)
            predicted_xyz= net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
            ###

            ## Render using Gaussian
            bg = white_bg
            # bg = black_bg
            render_pkg = renderFunc(cam, gaussians, pipe, bg, \
                                    override_color = None, specified_xyz=predicted_xyz,)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[ "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            render_depth,render_alpha = render_pkg["depth"],render_pkg["alpha"]
            image_name = cam.image_name
            
            alpha_msk = render_alpha>0.5
            depth = torch.clamp(minmax_norm(render_depth,torch.quantile(render_depth,0.01),torch.quantile(render_depth,0.99)),0,1) ###TODO: 为了ablation visualization

            time_folder = "%05d"%idx
            if not os.path.exists(os.path.join(outdir,f"viscam_len{len(test_cams)}")):
                os.makedirs(os.path.join(outdir,f"viscam_len{len(test_cams)}"))
            b8_image = tob8(torch.clamp(image.permute(1,2,0),0,1).cpu().numpy())
            # Image.fromarray(b8_image).save(os.path.join(outdir,f"viscam_len{len(test_cams)}",time_folder+postfix))
            b8_depth = tob8(torch.clamp(depth.to(torch.float32)[0],0,1).cpu().numpy())
            # Image.fromarray(b8_depth).save(os.path.join(outdir,f"viscam_len{len(test_cams)}",time_folder+f"_depth"+postfix))
            b8_alpha = tob8(torch.clamp(alpha_msk.to(torch.float32)[0],0,1).cpu().numpy())
            # Image.fromarray(b8_alpha).save(os.path.join(outdir,f"viscam_len{len(test_cams)}",time_folder+f"_alphaMsk"+postfix))
            img_list.append(b8_image)
            depth_list.append(b8_depth)
            alpha_list.append(b8_alpha)

        scene_name = os.path.basename(args.source_path.rstrip("/"))
        scene_name =""
        file_name = "".join(outdir.split("/")[-2:])

        out_folder = os.path.join("output/DemoFolder/spiral_demo")

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
    
        imageio.mimsave(os.path.join(out_folder,scene_name+file_name[:30]+"len{}_rgb.mp4".format(len(test_cams))), img_list, fps=10,)
        imageio.mimsave(os.path.join(out_folder,scene_name+file_name[:30]+"len{}_alpha.mp4".format(len(test_cams))), alpha_list, fps=10)
        imageio.mimsave(os.path.join(out_folder,scene_name+file_name[:30]+"len{}_depth.mp4".format(len(test_cams))), depth_list, fps=10)

        print("Saved mp4 at ",os.path.join(out_folder,"viscam_len{}_rgb.mp4".format(len(test_cams))) )


    
    pass 
if __name__=="__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_start_method('spawn')
    import configargparse
    parser = configargparse.ArgumentParser(description="Training script parameters")## LQM

    # mlp = MLP_Params(parser)

    parser.add_argument('--comments', type=str, default="None")
    parser.add_argument('--stage', type=str, default="stage1",choices=["stage1","stage2","stage12_cotraining"])
    parser.add_argument('--mode', type=str, default="Train",choices=["train","test","test_metric_cams"],help="train or test")
    parser.add_argument('--timestamp', type=str, default="None",help="given time stamp to load ckpt when test")
    parser.add_argument('--stage1_model', type=str, default="NeuralInverseTrajectory",help="Stage1 Table Completion Model Type")
    # parser.add_argument('--stage2_model', type=str, default="PointTackGaussian",help="Stage2 Gaussian model  Model Type")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6011)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1,100,3000,5000, 7_000,15000,25000,30000] )
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


    # if args.mode=="train":
    #     train_stage1and2_jointly(args,lp.extract(args), op.extract(args), pp.extract(args),mlp.extract(args),  testing_iterations=args.test_iterations, saving_iterations=args.save_iterations,checkpoint_iterations=args.checkpoint_iterations,checkpoint=args.checkpoint)
    # elif args.mode=="test":
    #     test_stage1and2(args,lp.extract(args), op.extract(args), pp.extract(args),mlp.extract(args),checkpoint=args.checkpoint)
    # elif args.mode=="test_metric_cams":
    #     test_stage1and2_Metric(args,lp.extract(args), op.extract(args), pp.extract(args),mlp.extract(args),checkpoint=args.checkpoint)
    # elif args.mode=="render_view":
    #     renderview_stage1and2(args,lp.extract(args), op.extract(args), pp.extract(args),mlp.extract(args),checkpoint=args.checkpoint)
    Render_vis_cams(args,lp.extract(args), op.extract(args), pp.extract(args),mlp.extract(args),checkpoint=args.checkpoint)
        # raise NotImplementedError("stage3 is not implemented yet")
    