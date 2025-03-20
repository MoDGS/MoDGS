



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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, msssim
from lpipsPyTorch import lpips_helper
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision.transforms as transforms
from PIL import Image


def evaluate_nvidia(render_path,gt_path,mask_path =None):
    time_folder =os.listdir(render_path)
    transform = transforms.ToTensor()
    toPil = transforms.ToPILImage()
    lpips = lpips_helper(net_type='vgg')
    
    ssim_dict = {}
    lpips_dict = {}
    psnr_dict = {}
    psnr_all =0.0
    ssim_all =0.0
    lpips_all =0.0
    for time in tqdm(time_folder):
        render_folder =os.path.join(render_path,time)
        
        for img_name in os.listdir(render_folder):
            if img_name.startswith("error") or img_name.startswith("alpha") or img_name.startswith("depth"):
                continue
        
            pre_img = transform(Image.open(os.path.join(render_folder,img_name)))
            _,H,W = pre_img.shape
            gt_img  = transform(Image.open(os.path.join(gt_path,time,img_name)).resize([W,H]) )   
            if mask_path is  None:
                
                if os.path.exists(os.path.join(render_folder,"alphaMsk_"+img_name)):
                    # print("alphaMsk_"+img_name)
                    alpha_img = Image.open(os.path.join(render_folder,"alphaMsk_"+img_name))
                    alpha = transform(alpha_img)
                    pre_img = pre_img*alpha
                    gt_img = gt_img*alpha
            else:
                print("use mask")
                if os.path.exists(os.path.join(mask_path,time,"alphaMsk_"+img_name)):
                    # print("alphaMsk_"+img_name)
                    alpha_img = Image.open(os.path.join(mask_path,time,"alphaMsk_"+img_name))
                    alpha = transform(alpha_img)
                    pre_img = pre_img*alpha
                    gt_img = gt_img*alpha
                else:
                    print("no mask,skip this image")
                    continue
                
                pass

            error  =toPil(torch.abs(pre_img-gt_img))     
            error.save(os.path.join(render_path,time,"error"+img_name))
            if os.path.exists(os.path.join(gt_path,time,"error"+img_name)):
                os.remove(os.path.join(gt_path,time,"error"+img_name))
            cur_psnr = psnr(pre_img, gt_img).mean().double().item()
            cur_ssim = ssim(pre_img, gt_img).mean().double().item()
            cur_lpips = lpips(pre_img.cuda(), gt_img.cuda()).mean().double().item()
            psnr_all+=cur_psnr
            ssim_all+=cur_ssim
            lpips_all+=cur_lpips    
            psnr_dict[img_name+time] = cur_psnr
            ssim_dict[img_name+time] = cur_ssim
            lpips_dict[img_name+time] = cur_lpips
              
    psnr_all = psnr_all/len(psnr_dict)
    ssim_all = ssim_all/len(ssim_dict)
    lpips_all = lpips_all/len(lpips_dict)
    
    print("  SSIM   : {:>12.7f}".format(torch.tensor(ssim_all).mean(), ".5"))

    print("  PSNR   : {:>12.7f}".format(torch.tensor(psnr_all).mean(), ".5"))
    print("  LPIPS  : {:>12.7f}".format(torch.tensor(lpips_all).mean(), ".5"))
    import json 
    with open(os.path.join(os.path.join(render_path,"../"),"metric_results.json"), 'w') as fp:
        json.dump({"SSIM":ssim_all,"PSNR":psnr_all,"LPIPS":lpips_all}, fp, indent=True)
    with open(os.path.join(os.path.join(render_path,"../"),"metric_psnr.json"), 'w') as fp:
        json.dump(psnr_dict, fp, indent=True)
    with open(os.path.join(os.path.join(render_path,"../"),"metric_ssim.json"), 'w') as fp:
        json.dump(ssim_dict, fp, indent=True)
    with open(os.path.join(os.path.join(render_path,"../"),"metric_lpips.json"), 'w') as fp:
        json.dump(lpips_dict, fp, indent=True)
    
    
    return psnr_all,psnr_dict
    
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    lpips = lpips_helper(net_type='vgg')
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                msssims = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    msssims.append(msssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx]))

                print("  SSIM   : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  MS-SSIM: {:>12.7f}".format(torch.tensor(msssims).mean(), ".5"))
                print("  PSNR   : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS  : {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                     "MS-SSIM": torch.tensor(msssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                         "MS-SSIM": {name: ssim for ssim, name in zip(torch.tensor(msssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(e)
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--pre_dir', type=str, default="None")
    parser.add_argument('--target_dir', type=str, default="None")
    parser.add_argument('--msk_dir', type=str, default="None")
    args = parser.parse_args()
    # evaluate(args.model_paths)
    print("eval metric args.", args)
    
    evaluate_nvidia(
       args.pre_dir,
       args.target_dir,
       None if args.msk_dir=="None" else args.msk_dir
        # "/data/qingmingliu/Dataset/dynamic/Selfmade_nvidia_short/nvidia_data_full/Balloon2-2/dense/mv_images/",
    )
    # evaluate_nvidia(
    #     "output/001PointTrackGS/IphoneDataset/Training_res/PointTrackExhaustivePair/selfmade_nvidia/balloon2-2/balloon2-2_tanh_origianlGS_RandBG_240510_114500_filtered_ExhaustivePair_CoTraining/240511_121924/metric_test_res",
    #     "/data/qingmingliu/Dataset/dynamic/Selfmade_nvidia_short/nvidia_data_full/Balloon2-2/dense/mv_images/",
    # )
