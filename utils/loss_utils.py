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
import pytorch3d
import torch
from pytorch3d.ops import knn_gather, knn_points
import torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
# from torchmetrics.functional.regression import pearson_corrcoef
from utils.pearson_coeff import pearson_corrcoef

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
####### from torchmetrics



def get_depthloss(render_depth,gt_depth,mask,method_name="pearson"):
    # if mask is not None:
    #     gt_depth=gt_depth*mask
    #     render_depth=render_depth*mask
    if method_name=="pearson":
        return pearson_corrcoef(render_depth,gt_depth)
    elif method_name=="pearson_fsgs":
        render_depth = render_depth.reshape(-1, 1)
        gt_depth = gt_depth.reshape(-1, 1)
        depth_loss = min(
                        (1 - pearson_corrcoef( - gt_depth, render_depth)),
                        (1 - pearson_corrcoef(1 / (gt_depth + 200.), render_depth))
        )
        return depth_loss
    elif method_name=="pearson_fsgs_selectmasked":
            # if mask is not None:
        gt_depth=gt_depth[mask[0]>0]
        render_depth=render_depth[mask>0]
        render_depth = render_depth.reshape(-1, 1)
        gt_depth = gt_depth.reshape(-1, 1)
        depth_loss = min(
                        (1 - pearson_corrcoef( - gt_depth, render_depth)),
                        (1 - pearson_corrcoef(1 / (gt_depth + 200.), render_depth))
        )
        return depth_loss
    elif method_name=="pearson_metric_depth_selectmasked":
            # if mask is not None:
        gt_depth=gt_depth[mask[0]>0]
        render_depth=render_depth[mask>0]
        render_depth = render_depth.reshape(-1, 1)
        gt_depth = gt_depth.reshape(-1, 1)
        depth_loss = 1 - pearson_corrcoef( gt_depth, render_depth)
        return depth_loss
    elif method_name=="l1":
        depth_mask = gt_depth > 0
        return l1_loss(render_depth.squeeze()[depth_mask],gt_depth[depth_mask])
    elif method_name=="l2":
        depth_mask = gt_depth > 0
        return l2_loss(render_depth.squeeze()[depth_mask],gt_depth[depth_mask])
        # return l2_loss(render_depth,gt_depth)
    else:
        raise ValueError("Unknown method_name")
    # pass
def ssimmap(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssimmap(img1, img2, window, window_size, channel, size_average)


def _ssimmap(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map    

def mask_l1_loss(network_output, gt,mask):
    return torch.abs((network_output - gt)*mask).mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
def mask_ssim(img1, img2,mask, window_size=11, size_average=True):
    channel = img1.size(-3)
    img1=img1*mask
    img2=img2*mask
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def msssim(rgb, gts):
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(rgb, gts).item()
    
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
if __name__ ==  "__main__":
    
    N = 100
    pcd= torch.randn([N,3])
    flow = torch.randn([N,3])
    # lsl=LocalSmoothnessLoss()
    # lsl(pcd,flow,10)
    
    pcd=torch.cat([torch.linspace(1,10,10).unsqueeze(1),torch.linspace(1,10,10).unsqueeze(1),torch.linspace(1,10,10).unsqueeze(1)],dim=1)
    res =lsl(pcd,pcd,2)