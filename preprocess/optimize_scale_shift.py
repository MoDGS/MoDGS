from glob import glob

import sys
import torch
import numpy as np
import torch.nn as nn
import os
from PIL import Image
import json
def sobel_by_quantile(img_points: np.ndarray, q: float):
    """Return a boundary mask where 255 indicates boundaries (where gradient is
    bigger than quantile).
    """
    dx0 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[1:-1, :-2], axis=-1
    )
    dx1 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[1:-1, 2:], axis=-1
    )
    dy0 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[:-2, 1:-1], axis=-1
    )
    dy1 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[2:, 1:-1], axis=-1
    )
    dx01 = (dx0 + dx1) / 2
    dy01 = (dy0 + dy1) / 2
    dxy01 = np.linalg.norm(np.stack([dx01, dy01], axis=-1), axis=-1)

    # (H, W, 1) uint8
    boundary_mask = (dxy01 > np.quantile(dxy01, q)).astype(np.float32)
    boundary_mask = (
        np.pad(boundary_mask, ((1, 1), (1, 1)), constant_values=False)[
            ..., None
        ].astype(np.uint8)
        * 255
    )
    return boundary_mask

# A reimplemented version in public environments by Xiao Fu and Mu Hu
def init_image_coor(height, width):
    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x[np.newaxis, :, :]
    x = x.astype(np.float32)
    x = torch.from_numpy(x.copy()).cuda()
    u_u0 = x - width/2.0

    y_col = np.arange(0, height)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y[np.newaxis, :, :]
    y = y.astype(np.float32)
    y = torch.from_numpy(y.copy()).cuda()
    v_v0 = y - height/2.0
    return u_u0, v_v0


def depth_to_xyz(depth, focal_length):
    b, c, h, w = depth.shape
    u_u0, v_v0 = init_image_coor(h, w)
    x = u_u0 * depth / focal_length
    y = v_v0 * depth / focal_length
    z = depth
    pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
    # print(pw.shape)
    return pw


def get_surface_normal(xyz, patch_size=3):
    # xyz: [1, h, w, 3]
    x, y, z = torch.unbind(xyz, dim=3)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 0)
    z = torch.unsqueeze(z, 0)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    patch_weight = torch.ones((1, 1, patch_size, patch_size), requires_grad=False).cuda()
    xx_patch = nn.functional.conv2d(xx, weight=patch_weight, padding=int(patch_size / 2))
    yy_patch = nn.functional.conv2d(yy, weight=patch_weight, padding=int(patch_size / 2))
    zz_patch = nn.functional.conv2d(zz, weight=patch_weight, padding=int(patch_size / 2))
    xy_patch = nn.functional.conv2d(xy, weight=patch_weight, padding=int(patch_size / 2))
    xz_patch = nn.functional.conv2d(xz, weight=patch_weight, padding=int(patch_size / 2))
    yz_patch = nn.functional.conv2d(yz, weight=patch_weight, padding=int(patch_size / 2))
    ATA = torch.stack([xx_patch, xy_patch, xz_patch, xy_patch, yy_patch, yz_patch, xz_patch, yz_patch, zz_patch],
                      dim=4)
    ATA = torch.squeeze(ATA)
    ATA = torch.reshape(ATA, (ATA.size(0), ATA.size(1), 3, 3))
    eps_identity = 1e-6 * torch.eye(3, device=ATA.device, dtype=ATA.dtype)[None, None, :, :].repeat([ATA.size(0), ATA.size(1), 1, 1])
    ATA = ATA + eps_identity
    x_patch = nn.functional.conv2d(x, weight=patch_weight, padding=int(patch_size / 2))
    y_patch = nn.functional.conv2d(y, weight=patch_weight, padding=int(patch_size / 2))
    z_patch = nn.functional.conv2d(z, weight=patch_weight, padding=int(patch_size / 2))
    AT1 = torch.stack([x_patch, y_patch, z_patch], dim=4)
    AT1 = torch.squeeze(AT1)
    AT1 = torch.unsqueeze(AT1, 3)

    patch_num = 4
    patch_x = int(AT1.size(1) / patch_num)
    patch_y = int(AT1.size(0) / patch_num)
    n_img = torch.randn(AT1.shape).cuda()
    overlap = patch_size // 2 + 1
    for x in range(int(patch_num)):
        for y in range(int(patch_num)):
            left_flg = 0 if x == 0 else 1
            right_flg = 0 if x == patch_num -1 else 1
            top_flg = 0 if y == 0 else 1
            btm_flg = 0 if y == patch_num - 1 else 1
            at1 = AT1[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                  x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
            ata = ATA[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                  x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
            n_img_tmp, _ = torch.solve(at1, ata)

            n_img_tmp_select = n_img_tmp[top_flg * overlap:patch_y + top_flg * overlap, left_flg * overlap:patch_x + left_flg * overlap, :, :]
            n_img[y * patch_y:y * patch_y + patch_y, x * patch_x:x * patch_x + patch_x, :, :] = n_img_tmp_select

    n_img_L2 = torch.sqrt(torch.sum(n_img ** 2, dim=2, keepdim=True))
    n_img_norm = n_img / n_img_L2

    # re-orient normals consistently
    orient_mask = torch.sum(torch.squeeze(n_img_norm) * torch.squeeze(xyz), dim=2) > 0
    n_img_norm[orient_mask] *= -1
    return n_img_norm

def get_surface_normalv2(xyz, patch_size=3):
    """
    xyz: xyz coordinates
    patch: [p1, p2, p3,
            p4, p5, p6,
            p7, p8, p9]
    surface_normal = [(p9-p1) x (p3-p7)] + [(p6-p4) - (p8-p2)]
    return: normal [h, w, 3, b]
    """
    b, h, w, c = xyz.shape
    half_patch = patch_size // 2
    xyz_pad = torch.zeros((b, h + patch_size - 1, w + patch_size - 1, c), dtype=xyz.dtype, device=xyz.device)
    xyz_pad[:, half_patch:-half_patch, half_patch:-half_patch, :] = xyz

    # xyz_left_top = xyz_pad[:, :h, :w, :]  # p1
    # xyz_right_bottom = xyz_pad[:, -h:, -w:, :]# p9
    # xyz_left_bottom = xyz_pad[:, -h:, :w, :]   # p7
    # xyz_right_top = xyz_pad[:, :h, -w:, :]  # p3
    # xyz_cross1 = xyz_left_top - xyz_right_bottom  # p1p9
    # xyz_cross2 = xyz_left_bottom - xyz_right_top  # p7p3

    xyz_left = xyz_pad[:, half_patch:half_patch + h, :w, :]  # p4
    xyz_right = xyz_pad[:, half_patch:half_patch + h, -w:, :]  # p6
    xyz_top = xyz_pad[:, :h, half_patch:half_patch + w, :]  # p2
    xyz_bottom = xyz_pad[:, -h:, half_patch:half_patch + w, :]  # p8
    xyz_horizon = xyz_left - xyz_right  # p4p6
    xyz_vertical = xyz_top - xyz_bottom  # p2p8

    xyz_left_in = xyz_pad[:, half_patch:half_patch + h, 1:w+1, :]  # p4
    xyz_right_in = xyz_pad[:, half_patch:half_patch + h, patch_size-1:patch_size-1+w, :]  # p6
    xyz_top_in = xyz_pad[:, 1:h+1, half_patch:half_patch + w, :]  # p2
    xyz_bottom_in = xyz_pad[:, patch_size-1:patch_size-1+h, half_patch:half_patch + w, :]  # p8
    xyz_horizon_in = xyz_left_in - xyz_right_in  # p4p6
    xyz_vertical_in = xyz_top_in - xyz_bottom_in  # p2p8

    n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
    n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

    # re-orient normals consistently
    orient_mask = torch.sum(n_img_1 * xyz, dim=3) > 0
    n_img_1[orient_mask] *= -1
    orient_mask = torch.sum(n_img_2 * xyz, dim=3) > 0
    n_img_2[orient_mask] *= -1

    n_img1_L2 = torch.sqrt(torch.sum(n_img_1 ** 2, dim=3, keepdim=True))
    n_img1_norm = n_img_1 / (n_img1_L2 + 1e-8)

    n_img2_L2 = torch.sqrt(torch.sum(n_img_2 ** 2, dim=3, keepdim=True))
    n_img2_norm = n_img_2 / (n_img2_L2 + 1e-8)

    # average 2 norms
    n_img_aver = n_img1_norm + n_img2_norm
    n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver ** 2, dim=3, keepdim=True))
    n_img_aver_norm = n_img_aver / (n_img_aver_L2 + 1e-8)
    # re-orient normals consistently
    orient_mask = torch.sum(n_img_aver_norm * xyz, dim=3) > 0
    n_img_aver_norm[orient_mask] *= -1
    n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  # [h, w, c, b]

    # a = torch.sum(n_img1_norm_out*n_img2_norm_out, dim=2).cpu().numpy().squeeze()
    # plt.imshow(np.abs(a), cmap='rainbow')
    # plt.show()
    return n_img_aver_norm_out#n_img1_norm.permute((1, 2, 3, 0))

def surface_normal_from_depth(depth, focal_length, valid_mask=None):
    # para depth: depth map, [b, c, h, w]
    b, c, h, w = depth.shape
    focal_length = focal_length[:, None, None, None]
    depth_filter = nn.functional.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
    depth_filter = nn.functional.avg_pool2d(depth_filter, kernel_size=3, stride=1, padding=1)
    xyz = depth_to_xyz(depth_filter, focal_length)
    sn_batch = []
    for i in range(b):
        xyz_i = xyz[i, :][None, :, :, :]
        normal = get_surface_normalv2(xyz_i)
        sn_batch.append(normal)
    sn_batch = torch.cat(sn_batch, dim=3).permute((3, 2, 0, 1))  # [b, c, h, w]
    mask_invalid = (~valid_mask).repeat(1, 3, 1, 1)
    sn_batch[mask_invalid] = 0.0

    return sn_batch
def optimize_st(depth_torch,normal_torch,self_defined_focal_length:float):
    input_depth = depth_torch.cuda()
    input_depth = input_depth.unsqueeze(0).unsqueeze(0)
    gt_normal= normal_torch.cuda().permute(0,3,1,2)
    focal_length= torch.Tensor([self_defined_focal_length,]).cuda()
    s= nn.Parameter(torch.Tensor([1.]).cuda().requires_grad_(True))
    t= nn.Parameter(torch.Tensor([0.0]).cuda().requires_grad_(True))
    
    optimizer = torch.optim.Adam([
                {'params':s, 'lr':1e-2},
                {'params':t, 'lr': 1e-2},
    
            ])
    last_step_loss = 100000.
    for step in range(500):
        optimizer.zero_grad()
        scaled_depth = s*input_depth+t
        # depth = s*input_depth+t
        depth_filter = nn.functional.avg_pool2d(scaled_depth, kernel_size=3, stride=1, padding=1)
        depth_filter = nn.functional.avg_pool2d(depth_filter, kernel_size=3, stride=1, padding=1)
        xyz = depth_to_xyz(depth_filter, focal_length)
        xyz_i = xyz[0, :][None, :, :, :]
        pre_normal = get_surface_normalv2(xyz_i).permute((3, 2, 0, 1))
        
        similarity = torch.nn.functional.cosine_similarity(pre_normal, -gt_normal, dim=1)
        # if similarity
        # loss = torch.nanmean(1 - similarity)
        loss = torch.nanmean(1 - similarity)
        
        loss.backward()
        optimizer.step()
        if step%40==0:
            if abs(loss.item()-last_step_loss)<1e-5:
                break
            last_step_loss = loss.item()
            
            print(loss.item())
    return s.item(),t.item()


def obtain_allimgs_st(depth_dirs,normals_dirs,st_dict,self_define_focal:float):
    for depth_path,normal_path in zip(depth_dirs,normals_dirs):
        assert os.path.basename(depth_path)==os.path.basename(normal_path)
        print("Optimizing :",os.path.basename(depth_path))
        depth_np = np.load(depth_path)
        basename= os.path.basename(depth_path)
        # gt_depth = torch.from_numpy(np.load(depth_gt_path))
        H, W = depth_np.shape
        depth_torch = torch.from_numpy(depth_np) # (B, h, w)
        normal_np = np.load(normal_path)
        normal_torch = torch.from_numpy(normal_np).unsqueeze(0) # (B, h, w,3)
        
        s,t = optimize_st(depth_torch,normal_torch,self_define_focal)
        st_dict[basename]={"scale":s,"shift":t}
        
        print(".==========.",s,t)

def obtain_static_msk(base_dir):
    """ obtain static msk from predicted_flow:
        NOTE: ONLY Works for static cam scene.
    """
    static_msk =None
    flow_dirs = glob( os.path.join(base_dir,"./flow_RAFT1/*.npz"))
    for flow_dir in flow_dirs:
        flow= np.load(flow_dir)
        dist_flow = np.linalg.norm(flow["flow"],ord=2,axis=-1)
        if static_msk is None:
            print("init msk")
            static_msk = np.ones_like(dist_flow)
        static_msk = np.logical_and(static_msk,dist_flow<0.8)
    return static_msk

def align_all_frames(depth_dirs:list, st_predicted,new_scale_alignFrame0:dict,static_msk):
    # depth_dirs = glob(os.path.join(base_dir,'GeoWizardOut/depth_npy/*.npy'))
    reference_depthdir = depth_dirs[5] ## 5 is the reference frame 这个必须要和 align scale的时候用的一致。
    basename = os.path.basename(reference_depthdir)

    reference_depth = np.load(reference_depthdir)
    # masked_reference_depth = reference_depth
    reference_s = st_predicted[basename]["scale"]
    reference_t = st_predicted[basename]["shift"]
    # if np.isnan()
    scaled_refer_depth_nomsk = reference_s *reference_depth+reference_t
    h,w = scaled_refer_depth_nomsk.shape
    static_msk = np.array(Image.fromarray(static_msk).resize((w,h)))>0

    scaled_refer_depth =scaled_refer_depth_nomsk[static_msk] 
    print("reference image name:",basename)
    print("reference Scale and shift:",reference_s,reference_t)
    
    # new_scale_alignFrame0=dict()
    Y =torch.from_numpy(scaled_refer_depth).unsqueeze(-1)
    for depth_dir in depth_dirs:
        if depth_dir == reference_depthdir:
            continue 
        cur_depth_name = os.path.basename(depth_dir)
        print("Dealing:",cur_depth_name)
        depth = np.load(depth_dir)
        cur_s= st_predicted[cur_depth_name]["scale"]
        cur_t= st_predicted[cur_depth_name]["shift"]
        cur_masked_depth = depth[static_msk]
        if not np.isnan(cur_s) or not np.isnan(cur_t):
            pass ## 都不是nan
        else:
            cur_s = st_predicted["mean_s"]
            cur_t = st_predicted["mean_t"]
        scaled_cur_depth = cur_masked_depth*cur_s+cur_t
        # print()

        ##### Solving using 
        A  =torch.from_numpy(scaled_cur_depth).unsqueeze(-1)
        
        res =torch.linalg.lstsq(A,Y  )
        new_scale_alignFrame0[cur_depth_name]=res.solution.item()
        print("S,T,align_scale:",cur_s,cur_t,res.solution.item())
    return new_scale_alignFrame0
    pass 
def export_scaled_pcd(st_predicted,new_scale_alignFrame0,data_dir,mean_st=False,):

    # import open3d as o3d
    for img_dir in depth_dirs:
        depth =np.load(img_dir)
        basename = os.path.basename(img_dir)
        if mean_st:
            s=st_predicted["mean_s"]
            t= st_predicted["mean_t"]
        else:
            s= st_predicted[basename]["scale"]
            t= st_predicted[basename]["shift"]
            
        depth = depth*s+t
        if basename in new_scale_alignFrame0:
            depth =depth*new_scale_alignFrame0[basename]
        else:
            print("No scale for:",basename)
        cam_xyz = depth_to_xyz(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda(), torch.Tensor([800,]).cuda())
        cam_xyz = cam_xyz.reshape(-1,3)

        save_dir = os.path.join(data_dir,"./output/exported_scaled_pcd",f"./dpeht_xyz_cellomyy_{basename}.txt")
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
            print("saving file to dir:",save_dir,)
        np.savetxt(save_dir,cam_xyz.cpu().numpy()[::100,:])

    
def generate_nonedge_msk(depth_dirs,st_predicted,new_scale_alignFrame0,valid_mask_dir,q = 0.95):
    # mean_st=False
    # q = 0.95
    print("Generating non-edge mask:quatile_threshold",q)
    for depth_dir in depth_dirs:
        basename = os.path.basename(depth_dir)    
        depth=np.load(depth_dir)
        # break
        s= st_predicted[basename]["scale"]
        t= st_predicted[basename]["shift"]
        if np.isnan(s) or np.isnan(t):
            s= st_predicted["mean_s"]
            t= st_predicted["mean_t"]
            print(basename,"nan")
        if basename in new_scale_alignFrame0:
            print(basename,s,t,new_scale_alignFrame0[basename])
            depth = (depth*s+t)*new_scale_alignFrame0[basename]
        else:
            depth = (depth*s+t)
        img_points = depth_to_xyz(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda(), torch.Tensor([800,]).cuda()).squeeze()

        
        mask = sobel_by_quantile(img_points.cpu().numpy() ,q)
        mask = (255*(1-mask.squeeze()/255.0)).astype(np.uint8)
        img = Image.fromarray(mask)
        save_path = os.path.join(valid_mask_dir,basename+".png")
        img.save(save_path)
        # print(depth_dir)
        
    pass

if __name__=="__main__":
    
    import configargparse
    parser = configargparse.ArgumentParser(description="Training script parameters")## LQM

    # mlp = MLP_Params(parser)

    parser.add_argument('--base_dir', type=str, required=True,help=" base directory of inputs.")
    parser.add_argument('--pre_defined_focal', type=float, required=True,help="self defined focal length")
    parser.add_argument('--export_scaled_pcd', action="store_true")
    parser.add_argument('--qauntile_q', type=float, default=0.90)
    args = parser.parse_args(sys.argv[1:]) 
    print("quatile_Q is:",args.qauntile_q)
    
    # base_dir =  "/data/qingmingliu/Dataset/dynamic/SelfMade/Pamela/"
    # base_dir =  "/data/qingmingliu/Dataset/dynamic/SelfMade/cellonado_staCam/"
    base_dir = args.base_dir
    depth_dirs =sorted(glob(os.path.join(args.base_dir,"GeoWizardOut/depth_npy/*.npy")))
    normal_dirs = sorted(glob(os.path.join(args.base_dir,"GeoWizardOut/normal_npy/*.npy")))
    
    st_predicted = {}
    new_scale_alignFrame0=dict()
    obtain_allimgs_st(depth_dirs,normal_dirs,st_predicted,args.pre_defined_focal)
    mean_s =0
    mean_t=0
    invalid = 0
    for k,v in st_predicted.items():
        # print(k)
        if not np.isnan(v["scale"]) and not np.isnan(v["shift"]):
            mean_s+=v["scale"] 
            mean_t+=v["shift"] 
        else:
            invalid-=1
    mean_s=mean_s/(len(st_predicted.keys())+invalid)
    mean_t=mean_t/(len(st_predicted.keys())+invalid)
    print("mean_s, mean_t:",mean_s,mean_t)
    print("invalid_number:",invalid)
    st_predicted["mean_s"]=mean_s
    st_predicted["mean_t"]=mean_t
    
    ## obtain static msk
    static_msk = obtain_static_msk(args.base_dir)
    align_all_frames(depth_dirs,st_predicted,new_scale_alignFrame0,static_msk)
    
    with open(os.path.join(base_dir,'./__re_scale_scene.json'), 'w') as f:
        json.dump(new_scale_alignFrame0, f)
    with open(os.path.join(base_dir,'./__scene.json'), 'w') as f:
        json.dump(st_predicted, f)
    if args.export_scaled_pcd:
        print("export scaled point")
        
        export_scaled_pcd(st_predicted,new_scale_alignFrame0,base_dir,mean_st=False,)
        pass
    
    ## compute depth non-edge mask:
    
    valid_mask_dir = os.path.join(base_dir,"GeoWizardOut","depth_valid_msk")  
    if not os.path.exists(valid_mask_dir):
        os.makedirs(valid_mask_dir) 
    generate_nonedge_msk(depth_dirs,st_predicted,new_scale_alignFrame0,valid_mask_dir,args.qauntile_q)
    
    