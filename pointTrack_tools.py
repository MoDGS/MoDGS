
import torch
import numpy as np
import open3d as o3d
import time
# from diff_gaussian_rasterization import GaussianRasterizer as Renderer
# from helpers import setup_camera, quat_mult
# from external import build_rotation
from utils.colormap import colormap
from copy import deepcopy
from tqdm import tqdm
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import SH2RGB
from dataloader.timePcdTable_dataset import TimePCDTable,BaseCorrespondenceDataset,NeighbourFlowPairsDataset,ExhaustiveFlowPairsDataset
from model.neuralsceneflowprior import Neural_InverseTrajectory_Trainer
# from pytorch3d.ops import knn_gather, knn_points
import open3d as o3d 
# open3d.geometry.voxel_down_sample

def get_canonical_space_PCD(table,mask,N_se,net_query_fn_cano,mode="average"):
    """  
    N: batch size
    T: time steps
    """
    if mode == "average": ### 对于table中的没一个点都将他warp到canonical space，然后取平均值（xyz， color）
        T= table.shape[1]
        empty_table = torch.zeros([table.shape[0],6],dtype=table.dtype,device=table.device)
        assert table.shape[2]==6,"No rgb color error"
        denominator = (N_se[:,2]-N_se[:,1]+1).unsqueeze(-1)
        assert (denominator==0).sum()==0,"error"
        for t in tqdm(range(T)):
            ## extend left side
            query_mask= N_se[:,1]<=N_se[:,2]
            valid_coords = N_se[query_mask]
            if valid_coords.shape[0]>0:
                assert (valid_coords[:,1]<T).all(),"error"
                valid_xyz = table[valid_coords[:,0],valid_coords[:,1],:3]
                valid_t= valid_coords[:,1]
                normed_valid_t = valid_t/T
                canno_xyz,_ = net_query_fn_cano(valid_xyz.cuda(),normed_valid_t.cuda())
                # canno_xyz= canno_xyz.squeeze(0)
                if canno_xyz.dim()==3:
                    canno_xyz= canno_xyz.squeeze(0)
                empty_table[valid_coords[:,0],:3] = empty_table[valid_coords[:,0],:3]+canno_xyz ## copy xyz
                empty_table[valid_coords[:,0],3:6] = empty_table[valid_coords[:,0],3:6]+table[valid_coords[:,0],valid_coords[:,1],3:6] ## copy color
                ## update N_se
                N_se[query_mask,1] = N_se[query_mask,1]+1

        empty_table[:,:6] = empty_table[:,:6]/denominator
    else:
        raise NotImplementedError("Not implemented yet")      
    
    
    return empty_table
    # gaussian = torch.zeros(N,T,3).to(device)
    # return gaussian
    pass

def Downsample_pcd_usingOpen3d(rgbpcd,voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rgbpcd.cpu().numpy()[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(rgbpcd.cpu().numpy()[:,3:6])
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    xyz = np.asarray(downsampled_pcd.points)
    rgb = np.asarray(downsampled_pcd.colors)
    xyzrgb = np.concatenate([xyz,rgb],1)
    return torch.from_numpy(xyzrgb).to(rgbpcd.device)
def get_gaussians_init_pcd(dataset:BaseCorrespondenceDataset,trainer:Neural_InverseTrajectory_Trainer,mode="average",number_init_pnts=-1,save_pcd=False):
    """_summary_

    Args:
        table (_type_): _description_
        mask (_type_): _description_
        N_se (_type_): _description_
        net_query_fn (_type_): _description_
    """
    print("Get Canonical Space PCD...")
    trainer.to_eval()
    network_query_fn_2canno = lambda inputs, times : trainer.forward_to_canonical(inputs.unsqueeze(0), times.unsqueeze(-1))
    with torch.no_grad():
        if  isinstance(dataset,TimePCDTable):
            print("Get Canonical Space PCD from TimePCDTable")
            valid_mask  =dataset.get_valid_mask()
            
            time_pcd =dataset.get_time_pcd(with_rgb=True).clone()
            N_se  = get_start_end(valid_mask)
            # xyz = dataset.get_xyz()
            # shs = dataset.get_sh()
            
            xyzrgb = get_canonical_space_PCD(time_pcd,valid_mask,N_se,network_query_fn_2canno,mode=mode)
            if number_init_pnts<xyzrgb.shape[0] and number_init_pnts>0:
                # number_init_pnts = xyzrgb.shape[0]
                index = torch.randperm(xyzrgb.shape[0])[:number_init_pnts,]
                index = torch.randperm(xyzrgb.shape[0])[:number_init_pnts,]
                xyzrgb = xyzrgb[index] #tensor([ 398489, 1009989,  928207,  ...,   86964, 1211713,  124801])
            xyz = xyzrgb[:,:3]
            rgb = xyzrgb[:,3:]
            pcd = BasicPointCloud(points=xyz.cpu().numpy(), colors=rgb.cpu().numpy(), normals=np.zeros((xyz.shape[0], 3)))
            
        elif isinstance(dataset,NeighbourFlowPairsDataset):
            print("Get Canonical Space PCD from NeighbourFlowPairsDataset")
            
            pcd_pairs = dataset.time_pcd
            T = len(pcd_pairs)
            canonical_list = []
            for item in tqdm(pcd_pairs):
                pcd_t =  item["pcd"]
                time_t = torch.Tensor([item["frame_id"]/T])
                xyz_cano ,_= network_query_fn_2canno(pcd_t[:,:3].cuda(),time_t.cuda())
                rgb_cano = pcd_t[:,3:6]
                canonical_list.append(torch.cat([xyz_cano.cpu().squeeze(0),rgb_cano],1))
                # break
            xyzrgb=torch.cat(canonical_list,0)    
            # xyzrgb = torch.stack(canonical_list,0)
            ## save the pcd
            assert xyzrgb.dim()==2,"error"
            
            voxel_dim = 500
            min_cano_xyz = xyzrgb[:,:3].min(0)[0]
            max_cano_xyz = xyzrgb[:,:3].max(0)[0]
            center = (min_cano_xyz+max_cano_xyz)/2
            size = (max_cano_xyz-min_cano_xyz)
            vsize = (size/voxel_dim).max()
            assert vsize>0,"error"
            
            downsampled_xyzrgb = Downsample_pcd_usingOpen3d(xyzrgb,voxel_size=vsize)
            if number_init_pnts<downsampled_xyzrgb.shape[0] and number_init_pnts>0:
                # number_init_pnts = xyzrgb.shape[0]
                index = torch.randperm(downsampled_xyzrgb.shape[0])[:number_init_pnts,]
                index = torch.randperm(downsampled_xyzrgb.shape[0])[:number_init_pnts,]
                downsampled_xyzrgb = downsampled_xyzrgb[index]
            print("DownSampling From {} to {}".format(xyzrgb.shape[0],downsampled_xyzrgb.shape[0]))
            if save_pcd:
                np.savetxt("./output/canonical_pcd_BeforeDownsample.txt",xyzrgb.cpu().numpy())
                np.savetxt("./output/canonical_pcd_AfterDownsampled.txt",downsampled_xyzrgb.cpu().numpy(),delimiter=' ')
            pcd = BasicPointCloud(points=downsampled_xyzrgb[:,:3].cpu().numpy(), colors=downsampled_xyzrgb[:,3:].cpu().numpy(), normals=np.zeros((downsampled_xyzrgb.shape[0], 3)))
        
        elif isinstance(dataset,ExhaustiveFlowPairsDataset):
            print("Get Canonical Space PCD from ExhaustiveFlowPairsDataset")
            
            pcd_pairs = dataset.time_pcd
            T = len(pcd_pairs)
            canonical_list = []
            for index , item in tqdm(enumerate(pcd_pairs)):
                pcd_t =  item["pcd"]
                assert int(item["frame_id"])/dataset.PCD_INTERVAL==index,"error"  ### 在Exhaustive paring的时候 frame_id 存储的是 image_id 比如 "000000", "000001", "000002"...
                time_t = int(item["frame_id"])/dataset.PCD_INTERVAL/float(T)
                time_t = torch.Tensor([time_t])
                xyz_cano ,_= network_query_fn_2canno(pcd_t[:,:3].cuda(),time_t.cuda())
                rgb_cano = pcd_t[:,3:6]
                canonical_list.append(torch.cat([xyz_cano.cpu().squeeze(0),rgb_cano.cpu()],1))
                # break
            xyzrgb=torch.cat(canonical_list,0)    
            # xyzrgb = torch.stack(canonical_list,0)
            ## save the pcd
            assert xyzrgb.dim()==2,"error"
            
            voxel_dim = 500
            min_cano_xyz = xyzrgb[:,:3].min(0)[0]
            max_cano_xyz = xyzrgb[:,:3].max(0)[0]
            center = (min_cano_xyz+max_cano_xyz)/2
            size = (max_cano_xyz-min_cano_xyz)
            vsize = (size/voxel_dim).max()
            assert vsize>0,"error"
         
            downsampled_xyzrgb = Downsample_pcd_usingOpen3d(xyzrgb,voxel_size=vsize)
            if downsampled_xyzrgb.shape[0]<100000:
                downsampled_xyzrgb=xyzrgb
            if number_init_pnts<downsampled_xyzrgb.shape[0] and number_init_pnts>0:
                # number_init_pnts = xyzrgb.shape[0]
                index = torch.randperm(downsampled_xyzrgb.shape[0])[:number_init_pnts,]
                index = torch.randperm(downsampled_xyzrgb.shape[0])[:number_init_pnts,]
                downsampled_xyzrgb = downsampled_xyzrgb[index]
            print("DownSampling From {} to {}".format(xyzrgb.shape[0],downsampled_xyzrgb.shape[0]))
            if save_pcd:
                np.savetxt("./output/pntcloud/canonical_pcd_BeforeDownsample.txt",xyzrgb.cpu().numpy())
                np.savetxt("./output/pntcloud/canonical_pcd_AfterDownsampled.txt",downsampled_xyzrgb.cpu().numpy(),delimiter=' ')
            pcd = BasicPointCloud(points=downsampled_xyzrgb[:,:3].cpu().numpy(), colors=downsampled_xyzrgb[:,3:].cpu().numpy(), normals=np.zeros((downsampled_xyzrgb.shape[0], 3)))
        
            pass 
        
        else:
            raise NotImplementedError("Not implemented yet")       
    trainer.to_trainning()
    print("Get Canonical Space PCD...[Done]")
    return pcd


def filter_TimeTableTrajectory(table,config:dict):
    """filter the table with the config, remove the trajectory that not satisfy the *flow threshold* and * color changes thresh hold*
     

    Args:
        table (_type_): _description_
    """
    raise NotImplementedError("Not implemented yet")
def get_start_end(mask):
    """
    return shape of tensor: N*3 (trajectory idx,start valid points index,end valid points index)
    
    """
    index_min= torch.ones_like(mask).to(torch.int64)*1000
    non_zero_index_min = torch.nonzero(mask)
    index_min[non_zero_index_min[:,0],non_zero_index_min[:,-1]]=non_zero_index_min[:,-1]
    start = index_min.min(1)[0]

    index_max= torch.ones_like(mask).to(torch.int64)*-1
    non_zero_index_max = torch.nonzero(mask)
    index_max[non_zero_index_max[:,0],non_zero_index_max[:,-1]]=non_zero_index_max[:,-1]
    end = index_max.max(1)[0]

    assert (end<start).sum()== 0,"error" 
    se = torch.stack([start,end],-1)
    N = torch.arange(se.shape[0]).reshape(se.shape[0],-1).to(se.device)
    N_se = torch.cat([N,se],1)
    # (~mask[N_se[:,0],((N_se[:,2]+N_se[:,1])/2.0).round()]).sum()
    return N_se



def table_completion(table,mask,N_se,net_query_fn):
    """ NSFF method predict FLOW to fill the table
    table: shape of [N,T,6]
    mask: shape of [N,T]
    N_se: shape of [N,3]
    net_query_fn:function , predict flow give xyzt(shape of N,4)
    """
    T= table.shape[1]
    
    for t in tqdm(range(table.shape[1])):
        ## extend left side
        left_margin_mask= N_se[:,1]>0
        left_margin_coords = N_se[left_margin_mask]
        if left_margin_coords.shape[0]>0:
            left_margin_xyz = table[left_margin_coords[:,0],left_margin_coords[:,1],:3]
            left_margin_t= left_margin_coords[:,1]
            normed_margin_t = left_margin_t/T
            flow = net_query_fn(left_margin_xyz.cuda(),normed_margin_t.cuda())
            if flow.dim()==3:
                flow= flow.squeeze(0)
            bwd_flow= flow[:,:3].to(left_margin_xyz.device)
            predicted_xyz_left= bwd_flow+ left_margin_xyz
            ### fill table
            assert (left_margin_coords[:,1]-1>=0).all(),"error in left margin"
            assert  ~ (~torch.isnan(table[left_margin_coords[:,0],left_margin_coords[:,1]-1,:3])).all(),"error in left margin value"
            table[left_margin_coords[:,0],left_margin_coords[:,1]-1,:3] = predicted_xyz_left
            table[left_margin_coords[:,0],left_margin_coords[:,1]-1,3:] = table[left_margin_coords[:,0],left_margin_coords[:,1],3:] ## copy其他值
            ## update N_se
            N_se[left_margin_mask,1] = N_se[left_margin_mask,1]-1
            
        ## extend left side end 
        
        ## extend right side start
        right_margin_mask= N_se[:,2]<table.shape[1]-1
        right_margin_coords = N_se[right_margin_mask]
        if right_margin_coords.shape[0]>0:
            right_margin_xyz = table[right_margin_coords[:,0],right_margin_coords[:,2],:3]
            right_margin_t= right_margin_coords[:,2]
            normed_margin_t_right = right_margin_t/T
            flow = net_query_fn(right_margin_xyz.cuda(),normed_margin_t_right.cuda())
            if flow.dim()==3:
                flow= flow.squeeze(0)
            fwd_flow= flow[:,3:].to(right_margin_xyz.device)
            predicted_xyz_right= fwd_flow+ right_margin_xyz
            ### fill table 下一个坐标
            assert (right_margin_coords[:,2]+1<T).all(),"error in right margin"
            assert ~((~torch.isnan(table[right_margin_coords[:,0],right_margin_coords[:,2]+1,3:])).all()),"error in right margin value"
            table[right_margin_coords[:,0],right_margin_coords[:,2]+1,:3] = predicted_xyz_right
            table[right_margin_coords[:,0],right_margin_coords[:,2]+1,3:] = table[right_margin_coords[:,0],right_margin_coords[:,2],3:] ## copy其他值
            ## update N-se
            N_se[right_margin_mask,2] = N_se[right_margin_mask,2]+1
        ## extend right side end
        if  right_margin_coords.shape[0]==0 and  left_margin_coords.shape[0]==0:
            assert (torch.isnan(table[...,:3]).any(-1)).sum()==0,"error!!! table not fully filled yet"
            break
            
    return table
def table_completion_NIT(table,mask,N_se,net_query_fn_2Canno,net_query_fn_2timet):
    """ Neural Inverse Trajectory Method predict xyz to fill the table
    table: shape of [N,T,6]
    mask: shape of [N,T]
    N_se: shape of [N,3]
    net_query_fn:function , predict flow give xyzt(shape of N,4)
    """
    T= table.shape[1]
    
    for t in tqdm(range(table.shape[1])):
        ## extend left side
        left_margin_mask= N_se[:,1]>0
        left_margin_coords = N_se[left_margin_mask]
        if left_margin_coords.shape[0]>0:
            left_margin_xyz = table[left_margin_coords[:,0],left_margin_coords[:,1],:3]
            left_margin_t= left_margin_coords[:,1]
            normed_margin_t = left_margin_t/T
            canno_xyz,_ = net_query_fn_2Canno(left_margin_xyz.cuda(),normed_margin_t.cuda())
            canno_xyz= canno_xyz.squeeze(0)
            normed_margin_tleft = (left_margin_coords[:,1]-1)/T
            xyz_left=net_query_fn_2timet(canno_xyz,normed_margin_tleft.cuda())
            if xyz_left.dim()==3:
                xyz_left= xyz_left.squeeze(0)
            predicted_xyz_left= xyz_left.to(left_margin_xyz.device)
            
            # predicted_xyz_left= bwd_flow+ left_margin_xyz
            ### fill table
            assert (left_margin_coords[:,1]-1>=0).all(),"error in left margin"
            assert  ~ (~torch.isnan(table[left_margin_coords[:,0],left_margin_coords[:,1]-1,:3])).all(),"error in left margin value"
            table[left_margin_coords[:,0],left_margin_coords[:,1]-1,:3] = predicted_xyz_left
            table[left_margin_coords[:,0],left_margin_coords[:,1]-1,3:] = table[left_margin_coords[:,0],left_margin_coords[:,1],3:] ## copy其他值
            ## update N_se
            N_se[left_margin_mask,1] = N_se[left_margin_mask,1]-1
            
        ## extend left side end 
        
        ## extend right side start
        right_margin_mask= N_se[:,2]<table.shape[1]-1
        right_margin_coords = N_se[right_margin_mask]
        if right_margin_coords.shape[0]>0:
            right_margin_xyz = table[right_margin_coords[:,0],right_margin_coords[:,2],:3]
            right_margin_t= right_margin_coords[:,2]
            normed_margin_t = right_margin_t/T
            canno_xyz,_ = net_query_fn_2Canno(right_margin_xyz.cuda(),normed_margin_t.cuda())
            canno_xyz= canno_xyz.squeeze(0)
            normed_margin_t_right = (right_margin_coords[:,2]+1)/T
            xyz_right=net_query_fn_2timet(canno_xyz,normed_margin_t_right.cuda())
            if xyz_right.dim()==3:
                xyz_right= xyz_right.squeeze(0)
            predicted_xyz_right= xyz_right.to(left_margin_xyz.device)
            ### fill table 下一个坐标
            assert (right_margin_coords[:,2]+1<T).all(),"error in right margin"
            assert ~((~torch.isnan(table[right_margin_coords[:,0],right_margin_coords[:,2]+1,3:])).all()),"error in right margin value"
            table[right_margin_coords[:,0],right_margin_coords[:,2]+1,:3] = predicted_xyz_right
            table[right_margin_coords[:,0],right_margin_coords[:,2]+1,3:] = table[right_margin_coords[:,0],right_margin_coords[:,2],3:] ## copy其他值
            ## update N-se
            N_se[right_margin_mask,2] = N_se[right_margin_mask,2]+1
        ## extend right side end
        if  right_margin_coords.shape[0]==0 and  left_margin_coords.shape[0]==0:
            assert (torch.isnan(table[...,:3]).any(-1)).sum()==0,"error!!! table not fully filled yet"
            break
            
    return table