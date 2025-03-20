import torch,imageio,cv2
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000 
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import os 
import pickle
import copy
_img_suffix = ['png','jpg','jpeg','bmp','tif']
import json
from tqdm import tqdm
from utils.general_utils import dict_to_tensor_cuda,dict_to_tensor
def load(path):
    suffix = path.split('.')[-1]
    if suffix in _img_suffix:
        img =  np.array(Image.open(path))#.convert('L')
        scale = 256.**(1+np.log2(np.max(img))//8)-1
        return img/scale
    elif 'exr' == suffix:
        return imageio.imread(path)
    elif 'npy' == suffix:
        return np.load(path)

# def dict_to_tensor_cuda(dic):
#     for k,v in dic.items():
#         if isinstance(v,dict):
#             dic[k]=dict_to_tensor_cuda(v)
#         elif torch.is_tensor(v):
#             dic[k]=v.cuda()
#         elif isinstance(v,np.ndarray):
#             dic[k]=torch.from_numpy(v).cuda()#.to(torch.float32)
#     return dic
        

from utils.system_utils import check_exist       
class BaseCorrespondenceDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    def clean(self):
        pass
    def __len__(self):
        return len(self.training_pairs)
    def get_val_dataset(self):
        # assert self.val_pairs is not None
        """Construct Validation dataset.

        Returns:
            _type_: _description_
        """
        if self.val_pairs is None:
            print("No validation set found, please set reserve_for_validation_rate>0 when init the dataset")
        return TimePCDTable_val(self.val_pairs,device=self.device)
class TimePCDTable(BaseCorrespondenceDataset):
    PCD_INTERVAL=8
    def __init__(self, cfg, keeprgb=False,max_points=-1, normalize_time =True ,do_sale = True,split="No",sampling_strategy="random",reserve_for_validation_rate=0.0,device="cpu"):
        
        datadir = cfg.TimePcd_dir
        self.device = torch.device(device)
        time_pcd = torch.from_numpy(np.load(datadir)).to(self.device)[:,:,:]
        self.max_points = max_points### get——item时候返回的最大点云数量。
        if keeprgb:
            print("Keep RGB channel")
            self.time_pcd = time_pcd[:,:,:3]
            self.time_pcd_rgb = time_pcd[:,:,3:6]
        else:
            print("Keep xyz channel only")
            self.time_pcd = time_pcd[:,:,:3]
            self.time_pcd_rgb = None
        del time_pcd
        # self.time_pcd = torch.from_numpy(np.load(datadir)).to(self.device)[:,:,:6]
        self.PCD_INTERVAL=cfg.pcd_interval
        print("PCD_Interval:",self.PCD_INTERVAL)
        self.time_pcd_valid_mask = torch.logical_not(torch.isnan(self.time_pcd).any(-1))
        if reserve_for_validation_rate>0:
            N= self.time_pcd.shape[0]
        if do_sale:
            if os.path.exists(os.path.join(cfg.outdir,"re_scale.json")):
                print(f"Found re_scale.json, will load it")
                with open(os.path.join(cfg.outdir,"re_scale.json"), 'r') as json_file:
                    dict_rescale = json.load(json_file)
                mean_xyz = torch.tensor(dict_rescale["mean_xyz"]).to(self.device)
                scale = dict_rescale["scale"]
                min_xyz = torch.tensor(dict_rescale["min_xyz"]).to(self.device)
                max_xyz = torch.tensor(dict_rescale["max_xyz"]).to(self.device)

            else:
                
                min_xyz = self.time_pcd[(~torch.isnan(self.time_pcd[:,:,:]).any(-1))][:,:3].min(0)[0]
                max_xyz = self.time_pcd[(~torch.isnan(self.time_pcd[:,:,:]).any(-1))][:,:3].max(0)[0]
                # mean_xyz = self.time_pcd[:,0,:][~torch.isnan(self.time_pcd[:,0,:]).any(-1)].mean(0)
                mean_xyz = (min_xyz+max_xyz)/2
                bbox = max_xyz-min_xyz
                scale = 2.0/bbox.max() ## TODO: 这里是保持三个方向的scale一致？还是每个方向的scale不一样（这样会有一些畸变。） ## rescale 到 【-1,1】
                # array= torch.tensor([min_xyz,max_xyz,mean_xyz,scale])
                dict_rescale = {"min_xyz":min_xyz.cpu().numpy().astype(np.float64).tolist(),
                                "max_xyz":max_xyz.cpu().numpy().astype(np.float64).tolist(),
                                "mean_xyz":mean_xyz.cpu().numpy().astype(np.float64).tolist(),"scale":scale.cpu().numpy().astype(np.float64).tolist()}
                with open(os.path.join(cfg.outdir,"re_scale.json"), 'w') as json_file:
                    json.dump(dict_rescale, json_file)
            self.re_scale_json= dict_rescale
            self.mean_xyz = mean_xyz
            self.scale = scale
            pcd = self.time_pcd[(~torch.isnan(self.time_pcd[:,:,:]).any(-1))][:,:3]
            pcd=(pcd-mean_xyz)*scale
            # self.time_pcd[(~torch.isnan(self.time_pcd[:,:,:]).any(-1)),:3]=pcd
            new_time_pcd_xyz = torch.full_like(self.time_pcd[:,:,:3],fill_value=np.nan)
            new_time_pcd_xyz[(~torch.isnan(self.time_pcd[:,:,:]).any(-1))]=pcd
            new_time_pcd= torch.cat([new_time_pcd_xyz,self.time_pcd[:,:,3:]],-1)
            self.time_pcd = new_time_pcd
        # self.batchsize = batchsize
        
        save_val_table=False
        if reserve_for_validation_rate>0:
            """reserve for validation dataset.
            """
            val_dir  = check_exist(os.path.join(os.path.dirname(datadir),f"validation_{str(reserve_for_validation_rate)}"))
            
            if os.path.exists(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl")):
                print(f"Validation set index found, will load it")
                # self.time_table_val_index = np.load(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.npy"),allow_pickle=True)
                # self.time_table_val_index = 
                with open(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"), 'rb') as f:
                    self.time_table_val_index = pickle.load(f)
                self.time_table_val_index= dict_to_tensor_cuda(self.time_table_val_index)
                # pickle.load(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"))
            else:
                self.time_table_val_index = None
                time_table_val_index = dict()
                save_val_table=True
                print(f"Validation set index not found, will create a new one")


        self.training_pairs,self.val_pairs = construct_valid_training_pair(reserve_for_validation_rate)
        if save_val_table:
            self.time_table_val_index= time_table_val_index
            # np.save(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"),self.time_table_val_index)
            
            with open(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"), 'wb') as f:
                pickle.dump(self.time_table_val_index, f)
                # pickle.load(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"),self.time_table_val_index)
        
        
        for data in self.training_pairs.values():
            data = dict_to_tensor_cuda(data)
            if data is None:
                raise ValueError(f"Item at index  is None")        
        self.idx_list = list(self.training_pairs.keys())
        
        def construct_valid_training_pair(reserve_for_validate_rate=0.0):
            training_pairs =dict()
            val_pairs =None
            if reserve_for_validate_rate>0:
                val_pairs =dict()
            T= self.time_pcd.shape[1]
            for frame_i in range(T):
                bwd_pair =None
                fwd_pair =None
                # # bwd_pair ={"bwd_valid_gt":None,"bwd_mask":None}
                # # fwd_pair ={"fwd_valid_gt":None,"fwd_mask":None}
                if reserve_for_validate_rate>0:
                    bwd_pair_val =None
                    fwd_pair_val =None
                    
            
                nonan_msk = torch.logical_not(torch.isnan(self.time_pcd[:,frame_i,:]).any(1))
                pcd_i = self.time_pcd[:,frame_i,:][nonan_msk]
                if reserve_for_validate_rate>0:
                    if self.time_table_val_index is not None:
                        val_mask = self.time_table_val_index[frame_i]
                        pcd_i_val = pcd_i[val_mask]
                        pcd_i = pcd_i[torch.logical_not(val_mask)]
                    else:
                        val_n= int(pcd_i.shape[0]*reserve_for_validate_rate)
                        all_idx = torch.randperm(pcd_i.shape[0])
                        val_mask = torch.zeros([pcd_i.shape[0],],dtype=torch.bool)
                        val_mask[all_idx[:val_n]]=True
                        time_table_val_index[frame_i] = val_mask.cpu().numpy()
                        # val_idx=sorted(all_idx[:val_n])
                        # train_idx = sorted(all_idx[:val_n])
                        pcd_i_val = pcd_i[val_mask]
                        pcd_i = pcd_i[torch.logical_not(val_mask)]
                    
                    
                if  frame_i>0:
                    ## 构建 bwd flow
                    pcd_i_bwd= self.time_pcd[nonan_msk,frame_i-1,:]
                    if reserve_for_validate_rate>0:
                        pcd_i_bwd_val = pcd_i_bwd[val_mask]
                        pcd_i_bwd = pcd_i_bwd[torch.logical_not(val_mask)]
                        
                        bwd_nonan_msk_val = torch.logical_not(torch.isnan(pcd_i_bwd_val).any(1))
                        bwd_pair_val  = {"bwd_valid_gt":pcd_i_bwd_val,"bwd_mask":bwd_nonan_msk_val}                    
                        # bwd_pair_val  = {"bwd_valid_gt":pcd_i_bwd_val[bwd_nonan_msk_val],"bwd_mask":bwd_nonan_msk_val}                    
                    bwd_nonan_msk = torch.logical_not(torch.isnan(pcd_i_bwd).any(1))
                    # bwd_pair  = {"bwd_valid_gt":pcd_i_bwd[bwd_nonan_msk],"bwd_mask":bwd_nonan_msk}
                    bwd_pair  = {"bwd_valid_gt":pcd_i_bwd,"bwd_mask":bwd_nonan_msk}

                if frame_i<self.time_pcd.shape[1]-1:
                    pcd_i_fwd= self.time_pcd[nonan_msk,frame_i+1,:]
                    if reserve_for_validate_rate>0:
                        pcd_i_fwd_val = pcd_i_fwd[val_mask]
                        pcd_i_fwd = pcd_i_fwd[torch.logical_not(val_mask)]
                        
                        
                        fwd_nonan_msk_val = torch.logical_not(torch.isnan(pcd_i_fwd_val).any(1))
                        # fwd_pair_val  = {"fwd_valid_gt":pcd_i_fwd_val[fwd_nonan_msk_val],"fwd_mask":fwd_nonan_msk_val}     
                        fwd_pair_val  = {"fwd_valid_gt":pcd_i_fwd_val,"fwd_mask":fwd_nonan_msk_val}     
                    
                    fwd_nonan_msk = torch.logical_not(torch.isnan(pcd_i_fwd).any(1))
                    fwd_pair  ={"fwd_valid_gt":pcd_i_fwd,"fwd_mask":fwd_nonan_msk}
                    ## 构建fwd flow
                # training_pairs[frame_i] = {"valid_xyz":pcd_i,"bwd_gt":bwd_pair,"fwd_gt":fwd_pair}
                training_pairs[frame_i] = {"valid_xyz":pcd_i,"time":torch.Tensor([frame_i/T]),"time_interval":torch.Tensor([1/T]),"index":frame_i}
                if fwd_pair is not None:
                    training_pairs[frame_i]["fwd_gt"] = fwd_pair
                if bwd_pair is not None:
                    training_pairs[frame_i]["bwd_gt"] = bwd_pair
                    
                if reserve_for_validate_rate>0:
                    val_pairs[frame_i] = {"valid_xyz":pcd_i_val,"time":torch.Tensor([frame_i/T]),"time_interval":torch.Tensor([1/T]),"index":frame_i}
                    if fwd_pair_val is not None:
                        val_pairs[frame_i]["fwd_gt"] = fwd_pair_val
                    if bwd_pair_val is not None:
                        val_pairs[frame_i]["bwd_gt"] = bwd_pair_val

                    
            return training_pairs,val_pairs
    def clean(self):
        del self.time_pcd,self.time_pcd_rgb,self.time_pcd_valid_mask
    def inverse_scale(self,pcd):
        return pcd/self.scale+self.mean_xyz
    def get_rescale_json(self):
        return self.re_scale_json
    def get_valid_mask(self):
        return ~(torch.isnan(self.time_pcd).any(-1))
    def get_time_pcd(self,with_rgb=False):
        if with_rgb and hasattr(self,"time_pcd_rgb"):
            return torch.cat([self.time_pcd,self.time_pcd_rgb],-1)
        else:
            print(",Donot have rgb channel")
        return self.time_pcd
     
    def filter_firstN(self,firstN):
        "只留下前面N帧的训练集"
        if firstN<=0:
            print("firstN smaller than 0, will not filter the dataset")
            return
        for k in list(self.training_pairs.keys()):
            if k>=firstN:
                print("poping frame:",k)
                
                self.training_pairs.pop(k)
        if self.val_pairs is not None:
            for k in list(self.val_pairs.keys()):
                if k>=firstN:
                    print(k)
                    
                    self.val_pairs.pop(k)
    

    
    def __len__(self):
        return len(self.training_pairs)
    def _random_select_pnts(self,):
        pass
    def getTrainingPairs(self):
        return self.training_pairs
    def getTrainningPairs_bacthShape(self):
        training_ = self.getTrainingPairs().copy()
        raise NotImplementedError("adsf")
    def __getitem__(self, idx):
        # 
        # print(idx,len(self.training_pairs))
        # pairs = self.training_pairs[idx-1]
        pairs = self.training_pairs[idx]

        if self.max_points>0:
            valid_xyz = pairs["valid_xyz"]
            if valid_xyz.shape[0]>self.max_points:
                pairs= copy.deepcopy(self.training_pairs[idx])
                
                pnts_idx = torch.randperm(valid_xyz.shape[0])[:self.max_points]
                pairs["valid_xyz"] = valid_xyz[pnts_idx]
                if "fwd_gt" in pairs:
                    pairs["fwd_gt"]["fwd_valid_gt"] = pairs["fwd_gt"]["fwd_valid_gt"][pnts_idx]
                    pairs["fwd_gt"]["fwd_mask"] = pairs["fwd_gt"]["fwd_mask"][pnts_idx]
                if "bwd_gt" in pairs:
                    pairs["bwd_gt"]["bwd_valid_gt"] = pairs["bwd_gt"]["bwd_valid_gt"][pnts_idx]
                    pairs["bwd_gt"]["bwd_mask"] = pairs["bwd_gt"]["bwd_mask"][pnts_idx]
        
        pairs = dict_to_tensor_cuda(pairs)
        # self.idx_list.remove(idx)
        
        # xyz,pre_xyz,next_xyz,pre_mask,next_mask=None,None,None,None,None    
        if pairs is None:
            raise ValueError(f"Item at index {idx} is None")
        return pairs
    

class NeighbourFlowPairsDataset(BaseCorrespondenceDataset):
    PCD_INTERVAL=8
    def __init__(self, cfg, keeprgb=False,max_points=-1 ,do_sale = True,sampling_strategy="random",reserve_for_validation_rate=0.0,device="cpu"):
        
        datadir = cfg.TimePcd_dir
        self.device = torch.device(device)
        
        with open(datadir, 'rb') as f:
            time_pcd = pickle.load(f)
        self.max_points = max_points### get——item时候返回的最大点云数量。
        self.PCD_INTERVAL=cfg.pcd_interval

        print("PCD_Interval:",self.PCD_INTERVAL)
        if keeprgb:
            for pair in time_pcd:
                # print("Keep RGB channel")
                pair["pcd"] = pair["pcd"][:,:6]
                if "pcd_pre" in pair:
                    pair["pcd_pre"] = pair["pcd_pre"][:,:6]
                if "pcd_next" in pair:
                    pair["pcd_next"] = pair["pcd_next"][:,:6]

        else:
            print("Keep xyz channel only")
            for pair in time_pcd:
                pair["pcd"] = pair["pcd"][:,:3]
                if "pcd_pre" in pair:
                    pair["pcd_pre"] = pair["pcd_pre"][:,:3]
                if "pcd_next" in pair:
                    pair["pcd_next"] = pair["pcd_next"][:,:3]

        self.time_pcd=time_pcd
        

        if do_sale:
            if os.path.exists(os.path.join(os.path.dirname(cfg.TimePcd_dir),"re_scale.json")):
                print(f"Found re_scale.json, will load it")
                with open(os.path.join(os.path.dirname(cfg.TimePcd_dir),"re_scale.json"), 'r') as json_file:
                    dict_rescale = json.load(json_file)
                mean_xyz = torch.tensor(dict_rescale["mean_xyz"]).to(time_pcd[0]["pcd"].device)
                scale = dict_rescale["scale"]
                min_xyz = torch.tensor(dict_rescale["min_xyz"]).to(time_pcd[0]["pcd"].device)
                max_xyz = torch.tensor(dict_rescale["max_xyz"]).to(time_pcd[0]["pcd"].device)

            else:          
                min_list = []
                max_list = []
                for pair in time_pcd:
                    min_xyz = pair["pcd"][:,:3].min(0)[0]
                    max_xyz = pair["pcd"][:,:3].max(0)[0]
                    min_list.append(min_xyz)
                    max_list.append(max_xyz)
                min_xyz = torch.stack(min_list).min(0)[0]
                max_xyz = torch.stack(max_list).max(0)[0]

                mean_xyz = (min_xyz+max_xyz)/2
                bbox = max_xyz-min_xyz
                scale = 2.0/bbox.max() ## TODO: 这里是保持三个方向的scale一致？还是每个方向的scale不一样（这样会有一些畸变。） ## rescale 到 【-1,1】
                # array= torch.tensor([min_xyz,max_xyz,mean_xyz,scale])
                dict_rescale = {"min_xyz":min_xyz.cpu().numpy().astype(np.float64).tolist(),
                                "max_xyz":max_xyz.cpu().numpy().astype(np.float64).tolist(),
                                "mean_xyz":mean_xyz.cpu().numpy().astype(np.float64).tolist(),"scale":scale.cpu().numpy().astype(np.float64).tolist()}
                with open(os.path.join(os.path.dirname(cfg.TimePcd_dir),"re_scale.json"), 'w') as json_file:
                    json.dump(dict_rescale, json_file)
            self.re_scale_json= dict_rescale
            self.mean_xyz = mean_xyz
            self.scale = scale
            print("Rescale the data to [-1,1]: mean_xyz:",mean_xyz,"scale:",scale)
            for pair in time_pcd:
                pair["pcd"][:,:3]=( pair["pcd"][:,:3]-mean_xyz)*scale
        
                if "pcd_pre" in pair:
                    pair["pcd_pre"][:,:3]=(pair["pcd_pre"][:,:3]-mean_xyz)*scale
                if "pcd_next" in pair:
                    pair["pcd_next"][:,:3]=(pair["pcd_next"][:,:3]-mean_xyz)*scale
            
        # self.batchsize = batchsize
        
        save_val_table=False
        if reserve_for_validation_rate>0:
            """reserve for validation dataset.
            """
            val_dir  = check_exist(os.path.join(os.path.dirname(datadir),f"validation_{str(reserve_for_validation_rate)}"))
            
            if os.path.exists(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl")):
                print(f"Validation set index found, will load it")
                # self.time_table_val_index = np.load(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.npy"),allow_pickle=True)
                # self.time_table_val_index = 
                with open(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"), 'rb') as f:
                    self.time_table_val_index = pickle.load(f)
                self.time_table_val_index= dict_to_tensor(self.time_table_val_index,self.device)
                # pickle.load(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"))
            else:
                self.time_table_val_index = None
                save_val_table=True
                print(f"Validation set index not found, will create a new one")


        self.training_pairs,self.val_pairs = self.construct_valid_training_pair(reserve_for_validation_rate)
        if save_val_table:
            # self.time_table_val_index= time_table_val_index
            # np.save(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"),self.time_table_val_index)
            
            with open(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"), 'wb') as f:
                pickle.dump(self.time_table_val_index, f)
                # pickle.load(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"),self.time_table_val_index)

        print("Transfer Data to device !",self.device)
        for data in self.training_pairs.values():
            data = dict_to_tensor(data,self.device)
            if data is None:
                raise ValueError(f"Item at index  is None")        
        self.idx_list = list(self.training_pairs.keys())
        
    def construct_valid_training_pair(self,reserve_for_validate_rate=0.0):
        training_pairs =dict()
        time_table_val_index=dict()
        val_pairs =None
        if reserve_for_validate_rate>0:
            val_pairs =dict()
        T= len(self.time_pcd)
        for frame_i in range(T):
            bwd_pair =None
            fwd_pair =None
            assert self.time_pcd[frame_i]["frame_id"]==frame_i,"index mismatch{},{}".format(self.time_pcd[frame_i]["frame_id"],frame_i)
            if reserve_for_validate_rate>0:
                bwd_pair_val =None
                fwd_pair_val =None
                
            data_pair = self.time_pcd[frame_i]
            pcd_i =  data_pair["pcd"][:,:3]
            if reserve_for_validate_rate>0:
                if self.time_table_val_index is not None:
                    val_mask = self.time_table_val_index[frame_i]
                else:
                    val_n= int(pcd_i.shape[0]*reserve_for_validate_rate)
                    all_idx = torch.randperm(pcd_i.shape[0])
                    val_mask = torch.zeros([pcd_i.shape[0],],dtype=torch.bool)
                    val_mask[all_idx[:val_n]]=True
                    time_table_val_index[frame_i] = val_mask
                    # val_idx=sorted(all_idx[:val_n])
                    # train_idx = sorted(all_idx[:val_n])
                pcd_i_val = pcd_i[val_mask]
                pcd_i = pcd_i[torch.logical_not(val_mask)]
                
                
            if "pcd_pre" in data_pair:
                ## 构建 bwd flow
                pcd_i_bwd=  data_pair["pcd_pre"][:,:3]
                pcd_i_bwd_msk = data_pair["pcd_pre_msk"]
                if reserve_for_validate_rate>0:
                    pcd_i_bwd_val = pcd_i_bwd[val_mask]
                    pcd_i_bwd_msk_val = pcd_i_bwd_msk[val_mask]
                    pcd_i_bwd = pcd_i_bwd[torch.logical_not(val_mask)]
                    pcd_i_bwd_msk = pcd_i_bwd_msk[torch.logical_not(val_mask)]
                    bwd_pair_val  = {"bwd_valid_gt":pcd_i_bwd_val,"bwd_mask":pcd_i_bwd_msk_val}                    
                    # bwd_pair_val  = {"bwd_valid_gt":pcd_i_bwd_val[bwd_nonan_msk_val],"bwd_mask":bwd_nonan_msk_val}                    
                # bwd_pair  = {"bwd_valid_gt":pcd_i_bwd[bwd_nonan_msk],"bwd_mask":bwd_nonan_msk}
                bwd_pair  = {"bwd_valid_gt":pcd_i_bwd,"bwd_mask":pcd_i_bwd_msk}

            if "pcd_next" in data_pair:
                pcd_i_fwd=  data_pair["pcd_next"][:,:3]
                pcd_i_fwd_msk = data_pair["pcd_next_msk"]
                if reserve_for_validate_rate>0:
                    pcd_i_fwd_val = pcd_i_fwd[val_mask]
                    pcd_i_fwd_msk_val = pcd_i_fwd_msk[val_mask]
                    pcd_i_fwd = pcd_i_fwd[torch.logical_not(val_mask)]
                    pcd_i_fwd_msk = pcd_i_fwd_msk[torch.logical_not(val_mask)]
                    # fwd_pair_val  = {"fwd_valid_gt":pcd_i_fwd_val[fwd_nonan_msk_val],"fwd_mask":fwd_nonan_msk_val}     
                    fwd_pair_val  = {"fwd_valid_gt":pcd_i_fwd_val,"fwd_mask":pcd_i_fwd_msk_val}     
                fwd_pair  ={"fwd_valid_gt":pcd_i_fwd,"fwd_mask":pcd_i_fwd_msk}
                ## 构建fwd flow
            # training_pairs[frame_i] = {"valid_xyz":pcd_i,"bwd_gt":bwd_pair,"fwd_gt":fwd_pair}
            training_pairs[frame_i] = {"valid_xyz":pcd_i,"time":torch.Tensor([frame_i/T]),"time_interval":torch.Tensor([1/T]),"index":frame_i}
            if fwd_pair is not None:
                training_pairs[frame_i]["fwd_gt"] = fwd_pair
            if bwd_pair is not None:
                training_pairs[frame_i]["bwd_gt"] = bwd_pair
                
            if reserve_for_validate_rate>0:
                val_pairs[frame_i] = {"valid_xyz":pcd_i_val,"time":torch.Tensor([frame_i/T]),"time_interval":torch.Tensor([1/T]),"index":frame_i}
                if fwd_pair_val is not None:
                    val_pairs[frame_i]["fwd_gt"] = fwd_pair_val
                if bwd_pair_val is not None:
                    val_pairs[frame_i]["bwd_gt"] = bwd_pair_val
        if self.time_table_val_index is  None:
            self.time_table_val_index = time_table_val_index
        
        
        return training_pairs,val_pairs
    def inverse_scale(self,pcd):
        return pcd/self.scale+self.mean_xyz
    def get_rescale_json(self):
        return self.re_scale_json
     
    def filter_firstN(self,firstN):
        "只留下前面N帧的训练集"
        if firstN<=0:
            print("firstN smaller than 0, will not filter the dataset")
            return
        for k in list(self.training_pairs.keys()):
            if k>=firstN:
                print("poping frame:",k)
                
                self.training_pairs.pop(k)
        if self.val_pairs is not None:
            for k in list(self.val_pairs.keys()):
                if k>=firstN:
                    print(k)
                    
                    self.val_pairs.pop(k)
    
    def get_val_dataset(self):
        # assert self.val_pairs is not None
        """Construct Validation dataset.

        Returns:
            _type_: _description_
        """
        if self.val_pairs is None:
            print("No validation set found, please set reserve_for_validation_rate>0 when init the dataset")
        return TimePCDTable_val(self.val_pairs,device=self.device)
    

    def _random_select_pnts(self,):
        pass
    def getTrainingPairs(self):
        return self.training_pairs
    
    def __getitem__(self, idx):
        # 
        # print(idx,len(self.training_pairs))
        # pairs = self.training_pairs[idx-1]
        pairs = self.training_pairs[idx]

        if self.max_points>0:
            valid_xyz = pairs["valid_xyz"]
            if valid_xyz.shape[0]>self.max_points:
                pairs= copy.deepcopy(self.training_pairs[idx])
                
                pnts_idx = torch.randperm(valid_xyz.shape[0])[:self.max_points]
                pairs["valid_xyz"] = valid_xyz[pnts_idx]
                if "fwd_gt" in pairs:
                    pairs["fwd_gt"]["fwd_valid_gt"] = pairs["fwd_gt"]["fwd_valid_gt"][pnts_idx]
                    pairs["fwd_gt"]["fwd_mask"] = pairs["fwd_gt"]["fwd_mask"][pnts_idx]
                if "bwd_gt" in pairs:
                    pairs["bwd_gt"]["bwd_valid_gt"] = pairs["bwd_gt"]["bwd_valid_gt"][pnts_idx]
                    pairs["bwd_gt"]["bwd_mask"] = pairs["bwd_gt"]["bwd_mask"][pnts_idx]
        
        pairs = dict_to_tensor_cuda(pairs)
        # self.idx_list.remove(idx)
        
        # xyz,pre_xyz,next_xyz,pre_mask,next_mask=None,None,None,None,None    
        if pairs is None:
            raise ValueError(f"Item at index {idx} is None")
        return pairs
   
def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k.strip(".png").strip(".jpg")] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k.strip(".png").strip(".jpg")][j.strip(".png").strip(".jpg")]  \
                = 1. * flow_stats[k][j] / total_num
    return sample_weights
 
class ExhaustiveFlowPairsDataset(BaseCorrespondenceDataset):
    PCD_INTERVAL=8
    def __init__(self, cfg, keeprgb=False,max_points=-1 ,do_sale = True,sampling_strategy="random",reserve_for_validation_rate=0.0,device="cpu"):
        
        datadir = cfg.TimePcd_dir
        self.device = torch.device(device)
        flow_stats = json.load(open(os.path.join(os.path.dirname(datadir), 'flow_stats.json')))
        self.sample_weights = get_sample_weights(flow_stats)
        self.max_time=None
        if os.path.exists(os.path.join(os.path.dirname(datadir), 'rgb_interlval1')):
            import glob 
            path = os.path.dirname(datadir)
            self.max_time =  len(glob.glob(os.path.join(path, "rgb_interlval1", "*.png")))-1 ## 从0开始编号所以-1
            print("Max time:",self.max_time)
        with open(datadir, 'rb') as f:
            time_pcd = pickle.load(f)
        self.max_points = max_points### get——item时候返回的最大点云数量。

        self.PCD_INTERVAL=cfg.pcd_interval ## Training 点云的间隔
        self.max_interval = cfg.init_interval # 在query Exhaustive training pair时，最大的时间间隔是多少。
        self.init_interval = cfg.init_interval
        print("PCD_Interval:",self.PCD_INTERVAL)
        if keeprgb:
            print("Keep RGB channel!!")
            for cur_frame in time_pcd: ## list
                # print("Keep RGB channel")
                # current_frame = frame["current_frame"]
                for target_dict_key  in  list(cur_frame["target_dicts"].keys()):
                    
                    cur_frame["target_dicts"][target_dict_key]["pcd"] = cur_frame["target_dicts"][target_dict_key]["pcd"][:,:6]

        else:
            print("Keep xyz channel only")
            for cur_frame in time_pcd:
                # print("Keep RGB channel")
                # current_frame = frame["current_frame"]
                for target_dict_key  in  list(cur_frame["target_dicts"].keys()):
                    
                    cur_frame["target_dicts"][target_dict_key]["pcd"] = cur_frame["target_dicts"][target_dict_key]["pcd"][:,:3]
                cur_frame["pcd"] = cur_frame["pcd"][:,:3]
        self.time_pcd=time_pcd
        
        
        

        if do_sale:
            if os.path.exists(os.path.join(os.path.dirname(cfg.TimePcd_dir),"re_scale.json")):
                print(f"Found re_scale.json, will load it")
                with open(os.path.join(os.path.dirname(cfg.TimePcd_dir),"re_scale.json"), 'r') as json_file:
                    dict_rescale = json.load(json_file)
                mean_xyz = torch.tensor(dict_rescale["mean_xyz"]).to(time_pcd[0]["pcd"].device)
                scale = dict_rescale["scale"]
                min_xyz = torch.tensor(dict_rescale["min_xyz"]).to(time_pcd[0]["pcd"].device)
                max_xyz = torch.tensor(dict_rescale["max_xyz"]).to(time_pcd[0]["pcd"].device)

            else:          
                min_list = []
                max_list = []
                for cur_frame in time_pcd:
                    min_xyz = cur_frame["pcd"][:,:3].min(0)[0]
                    max_xyz = cur_frame["pcd"][:,:3].max(0)[0]
                    min_list.append(min_xyz)
                    max_list.append(max_xyz)
                min_xyz = torch.stack(min_list).min(0)[0]
                max_xyz = torch.stack(max_list).max(0)[0]

                mean_xyz = (min_xyz+max_xyz)/2
                bbox = max_xyz-min_xyz
                scale = 2.0/bbox.max() ## TODO: 这里是保持三个方向的scale一致？还是每个方向的scale不一样（这样会有一些畸变。） ## rescale 到 【-1,1】
                # array= torch.tensor([min_xyz,max_xyz,mean_xyz,scale])
                dict_rescale = {"min_xyz":min_xyz.cpu().numpy().astype(np.float64).tolist(),
                                "max_xyz":max_xyz.cpu().numpy().astype(np.float64).tolist(),
                                "mean_xyz":mean_xyz.cpu().numpy().astype(np.float64).tolist(),"scale":scale.cpu().numpy().astype(np.float64).tolist()}
                with open(os.path.join(os.path.dirname(cfg.TimePcd_dir),"re_scale.json"), 'w') as json_file:
                    json.dump(dict_rescale, json_file)
            self.re_scale_json= dict_rescale
            self.mean_xyz = mean_xyz
            self.scale = scale
            print("Rescale the data to [-1,1]: mean_xyz:",mean_xyz,"scale:",scale)                
            for cur_frame in time_pcd:
                # print("Keep RGB channel")
                # current_frame = frame["current_frame"]
                cur_frame["pcd"][:,:3] = (cur_frame["pcd"][:,:3]-mean_xyz)*scale
                for target_dict_key  in  list(cur_frame["target_dicts"].keys()):    
                    cur_frame["target_dicts"][target_dict_key]["pcd"][:,:3] = (cur_frame["target_dicts"][target_dict_key]["pcd"][:,:3] -mean_xyz)*scale
            
        # self.batchsize = batchsize
        
        save_val_table=False
        if reserve_for_validation_rate>0:
            """reserve for validation dataset.
            """
            val_dir  = check_exist(os.path.join(os.path.dirname(datadir),f"validation_{str(reserve_for_validation_rate)}"))
            
            if os.path.exists(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl")):
                print(f"Validation set index found, will load it")
                # self.time_table_val_index = np.load(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.npy"),allow_pickle=True)
                # self.time_table_val_index = 
                with open(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"), 'rb') as f:
                    self.time_table_val_index = pickle.load(f)
                self.time_table_val_index= dict_to_tensor(self.time_table_val_index,self.device)
                # pickle.load(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"))
            else:
                self.time_table_val_index = None
                save_val_table=True
                print(f"Validation set index not found, will create a new one")


        self.training_pairs,self.val_pairs = self.construct_valid_training_pair(reserve_for_validation_rate)
        if save_val_table:
            # self.time_table_val_index= time_table_val_index
            # np.save(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"),self.time_table_val_index)
            
            with open(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"), 'wb') as f:
                pickle.dump(self.time_table_val_index, f)
                # pickle.load(os.path.join(val_dir,f"TimePCD_validation_set_index_{str(reserve_for_validation_rate)}.pkl"),self.time_table_val_index)

        print("Transfer Data to devices!",self.device)
        for data in self.training_pairs.values():
            data = dict_to_tensor(data,self.device)
            if data is None:

                raise ValueError(f"Item at index  is None")        
        self.idx_list = list(self.training_pairs.keys())
        
        self.img_names=list(self.sample_weights.keys())
        
    def construct_valid_training_pair(self,reserve_for_validate_rate=0.0):
        print("Constructing Training pairs and Validation pairs")
        training_pairs =dict()
        time_table_val_index=dict()
        val_pairs =None
        name_to_index_mapping = {}
        index_to_name_mapping = {}
        if reserve_for_validate_rate>0:
            val_pairs =dict()
        T= len(self.time_pcd)
        for frame_i in tqdm(range(T)):

            assert int(self.time_pcd[frame_i]["frame_id"])==self.PCD_INTERVAL*frame_i,"index mismatch"+"{},{}".format(self.time_pcd[frame_i]["frame_id"],frame_i)
            name_to_index_mapping[self.time_pcd[frame_i]["frame_id"]]=frame_i
            index_to_name_mapping[frame_i]=self.time_pcd[frame_i]["frame_id"]
            
            frame_id = self.time_pcd[frame_i]["frame_id"]
            imgname = self.time_pcd[frame_i]["imgname"]
            pcd_i =  self.time_pcd[frame_i]["pcd"][:,:3]
            
            if self.max_time is not None:
                time= torch.Tensor([int(frame_id)/self.max_time])
                print("Time:",time,"frameid:",frame_id,"maxtime:",self.max_time,"image_name:",imgname)
            else:
                time = torch.Tensor([frame_i/T])
            if reserve_for_validate_rate>0:
                if self.time_table_val_index is not None:
                    val_mask = self.time_table_val_index[frame_i]
                else:
                    val_n= int(pcd_i.shape[0]*reserve_for_validate_rate)
                    all_idx = torch.randperm(pcd_i.shape[0])
                    val_mask = torch.zeros([pcd_i.shape[0],],dtype=torch.bool)
                    val_mask[all_idx[:val_n]]=True
                    time_table_val_index[frame_i] = val_mask
                    # val_idx=sorted(all_idx[:val_n])
                    # train_idx = sorted(all_idx[:val_n])
                    
            
                pcd_i_val = pcd_i[val_mask]
                pcd_i = pcd_i[torch.logical_not(val_mask)]
                current_frame_val = {"frame_id":frame_id,"time": time,"pcd":pcd_i_val,"imgname":imgname,"target_dicts":{}}
                # current_frame_val[] = torch.Tensor([frame_i/T])
            training_pairs[frame_i] = {"frame_id":frame_id,"time": time,"pcd":pcd_i,"imgname":imgname,"target_dicts":{}}
                
            # self.time_pcd[frame_i]["pcd"] = pcd_i 
            # self.time_pcd[frame_i]["time"]=torch.Tensor([frame_i/T])
               
            for dict_key in self.time_pcd[frame_i]["target_dicts"].keys():
                target_frame_id = self.time_pcd[frame_i]["target_dicts"][dict_key]["frame_id"]
                if self.max_time is not None:
                    target_time = torch.Tensor([int(target_frame_id)/self.max_time])
                else:
                    target_time = torch.Tensor([int(target_frame_id)/self.PCD_INTERVAL/T])
                training_pairs[frame_i]["target_dicts"][dict_key] = {}
                training_pairs[frame_i]["target_dicts"][dict_key]["time"] =target_time
                training_pairs[frame_i]["target_dicts"][dict_key]["frame_id"] = target_frame_id
                training_pairs[frame_i]["target_dicts"][dict_key]["pcd"] \
                        = self.time_pcd[frame_i]["target_dicts"][dict_key]["pcd"][:,:3] 
                training_pairs[frame_i]["target_dicts"][dict_key]["pcd_target_msk"] \
                        = self.time_pcd[frame_i]["target_dicts"][dict_key]["pcd_target_msk"]
                if reserve_for_validate_rate>0:
                    ### 构建 validation  frame                
                    pcd_target_val = self.time_pcd[frame_i]["target_dicts"][dict_key]["pcd"][val_mask][:,:3] ## 明确留下xyz，避免后面 validation的时候出错。
                    valid_mask_target_val =  self.time_pcd[frame_i]["target_dicts"][dict_key]["pcd_target_msk"][val_mask]
                    
                    target_dict = {"frame_id":target_frame_id,
                                   "pcd":pcd_target_val,
                                   "time":target_time,
                                   "pcd_target_msk":valid_mask_target_val}
                    current_frame_val["target_dicts"][dict_key]=target_dict

                    ### 更新 Training frame
                    training_pairs[frame_i]["target_dicts"][dict_key]["pcd"] \
                        = self.time_pcd[frame_i]["target_dicts"][dict_key]["pcd"][torch.logical_not(val_mask)][:,:3] ## 明确留下xyz，避免后面 validation的时候出错。
                    training_pairs[frame_i]["target_dicts"][dict_key]["pcd_target_msk"] \
                        = self.time_pcd[frame_i]["target_dicts"][dict_key]["pcd_target_msk"][torch.logical_not(val_mask)]
            # training_pairs[frame_i] = self.time_pcd[frame_i]
            if reserve_for_validate_rate>0:
                val_pairs[frame_i] = current_frame_val
        self.name_to_index_mapping = name_to_index_mapping  
        self.index_to_name_mapping = index_to_name_mapping  
        if self.time_table_val_index is  None:
            self.time_table_val_index = time_table_val_index
                
        return training_pairs,val_pairs

    def increase_maxInterval(self,step):
        # if step 
        ## FIXME: OPTIMIZE there , how to update: max_interval
        self.max_interval = min(self.init_interval+int(step/5000),len(self)-1)
        # raise NotImplementedError
    
    
    def inverse_scale(self,pcd):
        return pcd/self.scale+self.mean_xyz
    def get_rescale_json(self):
        return self.re_scale_json
     
    def filter_firstN(self,firstN):
        "只留下前面N帧的训练集"
        if firstN<=0:
            print("firstN smaller than 0, will not filter the dataset")
            return
        for k in list(self.training_pairs.keys()):
            if k>=firstN:
                print("poping frame:",k)
                
                self.training_pairs.pop(k)
        if self.val_pairs is not None:
            for k in list(self.val_pairs.keys()):
                if k>=firstN:
                    print(k)
                    
                    self.val_pairs.pop(k)
    
    def get_val_dataset(self):
        # assert self.val_pairs is not None
        """Construct Validation dataset.

        Returns:
            _type_: _description_
        """
        if self.val_pairs is None:
            print("No validation set found, please set reserve_for_validation_rate>0 when init the dataset")
        return ExhaustivePairs_valDataset(self.val_pairs,device=self.device)
    

    def _random_select_pnts(self,):
        pass
    def getTrainingPairs(self):
        return self.training_pairs
    
    def __getitem__(self, idx):
        # 
        # print(idx,len(self.training_pairs))
        # pairs = self.training_pairs[idx-1]

        
        frame = self.training_pairs[idx]
        imgname = frame["imgname"]
        
        if hasattr(self,"sample_weights"):
            ## sampling more from neighbour frames.
        
            max_interval = min(self.max_interval, len(self) - 1)
            img2_candidates = sorted(list(self.sample_weights[imgname].keys()))
            img2_candidates = img2_candidates[max(idx - max_interval, 0):min(idx + max_interval, len(self) - 1)]
            
            # id2s = np.array([self.img_names.index(n) for n in img2_candidates])
            id2s = np.array([self.img_names.index(n) for n in img2_candidates])
            sample_weights = np.array([self.sample_weights[imgname][i] for i in img2_candidates])
            sample_weights /= np.sum(sample_weights)
            sample_weights[np.abs(id2s - idx) <=1] = 0.1
            sample_weights /= np.sum(sample_weights)
            imgname_target = np.random.choice(img2_candidates, p=sample_weights)
            imgname_target = imgname_target.strip(".png").strip(".jpg")
            dict_key = imgname+"_"+imgname_target
        else:
            pass
        target_data = frame["target_dicts"][dict_key]
        
        pair = {"pcd":frame["pcd"],"time":frame["time"],  "index":frame["frame_id"],"fromTo":dict_key,
                "target_gt":target_data}
        
        if self.max_points>0:
            valid_xyz = pair["pcd"]
            if valid_xyz.shape[0]>self.max_points:
                pnts_idx = torch.randperm(valid_xyz.shape[0])[:self.max_points]
                new_target_gt = {
                                 "pcd":target_data["pcd"][pnts_idx],
                                 "pcd_target_msk":target_data["pcd_target_msk"][pnts_idx],
                                 "time":target_data["time"],
                                 "frame_id":target_data["frame_id"],}
                selected_pair =  {"pcd":frame["pcd"][pnts_idx],"time":frame["time"],  "index":frame["frame_id"],"fromTo":dict_key,
                "target_gt":new_target_gt}
                pair = selected_pair
               
        
        pair = dict_to_tensor_cuda(pair)
        # self.idx_list.remove(idx)
        
        # xyz,pre_xyz,next_xyz,pre_mask,next_mask=None,None,None,None,None    
        if pair is None:
            raise ValueError(f"Item at index {idx} is None")
        return pair
    


class TimePCDTable_val(Dataset):
    def __init__(self,pair,device="cpu"):
        self.training_pairs = pair
        self.idx_list = list(self.training_pairs.keys())
        
    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        # 
        # print(idx,len(self.training_pairs))
        # pairs = self.training_pairs[idx-1]
        pairs = self.training_pairs[idx]
        pairs = dict_to_tensor_cuda(pairs)
        # self.idx_list.remove(idx)
        
        # xyz,pre_xyz,next_xyz,pre_mask,next_mask=None,None,None,None,None    
        if pairs is None:
            raise ValueError(f"Item at index {idx} is None")
        return pairs
class ExhaustivePairs_valDataset(Dataset):
    def __init__(self,pair,device="cpu"):
        self.training_pairs = pair
        self.idx_list = list(self.training_pairs.keys())
        
    def __len__(self):
        return len(self.training_pairs)*(len(self.training_pairs[0]["target_dicts"])-1)

    def __getitem__(self, idx):
        idx_frame = int(idx/len(self.training_pairs))
        target_idx_frame = idx%(len(self.training_pairs)-1)
        frame = self.training_pairs[idx_frame]
        imgname = frame["imgname"]
        target_name = list(frame["target_dicts"].keys())[target_idx_frame]
        dict_key=target_name
        assert target_name.startswith(imgname),"Error: target name not match"
        target_data = frame["target_dicts"][dict_key]
        
        pair = {"pcd":frame["pcd"],"time":frame["time"],  "index":frame["frame_id"],"fromTo":dict_key,
                "target_gt":target_data}
        
        # if self.max_points>0:
        #     valid_xyz = pair["pcd"]
        #     if valid_xyz.shape[0]>self.max_points:
        #         raise NotImplementedError("Not implemented")
               
        
        pair = dict_to_tensor_cuda(pair)
        # self.idx_list.remove(idx)
        
        # xyz,pre_xyz,next_xyz,pre_mask,next_mask=None,None,None,None,None    
        if pair is None:
            raise ValueError(f"Item at index {idx} is None")
        return pair



if __name__=="__main__":
    pass