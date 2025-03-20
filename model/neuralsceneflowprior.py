import torch

import sys
sys.path.append("../")
sys.path.append("./")
from torch.utils.data import DataLoader
import pdb
from pytorch3d.ops import knn_gather, knn_points

class Neural_Flow(torch.nn.Module):
    """Codes mainly borrowed from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/main/model.py#L4
    

    Args:
        torch (_type_): _description_
    """
    def __init__(self, input_dim=3, output_dim=3,filter_size=128, act_fn='relu', net_depth=8):
        super().__init__()
        self.net_depth = net_depth
        
        self.nn_layers = torch.nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if net_depth >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(input_dim, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(net_depth-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, output_dim))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim)))

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        for layer in self.nn_layers:
            x = layer(x)
                
        return x
    
from model.RealNVP import NVPSimplified
from model.mfn import GaborNet
import os
from utils.loss_utils import l1_loss, ssim,mask_l1_loss,mask_ssim,l2_loss
from utils.general_utils import get_expon_lr_func

class BasicTrainer:
    def  __init__(self) -> None:
        pass

class Neural_InverseTrajectory_Trainer(BasicTrainer):
    """Codes mainly borrowed from"""
    
    def __init__(self, args, device='cuda'):
        # super().__init__()
        super(Neural_InverseTrajectory_Trainer, self).__init__()
        self.scalars_to_log={}
        self.args = args
        self.device = device


                                # list(self.color_mlp.parameters())
        self.out_dir =args.outdir
        self.precompute_index = {}
    def training_setup(self,training_args=None):
                # self.read_data()
        if training_args is None or not hasattr(training_args, 'lr_feature'):
            print("trainning args not found using  default")
            training_args = self.args
        self.feature_mlp = GaborNet(in_size=1,
                                    hidden_size=256,
                                    n_layers=2,
                                    alpha=4.5,
                                    out_size=128).to(self.device)

        # self.deform_mlp = NVPSimplified(n_layers=6,
        self.deform_mlp = NVPSimplified(n_layers=4,
                                        feature_dims=128,
                                        hidden_size=[256, 256, 256],
                                        proj_dims=256,
                                        code_proj_hidden_size=[],
                                        proj_type='fixed_positional_encoding',
                                        pe_freq=training_args.pe_freq,
                                        normalization=False,
                                        affine=False,
                                        ).to(self.device)
        # self.to_cuda()
        self.optimizer = torch.optim.Adam([
            {'params': self.feature_mlp.parameters(), 'lr': training_args.lr_feature},
            {'params': self.deform_mlp.parameters(), 'lr': training_args.lr_deform},

        ])
        lr_rate_scheduler = training_args.lr_rate_scheduler
        if lr_rate_scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=training_args.lrate_decay_steps,
                                                            gamma=training_args.lrate_decay_factor)
        elif lr_rate_scheduler == 'ExponentialLR':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                            gamma=training_args.lrate_decay_factor)
        self.resume_step=0
        if training_args.ckpt_path !="None" and training_args.ckpt_path is not None:
            self.load_model(training_args.ckpt_path,load_optimizer=training_args.load_optimizer)
        self.learnable_params = list(self.feature_mlp.parameters()) + \
                                list(self.deform_mlp.parameters()) 
        pass
    def co_training_step(self,step,data):
        """step, from cannoical Space to other time t
        """
        pass
        
    def to_cuda(self):
        self.feature_mlp.to(self.device)
        self.deform_mlp.to(self.device)
    def load_model(self,ckpt_path=None,load_optimizer=True):
        if ckpt_path is None:
            if self.args.ckpt_path =="None":
                step = self.find_max_iter_ckpt()
                ckpt_path =os.path.join(self.out_dir, f'{step}_model.pth')
            else:
                ckpt_path=self.args.ckpt_path
        
        checkpoint = torch.load(ckpt_path)
        self.feature_mlp.load_state_dict(checkpoint['feature_mlp'])
        self.deform_mlp.load_state_dict(checkpoint['deform_mlp'])
        step=checkpoint['step']
        if 'optimizer' in checkpoint and load_optimizer:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                # self.scheduler= torch.optim.lr_scheduler.StepLR(self.optimizer,
                #                                                 step_size=self.args.lrate_decay_steps,
                #                                                 gamma=self.args.lrate_decay_factor)
                # for i in range(step):
                #     self.scheduler.step()
            except Exception as e:
                print(e)
                print("LOADING MODEL OPTIMIZER AND SCHEDULER FAILED,INITIALIZE NEW OPTIMIZER")  
                self.optimizer = torch.optim.Adam([
                    {'params': self.feature_mlp.parameters(), 'lr': self.args.lr_feature},
                    {'params': self.deform_mlp.parameters(), 'lr': self.args.lr_deform},
                ])
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=self.args.lrate_decay_steps,
                                                                gamma=self.args.lrate_decay_factor)
            else:
                print("LOADING MODEL OPTIMIZER AND SCHEDULER SUCCESSFULLY")
        
        self.resume_step=step
    def find_max_iter_ckpt(self,dir):
        """find the maximum iteration in the checkpoint folder
        """
        from glob import glob
        if dir is None:
            dir = self.out_dir
        ckpt_list = glob(os.path.join(dir, '*_model.pth'))
        
        ## 最大的迭代次数ckpt是
        max_iter = 0
        for ckpt in ckpt_list:
            iter = int(os.path.basename(ckpt).split('_')[0])
            if iter > max_iter:
                max_iter = iter
        max_iter_path = os.path.join(dir, f'{max_iter}_model.pth')
        return max_iter_path,max_iter
    def find_ckptByIter(self,dir , iter):
        """find the maximum iteration in the checkpoint folder
        """
        iter_path = os.path.join(dir, f'{iter}_model.pth')
        assert os.path.exists(iter_path),f"ckpt {iter_path} not found"
        
        return iter_path
    def save_model(self,step,keep_optimzer=True):
        state_dict = {
            'step': step,
            'feature_mlp': self.feature_mlp.state_dict(),
            'deform_mlp': self.deform_mlp.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
        }
        if self.args.stage1_max_steps>=step and keep_optimzer:
            state_dict["optimizer"]= self.optimizer.state_dict()
            state_dict["scheduler"]= self.scheduler.state_dict()
        # else: ## 不保存
        torch.save(state_dict, os.path.join(self.out_dir, f'{step}_model.pth'))
    def to_eval(self,):
        self.feature_mlp.eval()
        self.deform_mlp.eval()
    def to_trainning(self,):
        self.feature_mlp.train()
        self.deform_mlp.train()
    def validate_one_step(self,data):
        x = data["valid_xyz"]
        t= data["time"]
        x_canno, time_feature = self.forward_to_canonical(x,t)
        
        fwd_flow_loss = 0.0
        bwd_flow_loss =0.0
        if "fwd_gt" in data:
            next_time = t+data["time_interval"]
            next_xyz = data["fwd_gt"]["fwd_valid_gt"]
            next_mask = data["fwd_gt"]["fwd_mask"]
            next_xyz_pred = self.inverse_other_t(x_canno,next_time)
            fwd_flow_loss = l2_loss(next_xyz_pred[next_mask],next_xyz[next_mask])
            
            # pass
        
        if "bwd_gt" in data:
            pre_time = t- data["time_interval"]
            pre_xyz = data["bwd_gt"]["bwd_valid_gt"]
            pre_mask = data["bwd_gt"]["bwd_mask"]
            pre_xyz_pred = self.inverse_other_t(x_canno,pre_time)
            bwd_flow_loss = l2_loss(pre_xyz_pred[pre_mask],pre_xyz[pre_mask])
        loss = fwd_flow_loss + bwd_flow_loss
        return loss.item()
    def validate_table_completion(self,dataset,step,writer):
        self.to_eval()
        val_dataset= dataset.get_val_dataset()
        # with torch.no_grad():

        val_l2loss = 0.0
        dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0,shuffle=False)
        with torch.no_grad():
            for idx,data in enumerate(dataloader):
                loss = self.validate_one_step(data)
                val_l2loss+=loss
            val_l2loss/=len(val_dataset)
            writer.add_scalar('val_loss_k2', val_l2loss, global_step=step)
        # model.train()
        del val_dataset
        torch.cuda.empty_cache()
        self.to_trainning()
        return val_l2loss
    def forward_to_canonical(self, x,t): 
        """ 
        从时间t帧的点坐标x转换到时间t0的标准空间点坐标。
            [B, N, 3] -> [B,N,3]
            
        t：##torch.Size([B, 1])
        x：##torch.Size([B, N, 3])
        """
        time_feature = self.feature_mlp(t)#torch.Size([B, feature_dim])
        x = self.deform_mlp(t,time_feature,x)
        
        return x,time_feature
    def inverse_cycle_t(self, x,t, time_feature):
        """反向到同一个时刻,这个时候用在fwd时间步得到的time feature ，不用再次计算。

        Args:
            x (_type_): _description_
            t (_type_): _description_
            time_feature (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.deform_mlp.inverse(t,time_feature,x)
        return x
    
    def inverse_other_t(self, x,t):
        """反向到其他时刻,这个时候需要再次计算time_feature"""
        time_feature = self.feature_mlp(t)#torch.Size([B, feature_dim])
        x = self.deform_mlp.inverse(t,time_feature,x) 
        return x
    def expand_dim(self,data_dict,keys):
        for k,v in data_dict.items():
            if isinstance(v,torch.Tensor)  and k in keys:
                data_dict[k]=v.unsqueeze(0) ##[N,K] --> [1,N,K]
            if  isinstance(v,dict):
                data_dict[k]=self.expand_dim(v,keys=keys)
        return data_dict
    def random_select_data(self,pairs,N_pnts):
        valid_xyz = pairs["valid_xyz"]
        if valid_xyz.shape[0]>N_pnts:
            pnts_idx = torch.randperm(valid_xyz.shape[0])[:N_pnts]
            pairs["valid_xyz"] = valid_xyz[pnts_idx]
            if "fwd_gt" in pairs:
                pairs["fwd_gt"]["fwd_valid_gt"] = pairs["fwd_gt"]["fwd_valid_gt"][pnts_idx]
                pairs["fwd_gt"]["fwd_mask"] = pairs["fwd_gt"]["fwd_mask"][pnts_idx]
            if "bwd_gt" in pairs:
                pairs["bwd_gt"]["bwd_valid_gt"] = pairs["bwd_gt"]["bwd_valid_gt"][pnts_idx]
                pairs["bwd_gt"]["bwd_mask"] = pairs["bwd_gt"]["bwd_mask"][pnts_idx]
        return pairs
    def get_flow_loss(self, step,data,N_pnts=200000):
        keys =["valid_xyz","time","fwd_valid_gt","bwd_valid_gt","bwd_mask","fwd_mask","time_interval"]
        data= self.random_select_data(data,N_pnts  )
        self.expand_dim(data,keys)
        
        x = data["valid_xyz"]
        t= data["time"]
        x_canno, time_feature = self.forward_to_canonical(x,t)
        
        fwd_flow_loss = 0.0
        bwd_flow_loss =0.0
        loss=0.0
        if "fwd_gt" in data:
            next_time = t+data["time_interval"]
            next_xyz = data["fwd_gt"]["fwd_valid_gt"]
            next_mask = data["fwd_gt"]["fwd_mask"]
            next_xyz_pred = self.inverse_other_t(x_canno,next_time)
            fwd_flow_loss = l2_loss(next_xyz_pred[next_mask],next_xyz[next_mask])
            
            # pass
        
        if "bwd_gt" in data:
            pre_time = t- data["time_interval"]
            pre_xyz = data["bwd_gt"]["bwd_valid_gt"]
            pre_mask = data["bwd_gt"]["bwd_mask"]
            pre_xyz_pred = self.inverse_other_t(x_canno,pre_time)
            bwd_flow_loss = l2_loss(pre_xyz_pred[pre_mask],pre_xyz[pre_mask])
        loss = fwd_flow_loss + bwd_flow_loss
        
        return loss
        # pass
        
    def train_one_step(self,step,data):
        self.optimizer.zero_grad()
        x = data["valid_xyz"]
        t= data["time"]
        x_canno, time_feature = self.forward_to_canonical(x,t)
        
        fwd_flow_loss = 0.0
        bwd_flow_loss =0.0
        loss=0.0
        if "fwd_gt" in data:
            next_time = t+data["time_interval"]
            next_xyz = data["fwd_gt"]["fwd_valid_gt"]
            next_mask = data["fwd_gt"]["fwd_mask"]
            next_xyz_pred = self.inverse_other_t(x_canno,next_time)
            fwd_flow_loss = l2_loss(next_xyz_pred[next_mask],next_xyz[next_mask])
            
            # pass
        
        if "bwd_gt" in data:
            pre_time = t- data["time_interval"]
            pre_xyz = data["bwd_gt"]["bwd_valid_gt"]
            pre_mask = data["bwd_gt"]["bwd_mask"]
            pre_xyz_pred = self.inverse_other_t(x_canno,pre_time)
            bwd_flow_loss = l2_loss(pre_xyz_pred[pre_mask],pre_xyz[pre_mask])
        loss = fwd_flow_loss + bwd_flow_loss
        
        
        # loss = torch.nn.functional.mse_loss(x_pred, x)
        if self.args.grad_clip > 0:
            for param in self.learnable_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(param, self.args.grad_clip)
                if grad_norm > self.args.grad_clip:
                    print("Warning! Clip gradient from {} to {}".format(grad_norm, self.args.grad_clip))



        self.scalars_to_log['NIT_lr_feature'] = self.optimizer.param_groups[0]['lr']
        self.scalars_to_log['NIT_lr_deform'] = self.optimizer.param_groups[1]['lr']
        self.scalars_to_log['NIT_train_l2loss'] = loss.detach().item()

        if torch.isnan(loss):
            pdb.set_trace()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # self.optimizer.step()
        """runtimeerror: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [256, 2]], which is output 0 of AsStridedBackward0, is at version 3; expected version 2 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient.
        The variable in question was changed in there or anywhere later. Good luck!
        """
        return loss.item()
    
    
    ##############################################################################################################
    #################################### For Point Track Model Version 4.0 ,Exhaustive Pairs Training ##########
    ############################################################################################################
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
    def get_local_smoothness_loss(self,pcd,flow,index=None,neighbor_K=10,loss_type="l2"):
        if index is None:
            pairwise_dist = knn_points(pcd.unsqueeze(0), pcd.unsqueeze(0), K=neighbor_K, return_sorted=False)
            index = pairwise_dist.idx
        neighbor_flows = knn_gather(flow.unsqueeze(0), index)#neighbor_K)
        neighbor_flows=neighbor_flows[:,:,1:,:] ## remove the first point which is the point itself
        if loss_type=="l1":
            loss = torch.mean(torch.abs(flow.unsqueeze(0).unsqueeze(2)-neighbor_flows))
        else:
            loss = torch.mean(torch.square(flow.unsqueeze(0).unsqueeze(2)-neighbor_flows))   
            # loss = torch.mean(torch.square(flow.unsqueeze(0).unsqueeze(2)-neighbor_flows))
        return {"loss":loss,"index":index}
        # pass 
    def train_exhautive_one_step(self,step,data):
        """used for point track model version 4.0
        """
        self.optimizer.zero_grad()
        x = data["pcd"]
        t= data["time"]
        fromTo = data["fromTo"][0]
        next_time = data["target_gt"]["time"]
        next_pcd = data["target_gt"]["pcd"]
        next_msk= data["target_gt"]["pcd_target_msk"]
        
        x_canno_msked, time_feature = self.forward_to_canonical(x[next_msk].unsqueeze(0),t)
        next_xyz_pred_msked = self.inverse_other_t(x_canno_msked,next_time)
        flow_loss= l2_loss(next_xyz_pred_msked,next_pcd[next_msk].unsqueeze(0))
        
        
        loss=0.0
        loss = flow_loss
        
        # loss = torch.nn.functional.mse_loss(x_pred, x)
        if self.args.grad_clip > 0:
            for param in self.learnable_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(param, self.args.grad_clip)
                if grad_norm > self.args.grad_clip:
                    print("Warning! Clip gradient from {} to {}".format(grad_norm, self.args.grad_clip))

        if self.args.local_smoothness_loss>0:
            index = None
            if fromTo in  self.precompute_index:
                index = self.precompute_index[fromTo].cuda()
            pcd = x[next_msk]
            flow = next_xyz_pred_msked - pcd
            dic= self.get_local_smoothness_loss(pcd,flow.squeeze(0),index,self.args.neighbor_K)
            
            loss += self.args.local_smoothness_loss*dic["loss"]
            if not fromTo  in  self.precompute_index:
                self.precompute_index[fromTo]=dic["index"].cpu()
        if self.args.local_smoothness_loss:    
            self.scalars_to_log['localSmoothness_loss'] = dic["loss"].detach().item()
        self.scalars_to_log['flow_loss'] = flow_loss.detach().item()
        
        self.scalars_to_log['NIT_lr_feature'] = self.optimizer.param_groups[0]['lr']
        self.scalars_to_log['NIT_lr_deform'] = self.optimizer.param_groups[1]['lr']
        self.scalars_to_log['NIT_train_l2loss'] = loss.detach().item()

        if torch.isnan(loss):
            pdb.set_trace()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # self.optimizer.step()
        """runtimeerror: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [256, 2]], which is output 0 of AsStridedBackward0, is at version 3; expected version 2 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient.
        The variable in question was changed in there or anywhere later. Good luck!
        """
        return loss.item()
    
    def validate_exhaustive_one_step(self,data):
        """ used for point track model version 4.0
        """

        # self.optimizer.zero_grad()
        x = data["pcd"]
        t= data["time"]
        
        next_time = data["target_gt"]["time"]
        next_pcd = data["target_gt"]["pcd"]
        next_msk= data["target_gt"]["pcd_target_msk"]
        
        x_canno_msked, time_feature = self.forward_to_canonical(x[next_msk].unsqueeze(0),t)
        next_xyz_pred_msked = self.inverse_other_t(x_canno_msked,next_time)
        flow_loss= l2_loss(next_xyz_pred_msked.squeeze(0),next_pcd[next_msk])
        

        loss = flow_loss
        return loss.item()
    def validate_exhaustive_table_completion(self,dataset,step,writer):
        self.to_eval()
        val_dataset= dataset.get_val_dataset()
        # with torch.no_grad():

        val_l2loss = 0.0
        dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0,shuffle=False)
        with torch.no_grad():
            print("valid data flow loss.",end="")
            for idx,data in enumerate(dataloader):
                loss = self.validate_exhaustive_one_step(data)
                val_l2loss+=loss
            val_l2loss/=len(val_dataset)
            writer.add_scalar('val_loss_k2', val_l2loss, global_step=step)
        # model.train()
        del val_dataset
        torch.cuda.empty_cache()
        self.to_trainning()
        return val_l2loss
    

    def log(self,  step,writer):
        if len(self.scalars_to_log)==0:
                    self.scalars_to_log['NIT_lr_feature'] = self.optimizer.param_groups[0]['lr']
                    self.scalars_to_log['NIT_lr_deform'] = self.optimizer.param_groups[1]['lr']
        for key, value in self.scalars_to_log.items():
            writer.add_scalar(key, value, step)
        self.scalars_to_log = {}
    

                         
if __name__=="__main__":
    from arguments import MLP_Params
    import configargparse
    parser = configargparse.ArgumentParser(description="Training script parameters")## LQM
    
    mlp = MLP_Params(parser)
    # Test Neural_Flow
    args = parser.parse_args(sys.argv[1:1])
    args.pe_freq = 4
    model = Neural_InverseTrajectory_Trainer(args=args, device='cuda')    
    N=100
    x = torch.rand(1,N, 3).to('cuda')
    t = torch.rand(1, 1).to('cuda')
    out,time_feature = model.forward_to_canonical(x,t)## t(B,1) X(B,N,3)
    model.inverse(out,t,time_feature)
    pass
