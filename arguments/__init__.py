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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group
class MLP_Params(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.input_dim = 3
        self.output_dim = 6
        self.filter_size = 128
        self.act_fn="relu"
        self.ckpt_path="None"
        self.net_depth = 8 
        # self.image_mode = "fg"
        self.lrate_stage1=0.0001 
        self.lambda_stage1_l2=1
        self.lambda_stage1_l1=1
        self.stage1_max_steps=3000000
        self.stage1_save_interval=100000
        self.multires_xyz=4 
        self.multires_time=4 
        self.filterfirst_N=-1
        self.stage1_model_path="None"
        self.stage1_validation_step_interval = 10000
        self.TimePcd_dir="None"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
class NeuralInverseTrajectory_Params(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        ## optimization parameters
        self.pcd_interval = 8 
        self.exhaustive_training = False ## use dense Trajectory pairs or not.
        self.init_interval = 8 ## only used for Exhaustive training model.
        # self.act_fn="relu"
        self.ckpt_path="None"
        self.lr_feature = 1e-3
        self.lr_deform = 1e-4
        self.lrate_decay_steps = 20000
        self.lrate_decay_factor = 0.99991
        self.grad_clip = 0.0
        self.max_points_perstep=-1
        self.neighbor_K=20
        self.local_smoothness_loss=0.0
        # self.image_mode = "fg"
        self.pe_freq=4
        self.filterfirst_N=-1
        self.stage1_model_path="None"
        self.stage1_validation_step_interval = 10000
        self.stage1_max_steps = 3000000
        self.stage1_save_interval=100000
        self.normalize_time=True
        self.TimePcd_dir="None"
        self.load_optimizer=False
        self.lr_rate_scheduler="ExponentialLR"
        self.use_Global_NearFar=0
        super().__init__(parser, "Loading Parameters", sentinel)

    # def extract(self, args):
    #     g = super().extract(args)
    #     g.source_path = os.path.abspath(g.source_path)
    #     return g

class PointTrackModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.gs_model_version = "PointTrackIsotropicGaussianModel"
        self._resolution = -1
        self._white_background = False
        self.random_init_pcd= False ## fg, bg, fg_bg
        self.random_background= False ## fg, bg, fg_bg
        # self.random_background= False ## fg, bg, fg_bg
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        # self.is_blender = False
        self.validation_step_interval = 1000
        self.depth_folder="None"
        self.use_depthNonEdgeMsk = False
        self.all_SH = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PointTrackOptimizationParams(ParamGroup):
    def __init__(self, parser):
        ## optimization parameters
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 40_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.neighbor_k=10 
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.render_2flow_loss_start_iter = 3000
        ## lambda 
        self.lambda_dssim = 0.2
        self.lambda_recon = 1.0
        self.lambda_pcd_flow = 0.0
        self.lambda_gs_approx_flow= 0.0 ### wight of gaussian to approximate 3d flow 
        self.lambda_depth_plane= 0.0 ### wight of gaussian to approximate 3d flow 
        self.lambda_opacity_sparse= 0.0 ### wight of gaussian to approximate 3d flow 
        self.lambda_depthloss= 0.0 ### wight of gaussian to approximate 3d flow 
        self.lambda_2dflowloss= 0.0 ### wight of gaussian to approximate 3d flow 
        self.lambda_depthOderLoss = 0.0 ### wight of gaussian to approximate 3d flow 
        self.lambda_2dStatic_loss = 0.0 
        self.depth_loss_type="pearson_fsgs_selectmasked"
        self.depth_order_loss_type="None"
        self.render2dflow_loss_type="mae" ## normalized_mae, mae, 
        self.render2dflow_loss_mode="neighbor_flow" ## neighbor or exhaustive
        self.Alpha_tanh=100.0 ## alpha_value of tanh function
        self.Add_depth_noise=False ## whether add noise to depth
        ## custom densification  just for Our Mono3DGS
        self.custom_densification_start = 1000000000000
        self.custom_densification_end = -1
        self.custom_densification_interval = 100000
        # self.enable_2DRenderFlowSupervison=False
        
        super().__init__(parser, "Optimization Parameters")





class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.image_mode = "fg" ## fg, bg, fg_bg
        self.depth_folder="None"
        self._resolution = -1
        self.random_init_pcd= False ## fg, bg, fg_bg
        self._white_background = False
        self.eval = True
        self.approx_l = 5
        self.approx_l_global = -1
        self.data_device = "cuda"
        self.initPcdFromfirstframeDepth=False
        self.validation_step_interval = 10000
        # self.initPcdFromfirstframeDepth=True
        # self.static_util_iter=3000
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 100_00
        self.position_coeff_lr_init = 0.001
        self.position_coeff_lr_final = 0.00001
        self.position_coeff_lr_delay_mult = 0.01
        self.position_coeff_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.rotation_coeff_lr = 0.001
        self.global_coeff_lr=0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_flow = 0.0
        self.lambda_lasso = 0.001
        self.lambda_alpha = 1.0
        self.lambda_localsmoothness=0.0
        self.lambda_sparse_movement=0.01
        self.lambda_depthloss=0.00
        self.lambda_opacity=0.0
        self.neighbor_k=10 
        self.localsmoothness_delta_t=2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, args_cmdline.timestamp,"cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
