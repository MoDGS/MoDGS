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
from datetime import datetime
from errno import EEXIST
from os import makedirs, path
import os
from shutil import copyfile, copytree, ignore_patterns
import cv2
def save_cfg(args,outdir,timestamp):
        f = os.path.join(outdir, 'cfg_arg[{}].ini'.format(timestamp))
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                print(arg,attr)
                file.write('{} = {}\n'.format(arg, attr))
# pythonCopy code
import json
import numpy as np
def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj
# data = {"value": np.float32(3.14)}
# json_data = json.dumps(convert_to_serializable(data))
    
def check_exist(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        
    return dirs
def get_timestamp():
    return datetime.now().strftime(r"%y%m%d_%H%M%S")
def resize_flow(flow, img_h, img_w):
    # flow = np.load(flow_path)

    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w)/float(flow_w)
    flow[:, :, 1] *= float(img_h)/float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)
    return flow

def copy_files(src_dir, dst_dir, *ignores):
    copytree(src_dir, dst_dir, ignore=ignore_patterns(*ignores))

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise
def get_timestamp():
    return datetime.now().strftime(r"%y%m%d_%H%M%S")
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
def make_source_code_snapshot(log_dir):
    copy_files(
        ".",
        f"{log_dir}/source",
        ###### ignore files and directories ######
        "saved",
        "gaussian_render",
        "lpipsPyTorch",
        "__pycache__",
        "data",
        "logs",
        "scans",
        "Jupyter_test_exported",
        # ".vscode",
        "*.so",
        "*.a",
        ".ipynb_checkpoints",
        "build",
        "bin",
        "*.ply",
        "eigen",
        "pybind11",
        "*.npy",
        "*.pth",
        ".git",
        "debug",
        "assets",
        "output",
        ".ipynb_checkpoints",
        ".md",
        ".gitignore",
        ".gitmodules",
        ".yml",
    )