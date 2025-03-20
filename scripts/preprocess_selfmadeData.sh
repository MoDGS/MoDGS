#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

conda_path="/data/qingmingliu/Software/anaconda3/"





#####################################
### Self Made DyNeRF Dataset#########
#####################################

# data_dir=/data/qingmingliu/Dataset/dynamic/SelfMade_DyNeRF/flame_salmon_1
# data_dir=/data/qingmingliu/Dataset/dynamic/SelfMade_DyNeRF/flame_steak
pre_defined_focal=365.53 ## focal length ##  Original_resolution focal 1462.12526569, we use 1/4 origin_resolution ，divide by 4 == 365.5，

in_out_door=indoor
output_geowizard_dir="${data_dir}/GeoWizardOut"
exhaustive_training="True"
# geowizard_input_folder="rgb/2x"
geowizard_input_folder="rgb_interlval1/2x"

source "${conda_path}/bin/activate" geowizard
echo $(which python)
echo "--------Step1: Run Geowizard to Accquire depth --------------"
python  submodules/GeoWizard/geowizard/run_infer.py  \
    --input_dir $data_dir/$geowizard_input_folder/\*.png \
    --output_dir $data_dir/GeoWizardOut \
    --denoise_steps 10 \
    --ensemble_size 3 \
     --domain $in_out_door

wait
echo "--------Step1: Run Geowizard to Accquire depth [Done] --------------"

source "${conda_path}/bin/activate" nerfstudio
# echo $(which python)

# Step2. Run Omimotion Exhaustive Flow Pairing
echo "--------Step2: Run Omimotion Exhaustive Flow Pairing --------------"
cd submodules/omnimotion/preprocessing
python main_processing.py  \
   --data_dir $data_dir
wait
echo "--------Step2: Run Omimotion Exhaustive Flow Pairing [Done]--------------"

echo "--------Step3: Run NSFF Exhaustive Flow Pairing --------------"
cd  submodules/Neural-Scene-Flow-Fields/nsff_scripts
python run_flows_video_selfmadeDataset.py \
    --model ./models/raft-things.pth \
    --data_path $data_dir
wait
echo "--------Step3: Run NSFF Exhaustive Flow Pairing [Done]--------------"


echo "--------Step4: Optimize Scale and shift --------------"
cd  ../../../
python preprocess/optimize_scale_shift.py \
    --base_dir $data_dir \
    --pre_defined_focal $pre_defined_focal \
    --export_scaled_pcd
wait
echo "--------Step4: Optimize Scale and shift [Done]--------------"

echo "--------Step5:Point Tracking --------------"
echo $(which python)
if [ "$exhaustive_training" = "True" ]; then
    echo "Run exhaustive exhaustive Point Tracking"
    python pointTrack.py --exhaustive_training \
        --source_path $data_dir \
        --images rgb/4x
        
else
    echo "Run normal PointTracking"
    python pointTrack.py \
        --source_path $data_dir \
        --images rgb/4x
fi
wait
echo "--------Step5:Point Tracking [Done]--------------"