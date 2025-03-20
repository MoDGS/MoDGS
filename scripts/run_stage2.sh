###########
###########
###########
########### Stage 2: Co-training tanh threshold.
##DyNeRF
##DyNeRF
export CUDA_VISIBLE_DEVICES=3
python train_PointTrackGS.py  \
    --stage1_model NeuralInverseTrajectory \
    --mode train \
    --stage stage12_cotraining \
    --config configs/PointTrack_IphoneData/Exhaustive_PointTrack_training/selfmade_DyNeRF/cook_spinach/tanh_Thresh0.05_cook_spinach_5_OriginalGS_DepthOderLoss_Co-trainning_stage12.ini \
    --comments "depth oder loss."


