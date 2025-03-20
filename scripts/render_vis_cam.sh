# # ##### DyNeRF
# # ##### DyNeRF
# # ##### DyNeRF
# # ##### DyNeRF



export CUDA_VISIBLE_DEVICES=1
python render_PointTrackGS.py --stage1_model NeuralInverseTrajectory \
    --mode test_metric_cams \
    --stage stage12_cotraining \
    --config configs/PointTrack_IphoneData/Exhaustive_PointTrack_training/selfmade_nvidia_scene/Truck-2/tanh_threshold_Truck_DepthOrderLoss_Co-training_stage12.ini \
    --checkpoint output/Selfmade/Truck2-2/chkpnt30000.pth
# export CUDA_VISIBLE_DEVICES=1
# python render_PointTrackGS.py --stage1_model NeuralInverseTrajectory \
#     --mode test_metric_cams \
#     --stage stage12_cotraining \
#     --config configs/PointTrack_IphoneData/Exhaustive_PointTrack_training/selfmade_DyNeRF/flame_salmon/tanh_threshold_flame_salmon_5_OriginalGS_DepthOderLoss_Co-trainning_stage12.ini \
#     --checkpoint output/Selfmade/flame_salmon/chkpnt30000.pth 








### rendering videos


# export CUDA_VISIBLE_DEVICES=0
# python render_PointTrackGS.py --stage1_model NeuralInverseTrajectory \
#     --mode test_metric_cams \
#     --stage stage12_cotraining \
#     --config configs/PointTrack_IphoneData/Exhaustive_PointTrack_training/selfmade_DyNeRF/Coffee_martini/Coffee_martini_5_thah_threshold_OriginalGS_DepthOderLoss_Co-trainning_stage12.ini \
#     --checkpoint output/004ICLR2025Rebuttal/selfmade_DyNeRF/Coffee_martini/AblationDepthOderLoss_PerceptualLpips/241118_110347/chkpnt15000.pth \
#     --comments Depth\ LOSS\ test




