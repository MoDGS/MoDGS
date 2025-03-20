export CUDA_VISIBLE_DEVICES=1
python train_PointTrackGS.py \
    --stage1_model  NeuralInverseTrajectory \
    --mode train \
    --stage stage1 \
    --config configs/PointTrack_IphoneData/002SigAsia24Rebuttal/Time_test/Coffee_martini_5_training_NeuralInverseTrajectory_stage1.ini \
    --comments Filtered_exhaustive_PointTrack_training 
