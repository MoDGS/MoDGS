export CUDA_VISIBLE_DEVICES=2
## env nerfstudio
python train.py --config configs/default.txt --data_dir /data/qingmingliu/mnt/data/data_release/tennis --save_dir tennis_0923_2 --expname tenis --num_iters 10000000 --foreground_mask_path /data/qingmingliu/mnt/data/data_release/tennis/mask/00000.png 