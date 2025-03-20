#!/bin/bash
src_dir = "/data/qingmingliu/Dataset/dynamic/IphoneDataset/iphone/apple/"
dest_dir = "/data/qingmingliu/Dataset/dynamic/IphoneDataset/iphone/apple_8/"


file_lists=("camera" "rgb" "splits" "dataset.json" "emf.json" "scene.json" "extra.json" "points-before.npy" "points.npy")

# 遍历文件夹并复制
for file in "${file_lists[@]}"
do
    if [ -d "$src_dir/$file" ]
    then
        cp -r "$src_dir/$file" "$dest_dir"
    elif [ -f "$src_dir/$file" ]
    then
        cp "$src_dir/$file" "$dest_dir"
    else
        echo "Folder or File $src_dir/$file does not exist."
    fi
done
  