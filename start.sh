!/bin/bash
k=40
b=20
file='ATC'

CUDA_VISIBLE_DEVICES=0 /data/yljiang/anaconda3/envs/py3/bin/python3 train_mygcn_FB.py \
 --batch_size 4 \
 --n_hid1 128 \
 --n_hid2 128 \
 --n_expert 256 \
 --att_hid 256 \
 --steps 200 \
 --learning_rate 1e-3 \
 --verbose True \
 --extra_feats 0 \
 --weight_decay 4e-5 \
 --normalization NormAdj \
 --dropout 0.5 \
 --input_data_folder ~/data/Email \
 --b $b \
 --k $k \
 --epoch 300 \
 --data_set 686 \
 --dim 63 


