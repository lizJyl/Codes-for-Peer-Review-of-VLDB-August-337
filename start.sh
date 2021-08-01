!/bin/bash
k=40
b=20
file='ATC'
# file='Abla_noG'
# --weight_decay 5e-3 \
# python3 train_mygcn_FB_unsup.py \
python3 train_mygcn_FB_AllOComSplit.py \
 --batch_size 8\
 --n_hid1 256 \
 --n_hid2 128 \
 --n_expert 256 \
 --att_hid 256 \
 --steps 200 \
 --learning_rate 1e-3 \
 --verbose True \
 --extra_feats 0 \
 --weight_decay 4e-5 \
 --normalization i_norm \
 --dropout 0.5 \
 --input_data_folder ~/data/Email \
 --b $b \
 --k $k \
 --epoch 300 \
 --data_set 414 \
 --ego_data facebook \
 --dim 562 \
 --model_dir snapshot/gcn_F_${k}_${b}.pt #>>./Log/QfromNode/${file}/log_414.txt

#  CUDA_VISIBLE_DEVICES=0 /data/yljiang/anaconda3/envs/py3/bin/python3 train_mygcn_FB.py \
#  --batch_size 4 \
#  --n_hid1 128 \
#  --n_hid2 128 \
#  --n_expert 256 \
#  --att_hid 256 \
#  --steps 200 \
#  --learning_rate 1e-3 \
#  --verbose True \
#  --extra_feats 0 \
#  --weight_decay 4e-5 \
#  --normalization NormAdj \
#  --dropout 0.5 \
#  --input_data_folder ~/data/Email \
#  --b $b \
#  --k $k \
#  --epoch 300 \
#  --data_set 686 \
#  --dim 63 \
#  --model_dir snapshot/gcn_F_${k}_${b}.pt >>./Log/QfromNode/${file}/log_686.txt

#  CUDA_VISIBLE_DEVICES=0 /data/yljiang/anaconda3/envs/py3/bin/python3 train_mygcn_FB.py \
#  --batch_size 4 \
#  --n_hid1 128 \
#  --n_hid2 128 \
#  --n_expert 256 \
#  --att_hid 256 \
#  --steps 200 \
#  --learning_rate 1e-3 \
#  --verbose True \
#  --extra_feats 0 \
#  --weight_decay 4e-5 \
#  --normalization NormAdj \
#  --dropout 0.5 \
#  --input_data_folder ~/data/Email \
#  --b $b \
#  --k $k \
#  --epoch 300 \
#  --data_set 348 \
#  --dim 161 \
#  --model_dir snapshot/gcn_F_${k}_${b}.pt >>./Log/QfromNode/${file}/log_348.txt

#  CUDA_VISIBLE_DEVICES=0 /data/yljiang/anaconda3/envs/py3/bin/python3 train_mygcn_FB.py \
#  --batch_size 4 \
#  --n_hid1 128 \
#  --n_hid2 128 \
#  --n_expert 256 \
#  --att_hid 256 \
#  --steps 200 \
#  --learning_rate 1e-3 \
#  --verbose True \
#  --extra_feats 0 \
#  --weight_decay 4e-5 \
#  --normalization NormAdj \
#  --dropout 0.5 \
#  --input_data_folder ~/data/Email \
#  --b $b \
#  --k $k \
#  --epoch 300 \
#  --data_set 0 \
#  --dim 224 \
#  --model_dir snapshot/gcn_F_${k}_${b}.pt >>./Log/QfromNode/${file}/log_0.txt

#  CUDA_VISIBLE_DEVICES=0 /data/yljiang/anaconda3/envs/py3/bin/python3 train_mygcn_FB.py \
#  --batch_size 4 \
#  --n_hid1 128 \
#  --n_hid2 128 \
#  --n_expert 256 \
#  --att_hid 256 \
#  --steps 200 \
#  --learning_rate 1e-3 \
#  --verbose True \
#  --extra_feats 0 \
#  --weight_decay 4e-5 \
#  --normalization NormAdj \
#  --dropout 0.5 \
#  --input_data_folder ~/data/Email \
#  --b $b \
#  --k $k \
#  --epoch 300 \
#  --data_set 3437 \
#  --dim 262 \
#  --model_dir snapshot/gcn_F_${k}_${b}.pt >>./Log/QfromNode/${file}/log_3437.txt


# CUDA_VISIBLE_DEVICES=0 /data/yljiang/anaconda3/envs/py3/bin/python3 train_mygcn_FB.py \
#  --batch_size 4 \
#  --n_hid1 128 \
#  --n_hid2 128 \
#  --n_expert 256 \
#  --att_hid 256 \
#  --steps 200 \
#  --learning_rate 1e-3 \
#  --verbose True \
#  --extra_feats 0 \
#  --weight_decay 4e-5 \
#  --normalization NormAdj \
#  --dropout 0.5 \
#  --input_data_folder ~/data/Email \
#  --b $b \
#  --k $k \
#  --epoch 300 \
#  --data_set 1912 \
#  --dim 480 \
#  --model_dir snapshot/gcn_F_${k}_${b}.pt >>./Log/QfromNode/${file}/log_1912.txt


#  CUDA_VISIBLE_DEVICES=0 /data/yljiang/anaconda3/envs/py3/bin/python3 train_mygcn_FB.py \
#  --batch_size 4 \
#  --n_hid1 128 \
#  --n_hid2 128 \
#  --n_expert 256 \
#  --att_hid 256 \
#  --steps 200 \
#  --learning_rate 1e-3 \
#  --verbose True \
#  --extra_feats 0 \
#  --weight_decay 4e-5 \
#  --normalization NormAdj \
#  --dropout 0.5 \
#  --input_data_folder ~/data/Email \
#  --b $b \
#  --k $k \
#  --epoch 300 \
#  --data_set 1684 \
#  --dim 319 \
#  --model_dir snapshot/gcn_F_${k}_${b}.pt >>./Log/QfromNode/${file}/log_1684.txt

#  CUDA_VISIBLE_DEVICES=0 /data/yljiang/anaconda3/envs/py3/bin/python3 train_mygcn_FB.py \
#  --batch_size 4 \
#  --n_hid1 128 \
#  --n_hid2 128 \
#  --n_expert 256 \
#  --att_hid 256 \
#  --steps 200 \
#  --learning_rate 1e-3 \
#  --verbose True \
#  --extra_feats 0 \
#  --weight_decay 4e-5 \
#  --normalization NormAdj \
#  --dropout 0.5 \
#  --input_data_folder ~/data/Email \
#  --b $b \
#  --k $k \
#  --epoch 300 \
#  --data_set 107 \
#  --dim 576 \
#  --model_dir snapshot/gcn_F_${k}_${b}.pt >>./Log/QfromNode/${file}/log_107.txt


#  #sh run_gcn_F_1node.sh >>log_3437_1.txt
# #sh run_gcn_F_1node.sh >>log_3437_2.txt
# #sh run_gcn_F_1node.sh >>log_3437_3.txt
# #sh run_gcn_F_1node.sh >>log_3437_4.txt
# #sh run_gcn_F_1node.sh >>log_3437_5.txt
# #
# #sh run_gcn_F_1node.sh >>log_0_1.txt
# #sh run_gcn_F_1node.sh >>log_0_2.txt
# #sh run_gcn_F_1node.sh >>log_0_3.txt
# #sh run_gcn_F_1node.sh >>log_0_4.txt
# #sh run_gcn_F_1node.sh >>log_0_5.txt
# #
# #sh run_gcn_F_1node.sh >>log_414_1.txt
# #sh run_gcn_F_1node.sh >>log_414_2.txt
# #sh run_gcn_F_1node.sh >>log_414_3.txt
# #sh run_gcn_F_1node.sh >>log_414_4.txt
# #sh run_gcn_F_1node.sh >>log_414_5.txt
# ##

# #sh run_gcn_F_1node.sh >>./Log/log_348_1node_1.txt
# #sh run_gcn_F_1node.sh >>./Log/log_348_1node_2.txt
# #sh run_gcn_F_1node.sh >>./Log/log_348_1node_3.txt
# #sh run_gcn_F_1node.sh >>./Log/log_348_1node_4.txt
# #sh run_gcn_F_1node.sh >>./Log/log_348_1node_5.txt

# #sh run_gcn_F_1node.sh >>./Log/log_686_1node_1.txt
# #sh run_gcn_F_1node.sh >>./Log/log_686_1node_2.txt
# #sh run_gcn_F_1node.sh >>./Log/log_686_1node_3.txt
# #sh run_gcn_F_1node.sh >>./Log/log_686_1node_4.txt
# #sh run_gcn_F_1node.sh >>./Log/log_686_1node_5.txt

# #sh run_gcn_F_1node.sh >>log_1684_1.txt
# #sh run_gcn_F_1node.sh >>log_1684_2.txt
# #sh run_gcn_F_1node.sh >>log_1684_3.txt
# #sh run_gcn_F_1node.sh >>log_1684_4.txt
# #sh run_gcn_F_1node.sh >>log_1684_5.txt

# #
# #sh run_gcn_F_1node.sh >>log_wisconsin_1.txt
# #sh run_gcn_F_1node.sh >>log_wisconsin_2.txt
# #sh run_gcn_F_1node.sh >>log_wisconsin_3.txt
# #sh run_gcn_F_1node.sh >>log_wisconsin_4.txt
# #sh run_gcn_F_1node.sh >>log_wisconsin_5.txt
