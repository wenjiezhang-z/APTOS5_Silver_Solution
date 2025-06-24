cd train_scripts_M75

python3 train.py phase --split training --bn_off --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name M_75 --cfg configs/ophnet_M75.yaml

python3 train.py phase --split training --bn_off --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name CMR_2 --cfg configs/ophnet_CMR_2.yaml

python3 train.py phase --split training --bn_off --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name IUUU --cfg configs/ophnet_IUUU.yaml

python3 train.py phase --split training --bn_off --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name IUUUU --cfg configs/ophnet_IUUUU.yaml

python3 train.py phase --split training --bn_off --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name O2M --cfg configs/ophnet_O2M.yaml

python3 train.py phase --split training --bn_off --backbone convnext --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name convnext --cfg configs/ophnet_M75.yaml

python3 train.py phase --split training --bn_off --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name ophnet_Iuu --cfg configs/ophnet_Iuu.yaml

python3 train.py phase --split training --bn_off --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name ophnet_Iuuuu --cfg configs/ophnet_Iuuuu.yaml

cd ..
cd train_scripts_Convnextv2_Resnet

python3 train.py phase --split training --bn_off --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name covnetv2_Resnet --cfg configs/ophnet.yaml

cd ..
cd train_scripts_swin

python3 train.py phase --split training --bn_off --backbone swintransformer --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name swintransformer --cfg configs/ophnet.yaml

python3 train.py phase --split training --bn_off --backbone swintransformer --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name ntoken200 --cfg configs/ophnet_ntoken200.yaml

cd ..
cd train_scripts_swinv2

python3 train.py phase --split training --bn_off --backbone swintransformerv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name swintransformerv2 --cfg configs/ophnet_base.yaml

python3 train.py phase --split training --bn_off --backbone swintransformerv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name swintransformerv2_iuuu --cfg configs/ophnet_base_iuuu.yaml

python3 train.py phase --split training --bn_off --backbone swintransformerv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name swintransformerv2_iuuuu --cfg configs/ophnet_base_iuuuu.yaml


cd ..
cd train_scripts_swin_moe

python3 train.py phase --split training --bn_off --backbone swin_moe --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name swin_moe --cfg configs/ophnet.yaml

cd ..
cd train_scripts_vHeat

python3 train.py phase --split training --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name vheat --cfg configs/ophnet_base.yaml

python3 train.py phase --split training --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name vheat_on_tm --cfg configs/ophnet_base_no_tm.yaml

python3 train.py phase --split training --bn_off --backbone vheat123 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name vHeat_freeze01 --cfg configs/ophnet_base.yaml

python3 train.py phase --split training --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name vheat_Bins_20 --cfg configs/ophnet_base_B20.yaml

cd ..
cd train_scripts_vHeat_swin

python3 train.py phase --split training --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name vheat_swin --cfg configs/ophnet.yaml

cd ..
cd train_scripts_vHeat_swinv2_1

python3 train.py phase --split training --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name vheat_swinv2 --cfg configs/ophnet.yaml

cd ..
cd train_scripts_load_feature

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_only_temporal --only_temporal --cfg configs/ophnet_base.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_NO_Tem_fM2 --only_temporal --cfg configs/ophnet_base_m2.yaml 

python3 train.py phase --split training  --bn_off --backbone swintransformerv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Swinv2_No_TM_CMR --only_temporal --cfg configs/ophnet_base_No_TM_CMR.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_No_TM_CMR --only_temporal --cfg configs/ophnet_base_No_TM_CMR.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_only_Tem_iuu --only_temporal --cfg configs/ophnet_base_iuu.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_only_Tem_iuuu --only_temporal --cfg configs/ophnet_base_iuuu.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_only_Tem_iuuuuu --only_temporal --cfg configs/ophnet_base_iuuuuu.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_NO_Tem_bin8 --only_temporal --cfg configs/ophnet_base_Bin8.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_NO_Tem_bin12 --only_temporal --cfg configs/ophnet_base_Bin12.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_NO_Tem_bin20 --only_temporal --cfg configs/ophnet_base_Bin20.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_NO_Tem_mtoken100 --only_temporal --cfg configs/ophnet_base_mtoken100.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_NO_Tem_mtoken50 --only_temporal --cfg configs/ophnet_base_mtoken50.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_NO_Tem_mtoken200 --only_temporal --cfg configs/ophnet_base_mtoken200.yaml

python3 train.py phase --split training  --bn_off --backbone swintransformerv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Swinv2_only_Tem_Bin8 --only_temporal --cfg configs/ophnet_base_Bin8.yaml

python3 train.py phase --split training  --bn_off --backbone vheat --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Vheat_NO_Tem_ahead16_layer8 --only_temporal --cfg configs/ophnet_base_ahead16.yaml
