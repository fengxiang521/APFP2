gpuid=0
cd ../../

DATA_ROOT=/root/autodl-tmp/DataSet/mini-ImageNet
MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/5way_1shot.tar
MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/5way_5shot.tar


# N_SHOT=1
# python test.py --targetdataset mini_imagenet --origindataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_1SHOT_PATH --test_task_nums 5 --test_n_episode 2000 --tem 0.008

N_SHOT=5
python test.py --targetdataset mini_imagenet --origindataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_5SHOT_PATH --test_task_nums 5 --test_n_episode 2000 --tem 0.008
