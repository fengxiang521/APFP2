gpuid=0
N_SHOT=5

DATA_ROOT=../../DataSet/cub # path to the json file of CUB
#MODEL_PATH=./checkpoints/cub/5way_1shot.tar
MODEL_PATH=./checkpoints/cub/5way_5shot.tar
cd ../../

python test.py --targetdataset cub --origindataset cub --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 --tem 0.072
