 python3.7 train.py \
    --checkname FINETUNE_SSD300\
    --square_make_type pad\
    --data_config configs/data_configs/three_drones_real.data \
    --batch_size 12\
    --multiscale_training False\
    --pretrained_weights pretrained_ckpt/best_checkpoint.pth.tar \
    #--resume meta_data/checkpoints/three_drones/TRAIN_SSD300/experiment_0/best_checkpoint.pth.tar \


