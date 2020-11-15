 python3.7 demo_detection.py \
    --batch_size 32\
    --square_make_type pad\
    --checkname DEMO_SSD300\
    --test_img_dir example_images/three_drones/test_from_skydio_hd_ian_house \
    --pretrained_weights meta_data/checkpoints/three_drones/FINETUNE_SSD300/experiment_0/best_checkpoint.pth.tar \
    #--pretrained_weights pretrained_ckpt/best_checkpoint.pth.tar \


    #--test_img_dir tynguyen/github_workspaces/github_pytorch_object_detection/pytorch_object_detection/pytorch_yolov3/data/samples/three_drones/test_from_skydio_hd_ian_house \
    #--pretrained_weights meta_data/checkpoints/three_drones/TRAIN_SSD300/experiment_1/best_checkpoint.pth.tar \
