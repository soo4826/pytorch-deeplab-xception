CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone resnet --lr 0.005 --workers 4 --epochs 150 --batch-size 8 --gpu-ids 0,1 --checkname deeplab-resnet-full --eval-interval 1 --dataset carla --resume /home/happy/deeplab/CARLA_v2/pytorch-deeplab-xception/run/carla/deeplab-resnet-full/experiment_0/checkpoint.pth.tar