# If --save-result set to True, batch_size should be 1
# python evaluation.py --parameter /path/to/model_parameter/parameter.pth --dataset carla --batch-size 16 --save-path /path/to/save_model --save-result 0

python evaluation.py --parameter /home/ailab/jinsu/imageSegmentation/pytorch-deeplab-xception/run/carla/deeplab-resnet/experiment_2/checkpoint.pth.tar --dataset carla --batch-size 1 --save-path ./inference --save-result 1