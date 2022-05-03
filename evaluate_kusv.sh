# If --save-result set to True, batch_size should be 1
# python evaluation.py --parameter /home/user/jinsu/02_Semantic_Fusion/pytorch-deeplab-xception/run/kusv/kusv-deeplab-resnet/experiment_4/checkpoint.pth.tar --dataset kusv --batch-size 16 --save-path ./inference --save-result 0

python evaluation.py --parameter /home/user/jinsu/02_Semantic_Fusion/pytorch-deeplab-xception/run/kusv/kusv-deeplab-resnet/experiment_4/checkpoint.pth.tar --dataset kusv --batch-size 1 --save-path ./inference --save-result 1