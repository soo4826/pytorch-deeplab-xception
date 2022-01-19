# If --save-result set to True, batch_size should be 1
python evaluation.py --parameter /home/jieun/git/pytorch-deeplab-xception/checkpoint.pth.tar --dataset carla --batch-size 1 --save-path /home/jieun/git/pytorch-deeplab-xception/inference --save-result True
