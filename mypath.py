class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/home/ailab/datasets/cityscapes/'  # folder that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'synthetic_parkinglot':
            # return '/data/Synthetic_ParkingLot/synthetic_parkinglot_test'
            return '/data/Synthetic_ParkingLot/synthetic_parkinglot'
        elif dataset == 'morai':
            return '/home/ailab/jinsu/imageSegmentation/image_full'
        elif dataset == 'carla':
            # return '/home/ailab/Desktop/CARLA_DATASET_V1.0'git 
            return '/path/to/datasets/carla'
        
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError