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
            return '/home/ailab/Desktop/MORAI_DATA_V3.0/DATASET_COMPLETE/image_full'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
