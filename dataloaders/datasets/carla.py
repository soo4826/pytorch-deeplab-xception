import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class CarlaDataset(data.Dataset):
    NUM_CLASSES = 23

    def __init__(self, args, root=Path.db_root_dir('carla'), split="train"):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        '''
        <folder tree>
        /path/to/carla
            train
                Town01
                Town04
                Town10
            val
                Town03
            test
                Town05
                Town02

        <image nameing>
        rgb: <unique 6-digit number 000000~050000.png 
        seg: 21-09-15-02-35-13-160170_Semantic.png 
        이므로 폴더 이름으로 GT 로딩
        '''
        self.images_base = os.path.join(self.root, self.split, 'img')
        # print(self.images_base)
        self.annotations_base = os.path.join(self.root, self.split, 'seg')

        # 데이터셋 경로 상의 모든 파일을 train / test / val 각각으로 불러옴
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.classes = {  0	:	[	0	, 0	    , 0 	],	 # unlabeled     =   0
                        210	:	[	70	, 70	, 70	],	 # building      =   1
                         70	:	[	0	, 0 	, 70	],	 # PTW           =   2
                        225	:	[	55	, 90	, 80	],	 # other         =   3
                        300	:	[	220	, 20	, 60	],	 # pedestrian    =   4
                        459	:	[	153	, 153	, 153	],	 # pole          =   5
                        441	:	[	157	, 234	, 50	],	 # road line     =   6
                        320	:	[	128	, 64	, 128	],	 # road          =   7
                        511	:	[	244	, 35	, 232	],	 # sidewalk      =   8
                        230	:	[	0	, 0	    , 230	],	 # CommVeh       =   9
                        142	:	[	0	, 0	    , 142	],	 # vehicle       =  10
                        360	:	[	102	, 102	, 156	],	 # wall          =  11
                        440	:	[	220	, 220	, 0  	],	 # traffic sign  =  12
                        380	:	[	70	, 130	, 180	],	 # sky           =  13
                        162	:	[	81	, 0	    , 81    ],	 # ground        =  14
                        350	:	[	150	, 100	, 100	],	 # bridge        =  15
                        520	:	[	230	, 150	, 140	],	 # rail track    =  16
                        525	:	[	180	, 165	, 180	],	 # guard rail    =  17
                        450	:	[	250	, 170	, 30	],	 # traffic light =  18
                        460	:	[	110	, 190	, 160	],	 # static        =  19
                        340	:	[	170	, 120	, 50	],	 # dynamic       =  20
                        255	:	[	45	, 60	, 150	],	 # Bicycle       =  21
                        415	:	[	145	, 170	, 100	]}	 # terrain       =  22

        self.void_classes = [] # 10
        self.valid_classes = [0, 210, 70, 225, 300, 459, 441, 320, 511, 230, 142, 360, 440, 380, 162, 350, 520, 525, 450, 460, 340, 255, 415]
        self.class_names = ['unlabeled', 'building', 'PTW', 'other', 'pedestrian', 'pole', 'roadline', 'road', 'sidewalk', 'CommVeh', 'vehicle', \
                            'wall', 'trafficsign', 'sky', 'ground', 'bridge', 'railtrack', 'guardrail', 'trafficlight', 'static', 'dynamic', 'Bicycle', 'terrain']            

                
        self.ignore_index = 0
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        img_path.split()
        lbl_path = os.path.join(self.annotations_base, img_path.split('/')[-1])

        _img = Image.open(img_path)

        _tmp = np.array(Image.open(lbl_path))

        _tmp = np.sum(_tmp, axis=2)

        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray((_tmp).astype(np.uint8))
        
        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)



    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
            
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    carla_train = CarlaDataset(args, split='train')

    dataloader = DataLoader(carla_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='carla')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
