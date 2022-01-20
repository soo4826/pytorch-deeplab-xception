import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class MoraiDataset(data.Dataset):
    NUM_CLASSES = 13     # ['unlabeled', 'pedestrian', 'road-line', 'car']

    def __init__(self, args, root=Path.db_root_dir('morai'), split="train"):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        '''
        <folder tree>
        /path/to/morai
            rgb ( data 비율 정해야 함! )
                train
                    scene 또는 seq별로 분류
                test
                val
            seg ( rgb 디렉토리 파일구조 동일)
                train
                test
                val
        
        <image nameing>
        rgb: 21-09-15-02-35-13-160170_Intensity.png 
        seg: 21-09-15-02-35-13-160170_Semantic.png 
        이므로 뒤에 있는 _Intensity / _Semantic 이용해서 분류 진행
        '''
        self.images_base = os.path.join(self.root, self.split, 'rgb')
        # print(self.images_base)
        self.annotations_base = os.path.join(self.root, self.split, 'seg')

        # 데이터셋 경로 상의 모든 파일을 train / test / val 각각으로 불러옴
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')
        
        # MORAI Classes (Lidar / Camera 동일)
        """
        self.classes = {  
                        # class ID(Sum of RGB): [ R, G, B ] 형태로 저장
                        510: [ 0  , 255, 255],  # Sky
                        149: [ 85 , 22 , 42 ],  # ETC
                        473: [ 0  , 218, 255],  # Blue_Lane
                        561: [ 187, 187, 187],  # Asphalt
                        582: [ 203, 255, 124],  # Building
                        650: [ 255, 147, 248],  # Traffic_Light
                        765: [ 255, 255, 255],  # White_Lane
                        510: [ 255, 255, 0  ],  # Yellow_Lane
                        542: [ 231, 187, 124],  # Road_Sign
                        551: [ 148, 255, 148],  # Crosswalk
                        255: [ 255, 0  , 0  ],  # Stop_Line
                        521: [ 255, 170, 96 ],  # Sidewalk (only walkable side)
                        502: [ 178, 218, 106],  # Standing_OBJ
                        523: [ 246, 255, 22 ],  # Object_On_Road
                        299: [ 255, 22 , 22 ],  # Vehicle
                        444: [ 167, 22 , 255],  # Pedestrian
                        375: [ 255, 60 , 60 ],  # Sedan
                        405: [ 255, 75 , 75 ],  # SUV
                        435: [ 255, 90 , 90 ],  # Truck
                        465: [ 255, 105, 105],  # Bus
                        495: [ 255, 120, 120],     # Van
                        120: [ 120, 0  , 0  ],  # Stroller
                        150: [ 120, 15 , 15 ],  # Stroller_person
                        180: [ 140, 20 , 20 ],  # ElectronicScooter
                        210: [ 140, 35 , 35 ],  # ElectronicScooter_Person
                        240: [ 160, 40 , 40 ],  # Bicycle
                        270: [ 160, 55 , 55 ],  # Bicycle_Person
                        300: [ 180, 60 , 60 ],  # Motorbike
                        330: [ 180, 75 , 75 ],  # Motorbike_Person
                        360: [ 200, 80 , 80 ],  # Sportbike
                        390: [ 200, 95 , 95 ]}  # Sportbike_Person
        #  Custom Classes
        self.void_classes = [510, 149, 473, 561, 582, 650, 765, 510, 542, 551, 255, 521, 502, 523, 444, 375, 405, 435, 465, 495, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390]
        # self.valid_classes = [444, 299, 375, 405, 435, 465, 495, 240, 270, 30, 330, 360, 390, 582, 149]
        
        # ORIGINAL
        # self.valid_classes = [510, 149, 473, 561, 582, 650, 765, 510, 542, 551, 255, 521, 502, 523, 299, 444, 375, 405, 435, 465, 495, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390]
        # self.class_names = ['Sky', 'ETC', 'Blue_Lane', 'Asphalt', 'Building', 'Traffic_Light', 'White_Lane', 'Yellow_Lane', 'Road_Sign', 'Crosswalk', 'Stop_Line', 'Sidewalk', 'Standing_OBJ', 'Object_On_Road', 'Vehicle', 'Pedestrian', 'Sedan', 'SUV', 'Truck', 'Bus', 'Van', 'Stroller', 'Stroller_person', 'ElectronicScooter', 'ElectronicScooter_Person', 'Bicycle', 'Bicycle_Person', 'Motorbike', 'Motorbike_Person', 'Sportbike', 'Sportbike_Person']
        
        self.valid_classes = [299, 444, 561, 149]
        self.class_names = ['CAR', 'Ped', 'Asphalt', 'ETC']
        
        # cityscapes:
        # self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
        #                     'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
        #                     'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
        #                     'motorcycle', 'bicycle']
        # parkinglot ex
        # self.void_classes = [210, 496, 580, 459, 511, 284, 360, 440, 320]
        # self.valid_classes = [0, 300, 441, 142]
        # self.class_names = ['unlabeled', 'pedestrian', 'road-line', 'car']
"""

        # self.classes = {1: [255    160    160]
        #                 0: [   0,   0,   0],  # outlier
        #                 1 :[ 70,  70,  70],  # building
        #                 2 :[ 190, 153, 153],  # fence
        #                 3 :[ 250, 170, 160],  # other
        #                 4 :[ 220,  20,  60],  # pedestrian
        #                 5 :[ 153, 153, 153],  # pole
        #                 6 :[ 157, 234,  50],  # road-line
        #                 7 :[ 128,  64, 128],  # road
        #                 8 :[ 244,  35, 232],  # sidewalk
        #                 9 :[ 107, 142,  35],  # vegetation
        #                 10: [   0,   0, 142],  # car
        #                 11: [ 102, 102, 156],  # wall
        #                 12 :[ 220, 220,   0]}  # traffic-sign

        self.classes = {0  : [ 246 ,   255 ,   22  ],
                        1  : [ 255 ,   160 ,   160 ],
                        # 1  : [ 255 ,   160 ,   160 ],
                        2  : [ 255 ,   148 ,   148 ],
                        # 2  : [ 255 ,   133 ,   133 ],
                        3  : [ 208 ,   110 ,   110 ],
                        # 3  : [ 208 ,   128 ,   128 ],
                        4  : [ 219 ,   133 ,   133 ],
                        # 4  : [ 219 ,   148 ,   148 ],
                        # 4  : [ 229 ,   152 ,   152 ],
                        # 4  : [ 229 ,   164 ,   164 ],
                        5  : [ 255 ,   255 ,   255 ],
                        # 5  : [ 255 ,   255 ,   0   ],
                        # 5  : [ 255 ,   255 ,   255 ],
                        # 5  : [ 255 ,   255 ,   255 ],
                        # 5  : [ 231 ,   187 ,   124 ],
                        6  : [ 167 ,   22  ,   255 ],
                        7  : [ 203 ,   255 ,   124 ],
                        8  : [ 187 ,   187 ,   187 ],
                        9  : [ 178 ,   218 ,   106 ],   
                        10 : [ 255 ,   170 ,   96  ],
                        11 : [ 255 ,   147 ,   248 ],
                        12 : [ 0   ,   255 ,   255 ]}



        self.void_classes = []
        self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.class_names = ['outlier', 'com_car', 'ind_car', 'bicycle', 'scooter', 'lane_marking', 'pedestrian',\
                            'building', 'asphalt', 'standing_object', 'side_walk', 'traffic_light', 'sky']
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # print("index: " + str(index))
        img_path = self.files[self.split][index].rstrip()
        # print( img_path.split('/')[-1].split('_')[0])
        img_path.split()

        lbl_path = os.path.join(self.annotations_base, img_path.split('/')[-1].split('_')[0]+"_Semantic.png")#, os.path.basename(img_path))

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
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

    morai_train = MoraiDataset(args, split='train')

    dataloader = DataLoader(morai_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='morai')
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

