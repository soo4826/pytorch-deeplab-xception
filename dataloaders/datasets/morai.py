import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class MoraiDataset(data.Dataset):
    NUM_CLASSES = 14    # ['unlabeled', 'pedestrian', 'road-line', 'car']

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
        #  Mapping
        # self.classes = {0  : [ 246 ,   255 ,   22  ],
        #                 1  : [ 255 ,   160 ,   160 ],
        #                 # 1  : [ 255 ,   160 ,   160 ],
        #                 2  : [ 255 ,   148 ,   148 ],
        #                 # 2  : [ 255 ,   133 ,   133 ],
        #                 3  : [ 208 ,   110 ,   110 ],
        #                 # 3  : [ 208 ,   128 ,   128 ],
        #                 4  : [ 219 ,   133 ,   133 ],
        #                 # 4  : [ 219 ,   148 ,   148 ],
        #                 # 4  : [ 229 ,   152 ,   152 ],
        #                 # 4  : [ 229 ,   164 ,   164 ],
        #                 5  : [ 255 ,   255 ,   255 ],
        #                 # 5  : [ 255 ,   255 ,   0   ],
        #                 # 5  : [ 255 ,   255 ,   255 ],
        #                 # 5  : [ 255 ,   255 ,   255 ],
        #                 # 5  : [ 231 ,   187 ,   124 ],
        #                 6  : [ 167 ,   22  ,   255 ],
        #                 7  : [ 203 ,   255 ,   124 ],
        #                 8  : [ 187 ,   187 ,   187 ],
        #                 9  : [ 178 ,   218 ,   106 ],   
        #                 10 : [ 255 ,   170 ,   96  ],
        #                 11 : [ 255 ,   147 ,   248 ],
        #                 12 : [ 0   ,   255 ,   255 ]}
        # w/o Mapping
        # self.classes = {0  : [ 246 ,   255 ,   22  ], # Obstacle
        #                 ###################################################
        #                 1  : [ 255 ,   160 ,   160 ], # Truck
        #                 # 2  : [ 255 ,   160 ,   160 ], # Bus
        #                 ###################################################
        #                 3  : [ 255 ,   148 ,   148 ], # SUV
        #                 4  : [ 255 ,   133 ,   133 ], # Sedan
        #                 ###################################################
        #                 5  : [ 208 ,   110 ,   110 ], # 자전거 (Bicycle)
        #                 6  : [ 208 ,   128 ,   128 ], # 자전거 타는 사람 (Cyclist)
        #                 ###################################################
        #                 7  : [ 219 ,   133 ,   133 ], # 스쿠터 (Scooter)
        #                 8  : [ 219 ,   148 ,   148 ], # 스쿠터 타는 사람 
        #                 9  : [ 229 ,   152 ,   152 ], # 오토바이
        #                 10 : [ 229 ,   164 ,   164 ], # 오토바이 타는 사람
        #                 ###################################################
        #                 11 : [ 255 ,   255 ,   255 ], # 흰색 차선 (White Line)
        #                 12 : [ 255 ,   255 ,   0   ], # 노란색 차선 (Yellow Line)
        #                 13 : [ 255 ,   255 ,   255 ], # 횡단보도 (Cross Walk)
        #                 14 : [ 255 ,   255 ,   255 ], # 정지선 (Stop Line)
        #                 15 : [ 231 ,   187 ,   124 ], # 표지판 (Traffic Sign)
        #                 16 : [ 167 ,   22  ,   255 ], # Pedestrain
        #                 # # ###################################################
        #                 17 : [ 203 ,   255 ,   124 ], # Building
        #                 18 : [ 187 ,   187 ,   187 ], # Asphalt
        #                 19 : [ 178 ,   218 ,   106 ], # Standing Object (식생)
        #                 20 : [ 255 ,   170 ,   96  ], # Side Walk
        #                 21 : [ 255 ,   147 ,   248 ], # Traffic Light
        #                 22 : [ 0   ,   255 ,   255 ]} # Sky

        self.classes = {  0 : [   0 ,     0 ,     0 ], # Unlabeled
                        # 523 : [ 246 ,   255 ,    22 ], # Obstacle
                        ###################################################
                        575 : [ 255 ,   160 ,   160 ], # Truck & Bus
                        ###################################################
                        551 : [ 255 ,   148 ,   148 ], # SUV
                        # 521 : [ 255 ,   133 ,   133 ], # Sedan
                        ###################################################
                        428 : [ 208 ,   110 ,   110 ], # 자전거 (Bicycle)
                        # 464 : [ 208 ,   128 ,   128 ], # 자전거 타는 사람 (Cyclist)
                        ###################################################
                        485 : [ 219 ,   133 ,   133 ], # 스쿠터 (Scooter)
                        # 515 : [ 219 ,   148 ,   148 ], # 스쿠터 타는 사람 
                        # 533 : [ 229 ,   152 ,   152 ], # 오토바이
                        # 557 : [ 229 ,   164 ,   164 ], # 오토바이 타는 사람
                        ###################################################
                        765 : [ 255 ,   255 ,   255 ], # 흰색 차선 (White Line) & 횡단보도 (Cross Walk) & 정지선 (Stop Line)
                        # 510 : [ 255 ,   255 ,   0   ], # 노란색 차선 (Yellow Line)
                        # 542 : [ 231 ,   187 ,   124 ], # 표지판 (Traffic Sign)
                        444 : [ 167 ,   22  ,   255 ], # Pedestrain
                        # # ###################################################
                        582 : [ 203 ,   255 ,   124 ], # Building
                        561 : [ 187 ,   187 ,   187 ], # Asphalt
                        502 : [ 178 ,   218 ,   106 ], # Standing Object (식생)
                        521 : [ 255 ,   170 ,   96  ], # Side Walk
                        650 : [ 255 ,   147 ,   248 ], # Traffic Light
                        540 : [ 167 ,   120 ,   253 ], # Traffic Sign
                        510 : [ 0   ,   255 ,   255 ]} # Sky
        
        # self.void_classes = [523, 765, 510, 542, 582, 561, 502, 521, 650, 510] # 10
        # self.valid_classes = [0, 575, 551, 521, 428, 464, 485, 515, 533, 557, 444] # 11
        self.void_classes = [] # 10
        self.valid_classes = [0, 575, 551, 428, 485, 765, 444, 582, 561, 502, 521, 650, 540, 510] # 14
        # self.void_classes = [] # 12
        # self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # 11
        # self.class_names = ['unlabeled', 'Obstacle', 'TruckBus', 'SUV', 'Sedan', 'Bicycle', 'Cyclist', 'Scooter', 'Scooter_p',\
        #                     'Motorcycle', 'Motorcycle_p', 'Whiteline', 'Yellowline', 'Crosswalk',\
        #                     'Stopline', 'Trafficsign', 'Pedestrain', 'Building', 'Asphalt', 'Standingobject',\
        #                     'Sidewalk', 'Trafficlight', 'Sky']
        # self.class_names = ['unlabeled', 'TruckBus', 'SUV', 'Sedan', 'Bicycle', 'Cyclist', 'Scooter', 'Scooter_p',\
        #                     'Motorcycle', 'Motorcycle_p', 'Pedestrain']
        self.class_names = ['unlabeled', 'TruckBus', 'SUVSedan', 'Bicycle', 'Scooter', 'Lanemarking',\
                             'Pedestrain', 'Building', 'Asphalt', 'Standingobject', 'Sidewalk', 'Trafficlight', 'TrafficSign', 'Sky']
        self.ignore_index = 0
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        # print(dict(zip(self.valid_classes, range(self.NUM_CLASSES))))

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
        # lbl_path = os.path.join(self.annotations_base, img_path.split('/')[-1].split('_')[0]+"_gtFine_color.png")#, os.path.basename(img_path))

        _img = Image.open(img_path).convert('RGB')
        # _img = np.array(Image.open(img_path), dtype=np.uint8)
        # print(_img.shape)
        # _img = _img[:, :, 0:3]
        # print(_img.shape)
        # _img = Image.fromarray((_img).astype(np.uint8))
        # print(_img.shape)
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        # print(lbl_path)
        # print(_tmp.shape)
        
        # _tmp = _tmp[:, :, 0:3]
        _tmp = np.sum(_tmp, axis=2)
        # _tmp = np.min(_tmp, axis=2)
        # print(_tmp.shape)
        # print(_tmp[:,1])
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
            # print(mask[mask == _voidc])
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
            # print(self.valid_classes)
            # print(self.class_map[_validc])
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

