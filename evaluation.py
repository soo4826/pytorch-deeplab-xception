import argparse
from base64 import decode
import os
import numpy as np
import dataloaders 
from tqdm import tqdm
import torch
import json


from PIL import Image
from modeling.deeplab import *
from dataloaders import utils, make_data_loader
from dataloaders.datasets import carla
from utils.metrics import Evaluator
from torchvision import transforms
import torchvision.transforms as tr
from torch.utils.data import DataLoader
from dataloaders.utils import decode_segmap
class Tester(object):
    def __init__(self, args):
        if not os.path.isfile(args.parameter):
            raise RuntimeError("no checkpoint found at '{}'".format(args.parameter))
        self.args = args
        self.color_map = utils.get_carla_labels()
        self.nclass = int(args.num_class)
        self.save_result = args.save_result
        self.save_path = args.save_path
        self.batch_size=args.batch_size
        # Init dataloader
        test_set = carla.CarlaDataset(args, split='test')
        self.class_name = test_set.class_names
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        #Define model
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=False,
                        freeze_bn=False)
        
        self.model = model.cuda()
        device = torch.device('cpu')
        checkpoint = torch.load(args.parameter, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.evaluator = Evaluator(self.nclass)
        # print(self.model)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        

    def save_image(self, path, img, save_num):
        decode_pred = decode_segmap(img, dataset='carla', plot=False)
        # print(np.max(decode_pred.reshape(-1, 1)))
        save_img = Image.fromarray(decode_pred.astype('uint8'))
        save_img.save(os.path.join(path, str(save_num).zfill(6) + str('.png')))

    def transform(self):
        return tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        
    def inference(self):
        self.model.eval()
        self.evaluator.reset()
        save_num = 1
        for sample in (tqdm(self.test_loader)):        
            test_img, test_gt = sample['image'].cuda(), sample['label'].cuda()
            
            torch.set_grad_enabled(False)
            with torch.no_grad():
                output = self.model(test_img)
            pred = output.data.cpu().numpy()
            pred_ = pred.argmax(axis=1)
            
            # pred_ = pred.argmax(axis=0) # For batch size is not 1
            gt = test_gt.cpu().numpy()
            # print(gt.shape, pred_.shape)
            
            # For evaluation 
            self.evaluator.add_batch(gt, pred_)
            self.evaluator.Mean_Intersection_over_Union()
            
            # For save prediction result
            if self.save_result:
                assert self.batch_size == 1, "For save inference file, batch_size should be 1"

                pred_save = np.argmax(output.squeeze().cpu().numpy(), axis=0)
                self.save_image(self.save_path, pred_save, save_num)
            
            save_num= save_num+1
            print(self.class_name)
        
        print("========Model Evaluation Result========")
        print(self.class_name)
        self.evaluator.Mean_Intersection_over_Union()


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Evaluation")
    parser.add_argument('--backbone', type=str, default='resnet',
                    choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='pascal',
                    choices=['pascal', 'coco', 'cityscapes', 'morai', 'carla'],
                    help='dataset name (default: pascal)')
    parser.add_argument('--parameter', type=str, default='./checkpoint.pth',
                    help='model parameter path  (default: ./checkpoint.pth)')
    parser.add_argument('--num-class', type=str, default='6',
                    help='number of class, carla_dynamic: 6 / carla_full: 23  (default: 6)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--save-path', type=str, default='./inference',
                help='save inference result into (default: ./inference)')
    parser.add_argument('--save-result', type=int, default=0,
                        help='whether to save inference result (defalut: 0, do not save)')
    
    args = parser.parse_args()
    
    tester = Tester(args)

    tester.inference()

if __name__ == "__main__":
    main()