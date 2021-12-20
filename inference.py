from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tr

from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load("/home/ailab/Desktop/MORAI_DATA_V3.0/DATASET_COMPLETE/pytorch-deeplab-xception/deeplab-resnet.pth")
# checkpoint = torch.load("/home/ailab/Desktop/MORAI_DATA_V3.0/DATASET_COMPLETE/pytorch-deeplab-xception/run/morai/deeplab-mobilenet/experiment_1/7770.pth.tar")
# checkpoint = torch.load("/home/ailab/Desktop/MORAI_DATA_V3.0/DATASET_COMPLETE/pytorch-deeplab-xception/run/morai/deeplab-resnet/experiment_0/7868.pth.tar")
print(checkpoint)
model = DeepLab(num_classes=21,
                backbone='resnet',
                output_stride=16,
                sync_bn=True,
                freeze_bn=False)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

def transform(image):
    return tr.Compose([
        # tr.Resize(513),
        # tr.CenterCrop(513),
        # tr.RandomGaussianBlur(),
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)

torch.set_grad_enabled(False)

# image = Image.open('/home/ailab/Desktop/MORAI_DATA_V3.0/11월_보고/rgb/21-11-27-10-37-43-366892_Intensity.png').convert('RGB')
# image = Image.open('/home/ailab/Desktop/MORAI_DATA_V3.0/11월_보고/rgb/21-11-27-10-37-43-366892_Intensity.png').convert('RGB')
image = Image.open('/home/ailab/catkin_ws/src/time_sync/savefile/sungnam/1_2/img/302.png').convert('RGB')
inputs = transform(image).to(device)
output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
pred = np.argmax(output, axis=0)

# image.save("/home/ailab/Desktop/MORAI_DATA_V3.0/11월_보고/infer/21-11-27-10-37-43-366892_Intensity.png",'PNG')
# Then visualize it:
infer = decode_segmap(pred, dataset="cityscapes", plot=True)
save_img = Image.fromarray(infer.astype('uint8'))
save_img.save("/home/ailab/Desktop/MORAI_DATA_V3.0/11월_보고/infer/real_infer_test.png")