from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tr

from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load("/home/happy/deeplab/CARLA_v2/pytorch-deeplab-xception/run/carla/deeplab-resnet-full/model_best.pth.tar")
# print(checkpoint)
model = DeepLab(num_classes=23,
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
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)

torch.set_grad_enabled(False)

image = Image.open('/home/happy/Desktop/Dataset/CARLA/test/img/004539.png')
inputs = transform(image).to(device)
output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
pred = np.argmax(output, axis=0)
print(pred[0][0])

# Then visualize it:
decode_segmap(pred, dataset="carla", plot=True)