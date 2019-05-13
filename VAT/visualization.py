import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
layer = model.conv1
model.eval()
with torch.no_grad():
    img = Image.open('/home/yc3651/DS1008/ssl_data_96/supervised/val/n00015388/n00015388_10082.JPEG')
    transform = transforms.Compose(
        # [transforms.ToTensor()
            [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.Resize(130),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0.1),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
    )
    t_img = Variable(transform(img).unsqueeze(0))
    embedding = torch.zeros(47)
    def copy_data(m,i,o):
        embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    print(embedding)