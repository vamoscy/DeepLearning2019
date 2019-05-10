import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Architecture
        # TODO

        # Load pre-trained model
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.fc1_bn = nn.BatchNorm1d(1000)
        # self.fc1 = nn.Linear(4096, 2048)
        # self.fc2 = nn.Linear(2048, 1000)
        self.fc1 = nn.Linear(512, 1000)
        self.load_weights('weights.pth')

    def load_weights(self, pretrained_model_path, cuda=True):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda else "cpu")

        # Load pre-trained weights in current model
        with torch.no_grad():
            self.load_state_dict(pretrained_model, strict=True)

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    def forward(self, x):
        # TODO
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_bn(self.conv4(x)), 2))
        x = F.avg_pool2d(x,1)
        x = x.view(-1, 512)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return F.log_softmax(x)
