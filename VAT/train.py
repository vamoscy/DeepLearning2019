import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from vat import VATLoss
import data_utils
import utils
import math
import torchvision.models as models
from model import Model



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3)
        # self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3)
        # self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3)
        # self.conv4_3 = nn.Conv2d(256, 256, kernel_size=3)
        # self.conv4_4 = nn.Conv2d(256, 256, kernel_size=3)
        # self.dropout3 = nn.Dropout2d(p=0.5)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), kernel_size=3)
        x = F.relu(self.conv2_bn(self.conv2_1(x)))
        # x = F.relu(self.conv2_bn(self.conv2_2(x)))
        x = F.relu(self.conv3_bn(self.conv3_1(x)))
        x = self.dropout1(x)
        x = F.relu(self.conv3_bn(self.conv3_2(x)))
        # x = F.relu(self.conv3_bn(self.conv3_3(x)))
        x = F.relu(self.conv4_bn(self.conv4_1(x)))
        x = F.relu(self.conv4_bn(self.conv4_2(x)))
        x = self.dropout2(x)
        # x = F.relu(self.conv4_bn(self.conv4_3(x)))
        # x = F.relu(self.conv4_bn(self.conv4_4(x)))
        # x = self.dropout3(x)
        x = F.relu(self.conv5_bn(self.conv5_1(x)))
        x = F.relu(self.conv5_bn(self.conv5_2(x)))
        # x = F.relu(self.conv5_bn(self.conv5_3(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(args, model, device, data_loader_sup_train, data_loader_unsup, optimizer):
    model.train()

    ce_losses = utils.AverageMeter()
    vat_losses = utils.AverageMeter()
    prec1 = utils.AverageMeter()

    for batch_idx, (l, ul) in tqdm(enumerate(zip(data_loader_sup_train,data_loader_unsup))):
        x_l, y_l = l[0],l[1]
        x_ul, _ = ul[0], ul[1]
        x_l, y_l = x_l.to(device), y_l.to(device)
        x_ul = x_ul.to(device)

        optimizer.zero_grad()

        vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
        cross_entropy = nn.CrossEntropyLoss()

        lds = vat_loss(model, x_ul)
        output = model(x_l)
        classification_loss = cross_entropy(output, y_l)

        #uncomment following line and comment the second line below to use labeled and unlabeled data
        loss = classification_loss + args.alpha * lds
        # loss = classification_loss


        loss.backward()
        optimizer.step()

        acc = utils.accuracy(output, y_l)
        ce_losses.update(classification_loss.item(), x_l.shape[0])
        vat_losses.update(lds.item(), x_ul.shape[0])
        prec1.update(acc.item(), x_l.shape[0])
        if batch_idx % 100 == 0:
            print(f"\nBatch number : {batch_idx+1}\t"
                  f'CrossEntropyLoss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                  f'VATLoss {vat_losses.val:.4f} ({vat_losses.avg:.4f})\t'
                  f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})\t'
                  )



def evaluate(model, epoch, data_loader_sup_val, device, split, top_k=5):
    """ Method returning accuracy@1 and accuracy@top_k """
    print(f'\nEvaluating {split} set for epoch {epoch}...')
    model.eval()
    n_samples = 0.
    n_correct_top_1 = 0
    n_correct_top_k = 0
    i=0
    with torch.no_grad():
        for img, target in data_loader_sup_val:
            if i% 100 == 0:
                print(i)
            i+= 1
            img, target = img.to(device), target.to(device)
            batch_size = img.size(0)
            n_samples += batch_size

            # Forward
            output = model(img)
            cross_entropy = nn.CrossEntropyLoss()
            classification_loss = cross_entropy(output, target)

            # Top 1 accuracy
            pred_top_1 = torch.topk(output, k=1, dim=1)[1]
            n_correct_top_1 += pred_top_1.eq(target.view_as(pred_top_1)).int().sum().item()

            # Top k accuracy
            pred_top_k = torch.topk(output, k=top_k, dim=1)[1]
            target_top_k = target.view(-1, 1).expand(batch_size, top_k)
            n_correct_top_k += pred_top_k.eq(target_top_k).int().sum().item()

        # Accuracy
        top_1_acc = n_correct_top_1/n_samples
        top_k_acc = n_correct_top_k/n_samples

        # Log

        print(f'{split} top 1 accuracy: {top_1_acc:.4f}')
        print(f'{split} top {top_k} accuracy: {top_k_acc:.4f}')
        with open('test.csv', 'ab') as f:
            np.savetxt(f, np.array([epoch, classification_loss, top_1_acc,top_k_acc]) ,delimiter=",")

        return top_1_acc, top_k_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument('--xi', type=float, default=1e-6, metavar='XI',
                        help='hyperparameter of VAT (default: 10**-6)')
    parser.add_argument('--eps', type=float, default=8, metavar='EPS',
                        help='hyperparameter of VAT (default: 8.0)')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT (default: 1)')
    parser.add_argument('--workers', type=int, default=8, metavar='W',
                        help='number of CPU')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Net().to(device)
    model = nn.DataParallel(Model(), device_ids=[0,1,2,3]).to(device)
    # model = Model().to(device)
    curr_top_5_acc = 0
    for epoch in range(args.epochs):
        data_loader_sup_train, data_loader_sup_val, data_loader_unsup = data_utils.image_loader(
            path='../../ssl_data_96',
            batch_size=64,
        )
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        train(args, model, device, data_loader_sup_train, data_loader_unsup, optimizer)
        top_5_acc = evaluate(model, epoch, data_loader_sup_val, device, "Validation")[1]
        if top_5_acc - curr_top_5_acc < -0.005:
            print('early stopping')
            break
        else:
            curr_top_5_acc  = curr_top_5_acc
            torch.save(model.module.state_dict(), 'weights_new_VAT.pth')

if __name__ == '__main__':
    main()


    # for i in tqdm(range(args.iters)):
    #     if i % args.log_interval == 0:
    #         ce_losses = utils.AverageMeter()
    #         vat_losses = utils.AverageMeter()
    #         prec1 = utils.AverageMeter()
    #

        # # reset
        #
        #
        # x_l, y_l = next(data_iterators['labeled'])
        # x_ul, _ = next(data_iterators['unlabeled'])
        #
        # x_l, y_l = x_l.to(device), y_l.to(device)
        # x_ul = x_ul.to(device)
        #
        # optimizer.zero_grad()
        #
        # vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
        # cross_entropy = nn.CrossEntropyLoss()
        #
        # lds = vat_loss(model, x_ul)
        # output = model(x_l)
        #
        # classification_loss = cross_entropy(output, y_l)
        # loss = classification_loss + args.alpha * lds
        # loss.backward()
        # optimizer.step()
        #
        # acc = utils.accuracy(output, y_l)
        # ce_losses.update(classification_loss.item(), x_l.shape[0])
        # vat_losses.update(lds.item(), x_ul.shape[0])
        # prec1.update(acc.item(), x_l.shape[0])
        #
        #
        #
        # if i % args.log_interval == 0:
        #     print(f"\nIteration: {i}\t"
        #           f'CrossEntropyLoss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
        #           f'VATLoss {vat_losses.val:.4f} ({vat_losses.avg:.4f})\t'
        #           f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})\t'
        #           )
        #
        # if i > 0 and i % (args.log_interval * 100) == 0:
        #     torch.save(model.state_dict(), 'weights.pth')
