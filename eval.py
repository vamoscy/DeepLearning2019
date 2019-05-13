from model import Model
import argparse
import json
import torch
import time

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

def load_data(data_dir, batch_size, split):
    """ Method returning a data loader for labeled data """
    # TODO (optional): add data transformations if needed
    transform = transforms.Compose([
        # transforms.ToTensor()
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Resize(130),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    data = datasets.ImageFolder(f'{data_dir}/supervised/{split}', transform=transform)
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    return data_loader

def evaluate(model, data_loader, device, split, top_k=5):
    """ Method returning accuracy@1 and accuracy@top_k """
    print(f'\nEvaluating {split} set...')
    model.eval()
    n_samples = 0.
    n_correct_top_1 = 0
    n_correct_top_k = 0
    i = 0
    correct = []
    incorrect = []
    confusion_matrix = torch.zeros(1000, 1000)
    for img, target in data_loader:
        if i % 100 ==0:
            print(i)
        i+=1
        img, target = img.to(device), target.to(device)
        batch_size = img.size(0)
        n_samples += batch_size

        # Forward
        output = model(img)

        # Top 1 accuracy
        _, preds = torch.max(output, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        pred_top_1 = torch.topk(output, k=1, dim=1)[1]
        n_correct_top_1 += pred_top_1.eq(target.view_as(pred_top_1)).int().sum().item()
        # Top k accuracy
        pred_top_k = torch.topk(output, k=top_k, dim=1)[1]
        target_top_k = target.view(-1, 1).expand(batch_size, top_k)
        n_correct_top_k += pred_top_k.eq(target_top_k).int().sum().item()
    print(confusion_matrix.diagonal())
    a = confusion_matrix.diagonal().numpy()
    print(a.argsort()[-10:][::-1])
    print(a[a.argsort()[-10:][::-1]] / 64)
    # Accuracy
    top_1_acc = n_correct_top_1/n_samples
    top_k_acc = n_correct_top_k/n_samples

    # Log
    print(f'{split} top 1 accuracy: {top_1_acc:.4f}')
    print(f'{split} top {top_k} accuracy: {top_k_acc:.4f}')


if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--data_dir', type=str, default='../ssl_data_96',
                        help='location of data')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--model_path', type=str, default='./VAT/weights_new_VAT_64_sample.pth',
                        help='location of model weights')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1008, metavar='S',
                        help='random seed')

    # Parse arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Load pre-trained model
    model = torch.nn.DataParallel(Model(), device_ids=[0,1,2,3]).to(args.device)
    # model = Model().to(args.device) # DO NOT modify this line - if your Model() takes arguments, they should have default values
    print('n parameters: %d' % sum([m.numel() for m in model.parameters()]))

    # Load data
    data_loader_val = load_data(args.data_dir, args.batch_size, split='val')
    # data_loader_test = load_data(args.data_dir, args.batch_size, split='test')

    # Evaluate model
    with torch.no_grad():
        start = time.time()
        evaluate(model, data_loader_val, args.device, 'Validation')
        end = time.time()
        print(end-start, "evaluation time")
        # evaluate(model, data_loader_test, args.device, 'Test')
