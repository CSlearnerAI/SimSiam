import os
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.tool import get_conf, get_optimizer
from models.base import Linear
from utils.augment import linear_transform
from utils.lr_scheduler import LRScheduler

config = get_conf()
linear_conf = config['linear']
seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = linear_conf['batch_size']
num_epoch = linear_conf['num_epoch']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter(linear_conf['tensorboard_dir'])


def train_eval(loader, model, criterion, optimizer, scheduler, epoch, is_train=True):
    model.train() if is_train else model.eval()

    iterion = tqdm(loader)
    finish, total_loss, total_num, total_acc = 0, 0., 0, 0.
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in iterion:
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            finish += 1
            total_loss += loss.item()
            total_num += target.shape[0]
            pred = torch.argsort(output, dim=-1, descending=True)
            total_acc += torch.sum((pred[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            if is_train:
                loss.backward()
                optimizer.step()
                scheduler.step()

            iterion.set_description('> {}_Epoch {}_Iter {}: loss {:.4f}, acc {:.4f}'.format(
                'Train' if is_train else 'Test', epoch, finish, total_loss / finish, total_acc / total_num
            ))
        return total_loss / finish, total_acc / total_num


def main():
    data_dir = config['data_dir']
    transform = linear_transform()
    train_sets = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_sets = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_sets, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_sets, batch_size=batch_size, shuffle=False)

    model = Linear(out_dim=10).to(device)

    model.resnet.load_state_dict(torch.load(linear_conf['model_file']))
    for layer in model.resnet.parameters():
        layer.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        model, linear_conf['lr'], momentum=linear_conf['momentum'], wd=linear_conf['wd']
    )
    scheduler = LRScheduler(
        optimizer=optimizer,
        warmup_epochs=linear_conf['warmup_epoch'],
        warmup_lr=linear_conf['warmup_lr'],
        num_epochs=num_epoch,
        base_lr=linear_conf['lr'],
        final_lr=linear_conf['final_lr'],
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=True
    )

    best_loss_model_path = os.path.join(linear_conf['model_dir'], 'best_loss.pkl')
    best_acc_model_path = os.path.join(linear_conf['model_dir'], 'best_acc.pkl')
    last_model_path = os.path.join(linear_conf['model_dir'], 'last.pkl')

    best_loss = np.inf
    best_acc = -1
    for epoch in range(1, num_epoch + 1):
        print('#' * 20, 'Epoch {}'.format(epoch), '#' * 20)
        train_loss, train_acc = train_eval(train_loader, model, criterion, optimizer, scheduler, epoch)
        print('| Train_Epoch {}: loss {:.4f}, acc {:.4f}'.format(
            epoch, train_loss, train_acc
        ))
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('train_acc', train_acc, global_step=epoch)

        test_loss, test_acc = train_eval(test_loader, model, criterion, optimizer, scheduler, epoch, is_train=False)
        print('| Test_Epoch {}: loss {:.4f}, acc {:.4f}'.format(
            epoch, test_loss, test_acc
        ))
        writer.add_scalar('test_loss', test_loss, global_step=epoch)
        writer.add_scalar('test_acc', test_acc, global_step=epoch)

        torch.save(model.state_dict(), last_model_path)
        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_loss_model_path)
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_acc_model_path)


if __name__ == '__main__':
    main()
