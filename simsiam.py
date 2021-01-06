import os
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from utils.tool import get_conf, get_optimizer
from utils.augment import TwoCropAugment, simsiam_transform, linear_transform
from models.base import SimSiam
from utils.lr_scheduler import LRScheduler
from utils.knn import knn_monitor

config = get_conf()
simsiam_conf = config['simsiam']

seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = simsiam_conf['batch_size']
num_epoch = simsiam_conf['num_epoch']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter(simsiam_conf['tensorboard_dir'])


def train(loader, model, optimizer, scheduler, epoch):
    model.train()
    finish, total_loss = 0, 0.
    iterion = tqdm(loader)
    for (xi, xj), _ in iterion:
        xi, xj = xi.to(device), xj.to(device)

        if epoch == 1:
            grid = make_grid(xi[:32])
            writer.add_image('views_1', grid, global_step=epoch)

            grid = make_grid(xj[:32])
            writer.add_image('views_2', grid, global_step=epoch)

        model.zero_grad()
        loss = model(xi, xj)

        total_loss += loss.item()
        finish += 1

        loss.backward()
        optimizer.step()
        lr = scheduler.step()
        iterion.set_description('> Train_Epoch {}_Iter {}: loss {:.4f}'.format(
            epoch, finish, total_loss / finish
        ))
    return total_loss / finish, lr


def main():
    train_transform = simsiam_transform(resize=simsiam_conf['resize'])
    test_transform = linear_transform()

    data_dir = config['data_dir']
    train_sets = datasets.CIFAR10(data_dir, train=True, download=True, transform=TwoCropAugment(train_transform))
    memory_sets = datasets.CIFAR10(data_dir, train=True, download=True, transform=test_transform)
    test_sets = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_sets, batch_size=batch_size, shuffle=True)
    memory_loader = DataLoader(memory_sets, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_sets, batch_size=batch_size, shuffle=False)

    model = SimSiam().to(device=device)

    optimizer = get_optimizer(
        model, simsiam_conf['lr'], momentum=simsiam_conf['momentum'], wd=simsiam_conf['wd']
    )
    scheduler = LRScheduler(
        optimizer=optimizer,
        warmup_epochs=simsiam_conf['warmup_epoch'],
        warmup_lr=simsiam_conf['warmup_lr'],
        num_epochs=num_epoch,
        base_lr=simsiam_conf['lr'],
        final_lr=simsiam_conf['final_lr'],
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=True
    )
    last_model_path = os.path.join(simsiam_conf['model_dir'], 'last.pkl')
    best_loss_model_path = os.path.join(simsiam_conf['model_dir'], 'best_loss.pkl')
    best_acc_model_path = os.path.join(simsiam_conf['model_dir'], 'best_acc.pkl')

    best_loss = np.inf
    best_acc = -1
    for epoch in range(1, num_epoch + 1):
        print('#' * 20, 'Epoch {}'.format(epoch), '#' * 20)
        train_loss, lr = train(train_loader, model, optimizer, scheduler, epoch)
        print('| Train_Epoch {}: loss {}, lr {}'.format(
            epoch, train_loss, lr
        ))
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('lr', lr, global_step=epoch)

        acc = knn_monitor(model.encoder.resnet, memory_loader, test_loader, epoch,
                          k=200, t=0.1, hide_progress=False, device=device)
        print('| Eval from kNN_Epoch {}: acc {:.4f}'.format(
            epoch, acc
        ))
        writer.add_scalar('knn_acc', acc, global_step=epoch)

        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.encoder.resnet.state_dict(), best_loss_model_path)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.encoder.resnet.state_dict(), best_acc_model_path)

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'encoder': model.encoder.resnet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
        }, last_model_path)


if __name__ == '__main__':
    main()
