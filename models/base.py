import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock


class Predictor(nn.Module):

    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super(Predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class EncoderSiam(nn.Module):

    def __init__(self, hidden_dim=2048, out_dim=2048):
        super(EncoderSiam, self).__init__()
        self.resnet, self.in_dim = self._get_resnet()
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    @staticmethod
    def _get_resnet():
        resnet = models.resnet18()
        out_dim = resnet.fc.in_features
        net = []
        for name, module in resnet.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d) or isinstance(module, nn.Linear):
                continue
            net.append(module)
        net = nn.Sequential(*net)
        return net, out_dim

    @staticmethod
    def _get_resnet2(**kwargs):
        backbone = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        return backbone, backbone.output_dim

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class Linear(nn.Module):

    def __init__(self, out_dim=10):
        super(Linear, self).__init__()
        encoder = EncoderSiam()
        self.resnet, self.in_dim = encoder.resnet, encoder.in_dim
        self.fc = nn.Linear(in_features=self.in_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimSiam(nn.Module):
    
    def __init__(self):
        super(SimSiam, self).__init__()
        self.encoder = EncoderSiam()
        self.predictor = Predictor()

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return loss


def D(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return - (p * z).sum(dim=1).mean()


if __name__ == '__main__':
    model = models.resnet18()
    model = SimSiam()
    x1 = torch.randn((2, 3, 32, 32))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print('Finish')
