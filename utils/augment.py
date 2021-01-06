from torchvision.transforms import transforms


class TwoCropAugment(object):

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x = self.transform(sample)
        y = self.transform(sample)
        return x, y


def simsiam_transform(resize=32):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose([
        transforms.RandomResizedCrop(size=resize, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def linear_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
