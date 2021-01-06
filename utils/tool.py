import yaml
import os
import torch.optim as optim

py_path = os.path.dirname(os.path.realpath(__file__))


def get_conf():
    conf_file = os.path.join(py_path, '../config.yaml')
    config = yaml.load(open(conf_file, 'r', encoding='UTF-8'), Loader=yaml.FullLoader)
    return config


def get_optimizer(model, lr, momentum, wd):
    predictor_prefix = 'predictor'
    parameters = [{
        'name': 'encoder',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    }, {
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
    optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd)
    # optimizer = optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optimizer
