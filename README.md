## SimSiam for CIFAR10

#### Paper

Exploring Simple Siamese Representation Learning [1]

#### Attention

You need to create the folder "results"/"data" and its subfolders (referred to "config.yaml"). This project doesn't provide any code to create them.

#### Setting

The settings of related parameters can be seen in “config.yaml”

- dataset: CIFAR10 (in folder "data")

- encoder: ResNet-18(adjust for CIFAR10 [2])
- optimizer: SGD
- scheduler: warm-up(10 epoch) + cosine decay schedule

#### Tools

- Python 3.8.3

- Pytorch 1.7.1

#### Results of experiments

###### Epoch 100 for training

- knn monitor

> the acc of valid datasets during pre-training: 0.71

- linear evaluation

| Epoch | lr   | wd   | train_acc | test_acc |
| ----- | ---- | ---- | --------- | -------- |
| 100   | 0.3  | 0    | 0.765     | 0.746    |
| 100   | 0.03 | 0    | 0.771     | 0.748    |

###### Epoch 300 for training

- knn monitor

> the acc of valid datasets during pre-training: 0.88

- linear evaluation

| Epoch | lr   | wd   | train_acc | test_acc |
| ----- | ---- | ---- | --------- | -------- |
| 100   | 3    | 0    | 0.901     | 0.881    |
| 100   | 30   | 0    | 0.914     | 0.886    |

#### Reference

[1] [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566v1)

[2] [MoCo for CIFAR10](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb)