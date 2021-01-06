from tqdm import tqdm
import torch
import torch.nn.functional as F


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def knn_monitor(net, memory_data_loader, test_data_loader, epoch, k=200, t=0.1, hide_progress=False, device='cpu'):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0., 0, []
    with torch.no_grad():
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
            data = data.to(device)
            feature = net(data)
            feature = feature.view(feature.size(0), -1)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = feature.view(feature.size(0), -1)
            feature = F.normalize(feature, dim=1)
            pre_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)
            total_num += data.size(0)
            total_top1 += (pre_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Epoch': epoch, 'Accuracy': total_top1 / total_num})
        return total_top1 / total_num
