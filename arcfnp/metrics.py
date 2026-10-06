import torch

def HammingLoss(pred_labels, target_labels):
    return torch.mean((pred_labels != target_labels).float()).item()

def OneError(pred_scores, target_labels):
    _, index = torch.max(pred_scores, dim=1)
    oneerror = 0.0
    num_data = pred_scores.size(0)
    for i in range(num_data):
        if target_labels[i, index[i]] != 1:
            oneerror += 1
    return oneerror / num_data

def Coverage(pred_scores, target_labels):
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    coverage = 0.0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        if has_label[i, :].sum() > 0:
            r = torch.max(order[i, has_label[i, :]]).item() + 1
            coverage += r
    coverage = coverage / num_data - 1.0
    return coverage / num_classes

def RankingLoss(pred_scores, target_labels):
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    rankingloss = 0.0
    count = 0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        m = torch.sum(has_label[i, :]).item()
        n = num_classes - m
        if m != 0 and n != 0:
            rankingloss = rankingloss + (torch.sum(order[i, has_label[i, :]]).item() - m * (m - 1) / 2.0) / (m * n)
            count += 1
    return rankingloss / count

def AveragePrecision(pred_scores, target_labels):
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    ap = 0.0
    count = 0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        m = torch.sum(has_label[i, :]).item()
        if m != 0:
            sorts, _ = torch.sort(order[i, has_label[i, :]])
            temp = 0.0
            for j in range(sorts.size(0)):
                temp += (j + 1.0) / (sorts[j].item() + 1)
            ap += temp / m
            count += 1
    return ap / count
