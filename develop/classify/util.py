import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from colorama import Fore

def cosin_metric(x1, x2):
    u = torch.einsum('id, jd -> ij', x1, x2)
    d1 = torch.linalg.norm(x1, dim=1)
    d2 = torch.linalg.norm(x2, dim=1)
    d = torch.einsum('i, j -> ij', d1, d2)
    return u / d

def cosin_metric_uda(x1, x2):
    # x1 = F.softmax(x1,dim=1)
    # x2 = F.softmax(x2,dim=1)
    sim = F.cosine_similarity(x1, x2, dim=1)
    sim = torch.mean(sim)
    return -sim

def distance(a,b):
    return ((a-b)**2).mean(axis=(0,1))

def auto_test(label, pred, thr=-1):

    precision, recall, threshold = precision_recall_curve(label, pred)
    precision=precision[:-2]
    recall = recall[:-2]
    f1 = 2.0 * precision * recall / (precision + recall)
    if thr < 0.0:
        f1 = f1[~np.isnan(f1)] #
        k = np.where(f1 == np.max(f1))[0][0]
    else:
        threshold_ = np.abs(threshold - thr)
        k = np.where(threshold_ == np.min(threshold_))[0][0]

    print(f"{Fore.GREEN}best F1: precision {precision[k]:.4f} / recall {recall[k]:.4f}{Fore.RESET}")

    pred = np.array([0 if t < threshold[k] else 1 for t in pred])
    cm = confusion_matrix(label, pred, normalize=None)
    print(f"{Fore.YELLOW}{cm}{Fore.RESET}")

    return precision[k], recall[k]
