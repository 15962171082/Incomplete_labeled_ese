import numpy as np
from sklearn.metrics import roc_auc_score


def classification_auc(gold_labels:np, pred_sources:np):
    '''
    parameter:
        gold_labels: 金标标签 (num_simple, num_class)
        pred_sources: 预测概率(num_simple, num_class)
    return: 
        AUC
    '''
    roc_auc = dict()
    # micro: label和sources 展开压缩为一列, 以instance为单位忽视label 都当作二分类计算
    roc_auc['micro'] = roc_auc_score(gold_labels, pred_sources, average='micro')
    # macro: 以label为单位, 计算每个label的auc再取均值
    roc_auc['macro'] = roc_auc_score(gold_labels, pred_sources, average='macro')
    # macro_weighted: 以label为单位, 计算每个label的auc再按金标标签中每个类别的比例加权取均值
    roc_auc['macro_weighted'] = roc_auc_score(gold_labels, pred_sources, average='weighted')
    
    return roc_auc
   

def classification_p_at_k_one_instance(gold_label:list, pred_source:list, k:int, threshold:float):
    '''
    parameter:
        gold_label:  金标01串 (num_class)
        pred_source: 预测概率 (num_class)
        k: top_k
        threshold: 阈值
    return: 
        p_at_k: top k 的精确率
    '''
    if k == 0:
        return 0
    num_class = len(gold_label)
    if k > len(gold_label):
        raise ValueError(f'k值大于类别数量 k:{k}, num_class:{num_class}')
    pred_gold = []
    for gold, pred in zip(gold_label, pred_source):
        pred_gold.append((gold, pred))
    # 排序
    sorted_pred_gold = list(sorted(pred_gold, key=lambda x:x[1], reverse=True))
    tp = 0
    for i in range(k):
        if sorted_pred_gold[i][0] == 1 and sorted_pred_gold[i][1] >= threshold:
            tp += 1
    return tp/k


def classification_p_at_k(gold_labels:list, pred_sources:list, k:int, threshold:float):
    '''
    parameter:
        gold_labels: 金标标签 (num_simple, num_class)
        pred_sources: 预测概率(num_simple, num_class)
        k: top_k 
        threshold: 阈值
    return: 
        p_at_k: top k 的精确率
    '''
    all_p_at_k = 0
    for gold_label, pred_source in zip(gold_labels, pred_sources):
        all_p_at_k += classification_p_at_k_one_instance(gold_label, pred_source, k, threshold)
    return all_p_at_k/len(gold_labels)


def classification_average_p_at_k(gold_label:list, pred_source:list):
    '''
    parameter:
        gold_label:  金标01串 (num_class)
        pred_source: 预测概率 (num_class)
    return: 
        ap: average_p
    '''
    pred_gold = []
    for gold, pred in zip(gold_label, pred_source):
        pred_gold.append((gold, pred))
    # 排序
    sorted_pred_gold = list(sorted(pred_gold, key=lambda x:x[1], reverse=True))
    
    all_ap = 0
    gold_count = 0
    for idx, item in enumerate(sorted_pred_gold):
        if item[0] == 1:
            gold_count += 1
            all_ap += (gold_count/(idx + 1))
    return all_ap/gold_count


def classification_mean_average_p_at_k(gold_labels:list, pred_sources:list):
    '''
    parameter:
        gold_labels: 金标标签 (num_simple, num_class)
        pred_sources: 预测概率(num_simple, num_class)
    return: 
        map: mean_average_p
    '''
    all_ap = 0
    for gold, pred in zip(gold_labels, pred_sources):
        all_ap += classification_average_p_at_k(gold, pred)
    return all_ap/(len(gold_labels))