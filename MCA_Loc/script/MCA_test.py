"""
    Author：Jacy.Y
    Purpose：Pass
    Created: 2025/3/24
    File name: MCALoc_test.py
"""

# Dependencies
import pandas as pd
from MCA_model import *
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, jaccard_score, hamming_loss, precision_recall_curve, accuracy_score
from torch.utils.data import DataLoader
import re
import os
# Hyper Parameters

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

node_input_dim = MCALoc_config['node_input_dim']
hidden_dim = MCALoc_config['hidden_dim']
layer = MCALoc_config['layer']
dropout = MCALoc_config['dropout']
batch_size = 16
gpu = "0"  # 使用GPU 1
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
threshold = 0.5

############ path ############
script_path = 'path_to/script/'
test_path = '../Data/test.csv'    # 测试数据路径
outpath = '/media/sdc1/24jxyou/DSALoc/HPA_v24/'  # 处理数据存储路径
ProtTrans_path = "path_to/prot_t5_xl_uniref50/"
###############################################

def collate_fn(batch, max_length=850):

    node_feats, labels = zip(*batch)
    batch_size = len(batch)

    def pad_tensor(tensor, fixed_length):

        if tensor.shape[0] < fixed_length:
            pad_size = fixed_length - tensor.shape[0]
            padding_dims = [pad_size] + list(tensor.shape[1:])
            padding = torch.zeros(padding_dims, dtype=tensor.dtype)
            padded_tensor = torch.cat([tensor, padding], dim=0)
        else:
            padded_tensor = tensor[:fixed_length]
        return padded_tensor

    padded_node_feats = torch.stack([pad_tensor(tensor, max_length) for tensor in node_feats])
    mask = torch.zeros(batch_size, max_length, dtype=torch.int)
    labels_stacked = np.stack(labels)
    labels = torch.tensor(labels_stacked, dtype=torch.float32)
    for i, tensor in enumerate(node_feats):
        actual_length = min(tensor.shape[0], max_length)
        mask[i, :actual_length] = True

    return padded_node_feats, labels, mask


# remove_nonstandard
def clean_sequence(seq):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)


val_data = pd.read_csv(test_path)             # 直接读到的原始测试集数据

# Model Evaluation

# 测试集标签序列和ID分离
val_ID_list = val_data["ID"].tolist()
val_seq = val_data["Sequence"].tolist()
label_columns = val_data.columns[-12:]
val_labels = {ID: torch.tensor(label_values, dtype=torch.float32) for ID, label_values in zip(val_data["ID"], val_data[label_columns].values)}
print(len(val_ID_list))
val_dataset = MCALoc_Dataset(val_ID_list, val_labels, outpath=outpath)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, prefetch_factor=2, collate_fn=collate_fn)


fold_models = []
for fold in range(5):
    state_dict = torch.load(f'../model/LA_CrossCL_fold{fold+1}_10_256_16_1e-4_0.5_2_1e-5.pt', device)
    model = MCALoc(node_input_dim, hidden_dim, layer, dropout)
    model.eval()
    model = model.to(device)
    model.load_state_dict(state_dict, strict=False)
    fold_models.append(model)

# 用于存储每个fold的预测结果
fold_all_results = []  # 每个元素是 (num_samples, num_classes) 的概率
fold_all_preds = []  # 每个元素是 (num_samples, num_classes) 的二值预测
fold_all_labels = []
with torch.no_grad():
    # 先收集所有数据（或分批处理，此处采用逐模型预测）
    for model in fold_models:
        fold_results = []
        fold_preds = []
        fold_labels = []
        for h_V, label, mask in val_loader:
            h_V = h_V.to(device)
            mask = mask.to(device)
            labels = label.to(device)
            output = model(h_V, mask)
            prob = torch.sigmoid(output)  # 转换为概率
            pred = torch.ge(prob, threshold).int()
            no_pred_mask = (pred.sum(dim=1) == 0)
            # 3. 如果存在全 0 行，执行 Top-1 补偿
            if no_pred_mask.any():
                max_indices = prob[no_pred_mask].argmax(dim=1)
                pred[no_pred_mask] = pred[no_pred_mask].scatter_(1, max_indices.unsqueeze(1), 1)
            fold_labels.append(labels.cpu())
            fold_results.append(prob.cpu())
            fold_preds.append(pred.cpu())
        # 合并当前fold所有batch的结果
        fold_results = torch.cat(fold_results, dim=0).numpy()
        fold_preds = torch.cat(fold_preds, dim=0).numpy()
        fold_labels = torch.cat(fold_labels, dim=0).numpy()
        fold_all_results.append(fold_results)
        fold_all_preds.append(fold_preds)
        fold_all_labels.append(fold_labels)

# 真实标签只需获取一次（所有fold相同）
all_labels_list = []
with torch.no_grad():
    for h_V, labels, mask in val_loader:
        all_labels_list.append(labels.cpu())
all_labels = torch.cat(all_labels_list, dim=0).numpy()

# 为每个fold计算指标
metrics = {
    'Macro_AUC': [], 'Macro_AUPR': [], 'Macro_F1': [],
    'Micro_AUC': [], 'Micro_AUPR': [], 'Micro_F1': [],
    'mAP': [], 'Acc': [],  'Jaccard': [], 'H_Loss': []
}


for i in range(5):
    res = fold_all_results[i]
    pred = fold_all_preds[i]
    all_labels = fold_all_labels[i]
    macro_auc = roc_auc_score(all_labels, res, average='macro')
    micro_auc = roc_auc_score(all_labels, res, average='micro')
    mAP = average_precision_score(all_labels, res, average='macro')
    Acc = accuracy_score(all_labels, pred)
    macro_f1 = f1_score(all_labels, pred, average='macro')
    micro_f1 = f1_score(all_labels, pred, average='micro')
    jacc = jaccard_score(all_labels, pred, average='samples')
    hloss = hamming_loss(all_labels, pred)
    metrics['Macro_AUC'].append(macro_auc)
    metrics['Micro_AUC'].append(micro_auc)
    metrics['Macro_F1'].append(macro_f1)
    metrics['Micro_F1'].append(micro_f1)
    metrics['mAP'].append(mAP)
    metrics['Acc'].append(Acc)
    metrics['Jaccard'].append(jacc)
    metrics['H_Loss'].append(hloss)


for i in range(5):
    res = fold_all_results[i]
    pred = fold_all_preds[i]
    all_labels = fold_all_labels[i]
    macro_AUPR = 0
    for i in range(all_labels.shape[1]):
        # 计算当前标签的 Precision 和 Recall
        precision, recall, _ = precision_recall_curve(all_labels[:, i], res[:, i])
        sorted_indices = np.argsort(recall)
        recall = recall[sorted_indices]
        precision = precision[sorted_indices]
        macro_AUPR += np.trapz(precision, recall)  # np.trapz 使用梯形规则进行数值积分
    macro_AUPR /= all_labels.shape[1]

    all_labels = all_labels.flatten()
    all_results = res.flatten()
    precision, recall, _ = precision_recall_curve(all_labels, all_results)
    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]
    # 使用梯形规则计算Micro-AUPR
    micro_AUPR = np.trapz(precision, recall)

    metrics['Macro_AUPR'].append(macro_AUPR)
    metrics['Micro_AUPR'].append(micro_AUPR)

# 输出每个指标的均值和标准差
for name, values in metrics.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{name}: {mean_val:.4f} ± {std_val:.4f}")



