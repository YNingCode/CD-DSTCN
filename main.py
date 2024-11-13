from random import random
import pandas as pd
from matplotlib import pyplot as plt
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from scipy.stats import pearsonr
matplotlib.use('Agg')
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix, \
    roc_curve
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import SubsetRandomSampler
from Model.model_3 import *
from ReadData import *
import argparse
import os
import random
import warnings
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy(output, labels):
    _, pred = torch.max(output, dim=1)
    # print(output)
    correct = pred.eq(labels)
    # print(pred)
    # print(labels)
    acc_num = correct.sum()

    return acc_num

def stest(model, datasets_test):
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    gailv_all = []
    pro_all = []
    out_logits_all = []  # 保存每个batch的out_logits
    confidence_all = []
    all_final = []
    model.eval()
    for data_test, label_test in datasets_test:
        batch_num_test = len(data_test)
        data_test = data_test.reshape(batch_num_test, -1, num_nodes, args.patch_size)
        data_test = data_test.to(DEVICE)
        if args.dataset == 'SEED':
            label_test = label_test.squeeze(1).long().to(DEVICE)
        else:
            label_test = label_test.long().to(DEVICE)

        output, c_loss, out_logits, adj, confidence, final = model(data_test, label_test, "Test")

        out_logits_all.append(out_logits.cpu().detach().numpy())  # 保存out_logits
        confidence_all.append(confidence.cpu().detach().numpy())
        all_final.append(final.cpu().detach().numpy())
        # print(output)
        losss = 0.8*nn.CrossEntropyLoss()(output, label_test)+0.2*c_loss
        eval_loss += float(losss)
        _, pred = torch.max(output, dim=1)
        num_correct = (pred == label_test).sum()
        acc = int(num_correct) / batch_num_test
        eval_acc += acc
        pre = pred.cpu().detach().numpy()
        pre_all.extend(pre)
        label_true = label_test.cpu().detach().numpy()
        labels_all.extend(label_true)
        pro_all.extend(output[:, 1].cpu().detach().numpy())

    tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    eval_acc_epoch = accuracy_score(labels_all, pre_all)
    precision = precision_score(labels_all, pre_all)
    recall = recall_score(labels_all, pre_all)
    f1 = f1_score(labels_all, pre_all)
    my_auc = roc_auc_score(labels_all, pro_all)

    # # 检查并移除包含 NaN 的元素
    # labels_all = np.array(labels_all)
    # pro_all = np.array(pro_all)
    # mask = ~np.isnan(labels_all) & ~np.isnan(pro_all)
    # labels_all_clean = labels_all[mask]
    # pro_all_clean = pro_all[mask]
    # if labels_all_clean.size == 0 or pro_all_clean.size == 0:
    #     print("Error: After removing NaNs, there are no samples left.")
    #     my_auc = 0
    # else:
    #     my_auc = roc_auc_score(labels_all_clean, pro_all_clean)

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, \
           out_logits_all, adj, confidence_all, all_final

def save_roc_data(file_path, all_fpr, all_tpr):
    with open(file_path, 'w') as f:
        for fpr, tpr in zip(all_fpr, all_tpr):
            for fp, tp in zip(fpr, tpr):
                f.write(f"{fp}\t{tp}\n")

seed_value = 8 # 设定随机数种子
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(seed_value)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）
torch.backends.cudnn.deterministic = True


# main settings
parser = argparse.ArgumentParser(description='ASTGCN')
parser.add_argument('--data_root', type=str, default='./dataset', help='数据存放目录')
parser.add_argument('--dataset', type=str, default='ADNI', help='使用数据集名称')

parser.add_argument('--patch_size', type=int, default=65, help='每一个patch的时间长度')
parser.add_argument('--batch_size', type=int, default=20, help='batch size')
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.003, type=float)
args = parser.parse_args()

# dataset, num_nodes, seq_length, num_classes = dianxiandataloader(args)
dataset, num_nodes, seq_length, num_classes = ADNI(args)
# dataset, num_nodes, seq_length, num_classes = PD(args)
print(num_nodes, seq_length, num_classes)
args.num_nodes = num_nodes
args.num_classes = num_classes
args.seq_length = seq_length

dataset_length = len(dataset)
train_ratio = 0.9
valid_ratio = 0.1
kk = 10
test_acc = []
test_pre = []
test_recall = []
test_f1 = []
test_auc = []
label_ten = []
test_sens = []
test_spec = []
pro_ten = []
i = 0
skf = StratifiedKFold(n_splits=10, shuffle=True)
KF = KFold(n_splits=10, shuffle=True)
# 假设我们有一个 labels 列表，其中包含每个样本的类别标签
labels = [dataset[i][1] for i in range(dataset_length)]
combined_out_logits = []  # 保存所有折的out_logits
combined_labels_all = []  # 保存所有折的标签
# for flod, (train_and_val_indices, test_indices) in enumerate(skf.split(range(dataset_length), labels)):
for train_and_val_indices, test_indices in KF.split(dataset):
    print("*******{}-flod*********".format(i+1))

    # 将训练和验证集的索引进一步划分为独立的训练集和验证集
    train_size = int(train_ratio * len(train_and_val_indices))
    valid_size = len(train_and_val_indices) - train_size
    train_indices, val_indices = train_and_val_indices[:train_size], train_and_val_indices[train_size:]

    train_num = len(train_indices)
    val_num = len(val_indices)
    test_num = len(test_indices)
    print(train_num, val_num, test_num)

    # 提取训练、验证和测试的数据集
    datasets_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(train_indices))
    datasets_valid = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(val_indices))
    datasets_test = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(test_indices))

    model = Model()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    closs = nn.CrossEntropyLoss()
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    patiences = 500
    min_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for data, label in datasets_train:
            batch_num_train = len(data)
            data = data.reshape(batch_num_train, -1, num_nodes, args.patch_size)
            data = data.to(DEVICE)

            if args.dataset == 'SEED':
                label = label.squeeze(1).long().to(DEVICE)
            else:
                label = label.long().to(DEVICE)
            # print("train label:", label)

            # # 计算FLOPs
            # flops = FlopCountAnalysis(model, data)
            # print("FLOPs: ", flops.total())
            # # 计算参数数量
            # print("parameter_count: ", parameter_count_table(model))

            output, c_loss, out_logits, adj, confidence, _ = model(data, label, "Train")
            batch_loss_train = 0.8*closs(output, label)+0.3*c_loss
            optimizer.zero_grad()
            batch_loss_train.backward()
            optimizer.step()
            # torch.cuda.empty_cache()
            acc_num = accuracy(output, label)
            train_acc += acc_num/batch_num_train
            train_loss += batch_loss_train

        losses.append(train_loss / len(datasets_train))
        acces.append(train_acc / len(datasets_train))

        eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, out_logits_all,\
            adj, confidence, all_final= stest(model, datasets_test)

        if eval_acc_epoch > min_acc:
            min_acc = eval_acc_epoch
            torch.save(model.state_dict(), './Save/latest' + str(i) + '.pth')
            print("Model saved at epoch{}, Best Acc: {}".format(epoch, eval_acc_epoch))
            patience = 0
        else:
            patience += 1

        if patience > patiences:
            break

        eval_losses.append(eval_loss / len(datasets_test))
        eval_acces.append(eval_acc / len(datasets_test))

        print(
            'i:{},epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},precision : {'
            ':.6f},recall : {:.6f},f1 : {:.6f},my_auc : {:.6f} '
            .format(i, epoch, train_loss / len(datasets_train), train_acc / len(datasets_train),
                    eval_loss / len(datasets_test), eval_acc_epoch, precision, recall, f1, my_auc))

    model_test = Model()
    model_test = model_test.to(DEVICE)
    model_test.load_state_dict(torch.load('./Save/latest' + str(i) + '.pth'))
    eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, out_logits_all,\
        adj, confidence, all_final= stest(model_test, datasets_test)

    # 合并每一折的out_logits
    combined_out_logits.extend(out_logits_all)
    combined_labels_all.extend(labels_all)

    print("---", eval_acc_epoch)
    test_acc.append(eval_acc_epoch)
    test_pre.append(precision)
    test_recall.append(recall)
    test_f1.append(f1)
    test_auc.append(my_auc)
    test_sens.append(sensitivity)
    test_spec.append(specificity)
    label_ten.extend(labels_all)
    pro_ten.extend(pro_all)
    i = i+1

print("test_acc", test_acc)
print("test_pre", test_pre)
print("test_recall", test_recall)
print("test_f1", test_f1)
print("test_auc", test_auc)
print("test_sens", test_sens)
print("test_spec", test_spec)
avg_acc = sum(test_acc) / kk
avg_pre = sum(test_pre) / kk
avg_recall = sum(test_recall) / kk
avg_f1 = sum(test_f1) / kk
avg_auc = sum(test_auc) / kk
avg_sens = sum(test_sens) / kk
avg_spec = sum(test_spec) / kk
print("*****************************************************")
print('acc', avg_acc)
print('pre', avg_pre)
print('recall', avg_recall)
print('f1', avg_f1)
print('auc', avg_auc)
print("sensitivity", avg_sens)
print("specificity", avg_spec)

acc_std = np.sqrt(np.var(test_acc))
pre_std = np.sqrt(np.var(test_pre))
recall_std = np.sqrt(np.var(test_recall))
f1_std = np.sqrt(np.var(test_f1))
auc_std = np.sqrt(np.var(test_auc))
sens_std = np.sqrt(np.var(test_sens))
spec_std = np.sqrt(np.var(test_spec))
print("*****************************************************")
print("acc_std", acc_std)
print("pre_std", pre_std)
print("recall_std", recall_std)
print("f1_std", f1_std)
print("auc_std", auc_std)
print("sens_std", sens_std)
print("spec_std", spec_std)
print("*****************************************************")

print(label_ten)
print(pro_ten)

for n in range(10):
    full_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    best_model = Model()
    best_model = best_model.to(DEVICE)
    best_model.load_state_dict(torch.load('./Save/latest'+str(n)+'.pth'))  # Load the best model saved during cross-validation

    # Run the model on the full dataset
    eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all , out_logits_all,\
        adj, confidence, all_final = stest(best_model, full_dataloader)
    # Print the final evaluation metrics
    print("N:", n)
    print("Final Eval Acc:", eval_acc_epoch)
    print("Final Precision:", precision)
    print("Final Recall:", recall)
    print("Final F1:", f1)
    print("Final AUC:", my_auc)
    print("Final Sensitivity:", sensitivity)
    print("Final Specificity:", specificity)
    print("---------------------")

    # # pagerank前10个节点
    # all_final = np.concatenate(all_final,axis=0)
    # node_mean = np.mean(all_final, axis=0)
    # # 计算皮尔逊相关系数矩阵
    # adjacency_matrix = np.zeros((90, 90))
    # for i in range(90):
    #     for j in range(90):
    #         if i != j:
    #             # 计算皮尔逊相关系数
    #             corr, _ = pearsonr(node_mean[i], node_mean[j])
    #             # 处理负相关和零相关系数
    #             if corr < 0:
    #                 corr = 0  # 或者使用其他处理方法
    #             adjacency_matrix[i, j] = corr
    # # 构建图结构
    # G = nx.from_numpy_array(adjacency_matrix)
    # # 计算 PageRank
    # pagerank_scores = nx.pagerank(G)
    # # 将 PageRank 评分按降序排列，并选择评分最高的前 10 个节点
    # top_10_nodes = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[:10]
    # top_10_scores = [pagerank_scores[node] for node in top_10_nodes]
    # # 输出前 10 个节点的编号及其重要性分数
    # output_lines = []
    # for node, score in zip(top_10_nodes, top_10_scores):
    #     line = f"Node: {node}, Score: {score}"
    #     output_lines.append(line)
    # # 将结果保存到txt文件中
    # with open('pagerank_'+str(n)+'.txt', 'w') as file:
    #     for line in output_lines:
    #         file.write(line + "\n")

    # # confidence箱型图
    # confidence = np.concatenate(confidence, axis=0)
    # labels = np.array(labels_all)
    # con_0 = confidence[labels == 1]
    # # 将数据转换为(样本数, 时间片数)
    # con_0 = con_0.squeeze(axis=2)  # 去掉第三个维度
    # con_0 = np.transpose(con_0, (1, 0))  # 转置为(时间片数, 样本数)
    # # 将数据保存到CSV文件中
    # df = pd.DataFrame(con_0, columns=["Sample_" + str(i) for i in range(con_0.shape[1])])
    # df.to_csv('boxplot_data_12_' + str(n) + '.csv', index=False)
    # # 绘制箱型图
    # plt.boxplot(con_0.T, labels=["Time 1", "Time 2", "Time 3"], showfliers=False)
    # plt.xlabel("Time slices")
    # plt.ylabel("Score")
    # plt.title("Boxplot of Scores for Label 0")
    # plt.savefig('Box_12_' + str(n) + '.png')
    # plt.clf()  # 清除当前的绘图区域
    # plt.show()

    # 保存矩阵
    # file_name = './Adj/matrix_file_01_'+str(n)+'.txt'
    # # 转换 Tensor 形状并保存到 txt 文件
    # for idx, tensor in enumerate(adj):
    #     with open(file_name, 'w') as f:
    #         for matrix in tensor:
    #             matrix = matrix.cpu().numpy()
    #             matrix[matrix < 0] = 0
    #             np.savetxt(f, matrix, fmt='%.6f')
    #             f.write('\n')  # 在每个矩阵之间添加一个空行

    # 散点图
    # 使用PCA进行降维到2维
    # pca = PCA(n_components=2)
    # combined_logits = np.concatenate(out_logits_all, axis=0)  # shape: (140, 256)
    # combined_labels = np.array(labels_all)
    # pca_result = pca.fit_transform(combined_logits)
    # # 绘制降维后的散点图，并根据标签不同使用不同颜色表示
    # plt.figure(figsize=(8, 6))
    # plt.scatter(pca_result[combined_labels == 0, 0], pca_result[combined_labels == 0, 1],
    #             color='b', alpha=0.8, label='Class 0')  # 类别0用蓝色表示
    # plt.scatter(pca_result[combined_labels == 1, 0], pca_result[combined_labels == 1, 1],
    #             color='r', alpha=0.8, label='Class 1')  # 类别1用红色表示
    # plt.title('PCA of Combined out_logits_all with Labels')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.grid(True)
    # plt.savefig('PCA_'+str(n)+'.png')
    # plt.show()
    # 使用t-SNE降维
    # tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=0)
    tsne = TSNE(n_components=2, random_state=183)
    combined_logits = np.concatenate(out_logits_all, axis=0)  # shape: (140, 256)
    combined_labels = np.array(labels_all)
    tsne_result = tsne.fit_transform(combined_logits)
    # 绘制降维后的散点图，并根据标签不同使用不同颜色表示
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[combined_labels == 0, 0], tsne_result[combined_labels == 0, 1],
                color='b', alpha=0.8, label='Class 0')  # 类别0用蓝色表示
    plt.scatter(tsne_result[combined_labels == 1, 0], tsne_result[combined_labels == 1, 1],
                color='r', alpha=0.8, label='Class 1')  # 类别1用红色表示
    plt.title('t-SNE of Combined out_logits_all with Labels')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(False)
    plt.legend()
    plt.savefig('tSNE_1_2' + str(n) + '.png')
    plt.show()

    # ROC曲线
    # # Save the final ROC data
    # fpr, tpr, _ = roc_curve(labels_all, pro_all)
    # save_roc_data('./roc/1_2/final_roc_data'+str(n)+'.txt', [fpr], [tpr])
    #