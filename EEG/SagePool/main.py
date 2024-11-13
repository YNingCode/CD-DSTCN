from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix, \
    roc_curve
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import SubsetRandomSampler, DataLoader
from EEG.SagePool.model import *
from EEG.readdata import *
import argparse
import warnings
from fvcore.nn import FlopCountAnalysis, parameter_count_table
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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
    model.eval()
    for data_test, label_test in datasets_test:
        batch_num_test = len(data_test)
        data_test = data_test.to(DEVICE)
        label_test = label_test.long().to(DEVICE)

        output = model(data_test)
        # print(output)
        losss = nn.CrossEntropyLoss()(output, label_test)
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

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all

def save_roc_data(file_path, all_fpr, all_tpr):
    with open(file_path, 'w') as f:
        for fpr, tpr in zip(all_fpr, all_tpr):
            for fp, tp in zip(fpr, tpr):
                f.write(f"{fp}\t{tp}\n")

# main settings
parser = argparse.ArgumentParser(description='SagePool')
parser.add_argument('--batch_size', type=int, default=20, help='batch size')
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.003, type=float)
args = parser.parse_args()

all_results = {}

for subj_idx in range(1, 26):
    dataset = getdataset(subj_idx=subj_idx)

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
    KF = KFold(n_splits=10, shuffle=True)
    labels = [dataset[i][1] for i in range(dataset_length)]
    for train_and_val_indices, test_indices in KF.split(dataset):
        print("*******{}-flod*********".format(i+1))

        # 将训练和验证集的索引进一步划分为独立的训练集和验证集
        train_size = int(train_ratio * len(train_and_val_indices))
        valid_size = len(train_and_val_indices) - train_size
        train_indices, val_indices = train_and_val_indices[:train_size], train_and_val_indices[train_size:]

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
        patiences = 50
        min_acc = 0

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            train_acc = 0
            for data, label in datasets_train:
                batch_num_train = len(data)
                data = data.to(DEVICE)
                label = label.long().to(DEVICE)
                # print("train label:", label)
                output = model(data)
                batch_loss_train = closs(output, label)
                optimizer.zero_grad()
                batch_loss_train.backward()
                optimizer.step()
                # torch.cuda.empty_cache()
                acc_num = accuracy(output, label)
                train_acc += acc_num/batch_num_train
                train_loss += batch_loss_train

            losses.append(train_loss / len(datasets_train))
            acces.append(train_acc / len(datasets_train))

            eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all = stest(
                model, datasets_test)

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
                'subj_idx:{},i:{},epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},precision : {'
                ':.6f},recall : {:.6f},f1 : {:.6f},my_auc : {:.6f} '
                .format(subj_idx, i, epoch, train_loss / len(datasets_train), train_acc / len(datasets_train),
                        eval_loss / len(datasets_test), eval_acc_epoch, precision, recall, f1, my_auc))

        model_test = Model()
        model_test = model_test.to(DEVICE)
        model_test.load_state_dict(torch.load('./Save/latest' + str(i) + '.pth'))
        eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all = stest(
            model_test, datasets_test)

        print(f"subj_idx: {subj_idx}, fold {i + 1} - Eval Acc: {eval_acc_epoch:.6f}")
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

    # 保存每个受试者的结果到all_results字典
    all_results[subj_idx] = {
        "Accuracy": avg_acc,
        "Test F1": avg_f1,
        "Test AUC": avg_auc,
    }

# 输出所有受试者的结果
print("Description:", parser.description)
for subj_idx, results in all_results.items():
    print(f"****** subj_idx: {subj_idx} Results ******")
    print(f"Accuracy: {results['Accuracy']}")
    print(f"F1: {results['Test F1']}")
    print(f"AUC: {results['Test AUC']}")
    print("******************************************")


