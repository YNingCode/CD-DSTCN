import matlab.engine  # import matlab引擎

# 启动一个新的MATLAB进程，并返回Python的一个变量，它是一个MatlabEngine对象，用于与MATLAB过程进行通信。
import numpy
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC


n = loadmat('../../STGCN_brain/dataset/ADNI/ADNI.mat')
label = n['label'][0]
for i in range(label.shape[0]):
    if label[i] == 2:
        label[i] = 1
eng = matlab.engine.start_matlab()  # 可以调用matlab的内置函数。
d = eng.loaddata()


for l in [2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),1,2,4]:
    for s in  [2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),1,2,4]:
        l = matlab.double([l])
        s = matlab.double([s])
        fdata = np.array(eng.SLR(d, l,s))
        fdata = fdata.transpose((2,0,1))
        fdata = fdata.reshape(306,-1)
        index = [i for i in range(fdata.shape[0])]
        np.random.shuffle(index)
        fdata = fdata[index]
        label = label[index]
        fold = 10
        KF = KFold(n_splits=fold, shuffle=False)
        acc_tr_list = list()
        acc_val_list = list()
        acc_te_list = list()
        pre_list = []
        recall_list = []
        auc_list = list()
        f1_list = list()
        run = 1
        for train_idx, test_idx in KF.split(fdata):
            x_tr, x_te = fdata[train_idx], fdata[test_idx]
            y_tr, y_te = label[train_idx], label[test_idx]
            clf = SVC(kernel='linear', C=1, degree=10)
            clf.fit(x_tr, y_tr.ravel())
            acc_tr = clf.score(x_tr, y_tr)  # trian
            acc_te = clf.score(x_te, y_te)  # test
            acc_tr_list.append(acc_tr)
            acc_te_list.append(acc_te)
            auc = roc_auc_score(y_te, clf.decision_function(x_te))
            auc_list.append(auc)
            y_true = y_te.squeeze()
            y_pre = clf.predict(x_te)
            f1 = f1_score(y_true, y_pre)
            f1_list.append(f1)
            pre = precision_score(y_true, y_pre)
            pre_list.append(pre)
            recall = recall_score(y_true, y_pre)
            recall_list.append(recall)

            print("l",l,"s",s,'run', run, 'acc_tr', acc_tr, 'acc_te', acc_te, 'pre', pre, 'recall', recall, 'f1', f1, 'auc', auc)
            run += 1
        print("l",l,"s",s,'our_acc_tr', sum(acc_tr_list) / fold)
        print("l",l,"s",s,'our_acc_te', sum(acc_te_list) / fold)
        print("l",l,"s",s,'our_pre_te', sum(pre_list) / fold)
        print("l",l,"s",s,'our_recall_te', sum(recall_list) / fold)
        print("l",l,"s",s,'our_f1_te', sum(f1_list) / fold)
        print("l",l,"s",s,'our_auc_te', sum(auc_list) / fold)
