import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tsai.all import *
import csv

def replace_nan_with_zero(my_list):
    my_list = np.array(my_list)
    my_list[np.isnan(my_list)] = 0
    my_list = my_list.tolist()
    return my_list

clf = SVC()
target_list = get_UCR_univariate_list()
cls_name = []
dset_name = []
total_acc = []
root = './dataset'
for t in target_list:
    
    X_train, y_train, X_test, y_test  = get_UCR_data(t, return_split=True)
    # X_train = np.array(replace_nan_with_zero(X_train))
    # X_test = np.array(replace_nan_with_zero(X_test))
    # X_train = X_train.reshape(y_train.shape[0], -1)
    # X_test = X_test.reshape(y_test.shape[0], -1)
    
    X_train = []
    X_test = []
    
    data = pd.read_csv(root + '/' + t + '/ddp_bert_train.csv')
    data1 = pd.read_csv(root + '/' + t + '/sdp_bert_train.csv')
    data2 = pd.read_csv(root + '/' + t + '/tfp_bert_train.csv')
    for i in range(0, len(data)):
        X_train.append(data.iloc[i].tolist() + data1.iloc[i].tolist() + data2.iloc[i].tolist())
    
    if "ddp_bert_test1.csv" in os.listdir(root + '/' + t):
        data = pd.read_csv(root + '/' + t + '/ddp_bert_test1.csv')
        tempdata1 = pd.read_csv(root + '/' + t + '/ddp_bert_test2.csv')
        tempdata2 = pd.read_csv(root + '/' + t + '/ddp_bert_test3.csv')
        tempdata3 = pd.read_csv(root + '/' + t + '/ddp_bert_test4.csv')
        data.append(tempdata1)
        data.append(tempdata1)
        data.append(tempdata1)
    else:
        data = pd.read_csv(root + '/' + t + '/ddp_bert_test.csv')
    data1 = pd.read_csv(root + '/' + t + '/sdp_bert_test.csv')
    data2 = pd.read_csv(root + '/' + t + '/tfp_bert_test.csv')
    for i in range(0, len(data)):
        X_test.append(data.iloc[i].tolist() + data1.iloc[i].tolist() + data2.iloc[i].tolist())

    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    cls_name.append("SVM-fusion")
    dset_name.append(t)
    total_acc.append(accuracy_score(y_test, predicted))
    print(t, "\t", accuracy_score(y_test, predicted))
data = list(zip(cls_name, dset_name, total_acc))

# 指定要保存的文件名
csv_file_path = './ts_result/longformer/svm_fusion.csv'

# 写入CSV文件
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # 写入表头
    csv_writer.writerow(['classifier_name', 'dataset_name', 'accuracy'])
    
    # 写入数据
    csv_writer.writerows(data)