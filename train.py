import random
import torch.nn as nn
from data_process import create_CPI_dataset2
from utils import *
from Model1 import CoAffDTI
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, auc
import os
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
CPIdatasets = ['data']
cuda_name = 'cuda:0'
ratio = 5
print('cuda_name:', cuda_name)
print('dataset:', CPIdatasets)

'''set random seed'''
SEED = 340788888 # 3407888
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

model_type = CoAffDTI
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 0.0001

NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(os.path.join(results_dir, CPIdatasets[0])):
    os.makedirs(os.path.join(results_dir, CPIdatasets[0]))

result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = model_type()
model.to(device)
model_st = model_type.__name__
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for dataset in CPIdatasets:

    train_data, dev_data, test_data = create_CPI_dataset2()

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE,
                                               shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=TEST_BATCH_SIZE,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE,
                                              shuffle=False)
    model_file_name = 'models/model_' + model_st + '_' + dataset + '_' + str(ratio) + '.model'
    best_auc = 0
    for epoch in range(NUM_EPOCHS):

         train(model, device, train_loader, optimizer, epoch + 1, NUM_EPOCHS, TRAIN_BATCH_SIZE)
         print('predicting for test data')
         T_D, P_D,_ = predicting(model, device, dev_loader,epoch=epoch)
         T, P,_ = predicting(model, device, test_loader,epoch=epoch)

         val_auc_score = roc_auc_score(T, P)
         val_acc = accuracy_score(T, np.where(P >= 0.5, 1, 0))
         val_precision = precision_score(T, np.where(P >= 0.5, 1, 0))
         val_recall = recall_score(T, np.where(P >= 0.5, 1, 0))
         val_f1 = f1_score(T, np.where(P >= 0.5, 1, 0))
         dev_auc_score = roc_auc_score(T_D, P_D)
         dev_acc = accuracy_score(T_D, np.where(P_D >= 0.5, 1, 0))
         dev_precision = precision_score(T_D, np.where(P_D >= 0.5, 1, 0))
         dev_recall = recall_score(T_D, np.where(P_D >= 0.5, 1, 0))
         dev_f1 = f1_score(T_D, np.where(P_D >= 0.5, 1, 0))
         print('test result:', val_auc_score, val_acc, val_precision, val_recall, val_f1, '\n'
			   'dev result:', dev_auc_score, dev_acc, dev_precision, dev_recall, dev_f1)

         if val_auc_score>best_auc :
             best_auc = val_auc_score
             torch.save(model.state_dict(), model_file_name)
    # test
    print('all training done. Testing...')
    model_p = model_type()
    model_p.to(device)
    model_p.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    test_T, test_P,pred = predicting(model_p, device, test_loader,epoch=100)

    # --- 计算 AUPR ---
    precision_list, recall_list, _ = precision_recall_curve(test_T, test_P)
    test_aupr = auc(recall_list, precision_list)
    # ----------------

    test_auc = roc_auc_score(test_T, test_P)
    test_acc = accuracy_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_recall = recall_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_precision = precision_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_f1_score = f1_score(test_T, np.where(test_P >= 0.5, 1, 0))

    # 将 AUPR 加入结果字符串
    result_str = 'test result:'  + \
                 'test_auc:' + str(test_auc) +   \
                 'test_aupr:' + str(test_aupr) + \
                 'test_acc:' + str(test_acc)  + \
                 'test_precision:' + str(test_precision)  + \
                 'test_recall:' + str(test_recall)  + \
                 'test_f1_core:' + str(test_f1_score) + '\n'

    print(result_str)

    save_file = os.path.join(results_dir, dataset, 'test_restult_' + str(ratio) + '_' + model_st + '.txt')
    open(save_file, 'w').writelines(result_str)

