import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from imblearn.over_sampling import SMOTE
import torch.optim as optim
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datetime import date
import warnings
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from utils.storage import build_experiment_folder
from meta_neural_network_architectures import MLPReLUNormNetwork
from utils.parser_utils import get_args
from baselines.early_stopping import EarlyStopping
from sktime.datasets import load_from_tsfile, load_from_tsfile_to_dataframe
warnings.filterwarnings("ignore")
if __name__== "__main__" :
    
    np.random.seed(3407)
    args, device = get_args()
    # device='cuda:1'
    save_path = 'result/MeTARocketc-{}-{}-{}/'.format(args.openworld, args.dataset_name[0], date.today())
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    filepath = os.path.join("result/MeTARocketc-all-04281143/saved_models", "{}_{}".format("train_model", '97'))
    state = torch.load(filepath)
    state_dict_loaded = state['network']
    state_dict = dict()
    for key, value in state_dict_loaded.items():
        if 'classifier' in key:
            state_dict[key.replace("classifier.module.",'')] = value

    model = MLPReLUNormNetwork(im_shape=(2,2,40), num_output_classes=args.
                                             num_classes_per_set,
                                             args=args, device=device, meta_classifier=True).to(device=device)
    model.load_state_dict(state_dict=state_dict)

    if not args.closed_world:
        train_data, test_data = None,None
        for root, _, files in os.walk(os.path.join(args.dataset_path, args.dataset_name[0])):
            for file in files:
                tsv_path = os.path.join(root, file)
                if 'train' in tsv_path:
                    train_data, train_label = load_from_tsfile(tsv_path)
                    train_label = np.where(train_label=='0', 0, 1)
                    train_label = torch.tensor(train_label).to(device)
                if 'test' in tsv_path:
                    test_data, test_label = load_from_tsfile(tsv_path)
                    test_label = np.where(test_label=='0', 0, 1)
                    test_label = torch.tensor(test_label).to(device)
        if train_data is None and test_data is None:
            raise("There is not tsv file train/test dataset")
    else:
        tsv_path = os.path.join(args.dataset_path, args.dataset_name[0], '{}.ts'.format(args.dataset_name[0]))
        dataset, label = load_from_tsfile(tsv_path)
        train_index = pd.DataFrame({'label':label.tolist()}).groupby("label").sample(frac=0.7, replace=False).index
        train_data = dataset.loc[train_index]
        train_label = label[train_data.index]
        test_data = dataset[~dataset.index.isin(train_data.index)]
        test_label = label[test_data.index]
        encoder = LabelEncoder()
        train_label = torch.tensor(encoder.fit_transform(train_label))
        test_label = torch.tensor(encoder.transform(test_label))
    train_data = train_data.values
    test_data = test_data.values
    train_tmp = list()
    test_tmp = list()
    for i in range(len(train_data)):
        train_tmp.append([train_data[i][0].tolist(), train_data[i][1].tolist()])
    for i in range(len(test_data)):
        test_tmp.append([test_data[i][0].tolist(), test_data[i][1].tolist()])
    train_data = torch.tensor(train_tmp)
    test_data = torch.tensor(test_tmp)
    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)
    
    model.reset_clf(np.unique(test_label.cpu()).shape[0])

    for name, param in model.named_parameters():
        if 'mlp' in name and param.requires_grad==True:
            param.requires_grad_(False)
    optimizer = optim.Adam(model.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)

    early_stopping = EarlyStopping(save_path, patience=30)
    start_time = time.time()
    print("Freezing!")
    for epoch in range(args.total_epochs*6//10):
        epoch = int(epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break 
        for index, batch in enumerate(train_dataloader):
            if (index+1)%10==0:
                tmp_time = time.time()
                model.eval()
                X, y = next(iter(test_dataloader))
                y_true = y.to(device)
                y_pred = model(X.to(device),epoch)
                loss = F.cross_entropy(input=y_pred, target=y_true)
                y_pred = y_pred.argmax(dim=1)
                AC = accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
                PR = precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
                RC = recall_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
                F1 = f1_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
                cur_time = time.time()
                print("Val epoch: {} | Batch:{} | Time:{:.4f} | Loss: {:.4f} | AC: {:.4f} | PR: {:.4f} | RC: {:.4f} | F1: {:.4f}".format(epoch+1,(index+1)//10, cur_time-tmp_time, loss.item(),AC,PR,RC,F1))
                early_stopping(loss, accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()), model)
                tmp_time = cur_time
                if early_stopping.early_stop:
                    print("Early stopping")
                    break #跳出迭代，结束训练
            tmp_time = time.time()
            model.train()
            X, y = batch[0], batch[1]
            y_true = y.to(device)
            y_pred = model(X.to(device),epoch)
            loss = F.cross_entropy(input=y_pred, target=y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            y_pred = y_pred.argmax(dim=1)
            AC = accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            PR = precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
            RC = recall_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
            F1 = f1_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
            cur_time = time.time()
            print("Train epoch: {} | Batch:{} | Time:{:.4f} | Loss: {:.4f} | AC: {:.4f} | PR: {:.4f} | RC: {:.4f} | F1: {:.4f}".format(epoch+1,(index+1), cur_time-tmp_time,loss.item(),AC,PR,RC,F1))
            tmp_time = cur_time
    print("unFreezing!")
    for name, param in model.named_parameters():
        if 'mlp' in name and param.requires_grad==True:
            param.requires_grad_(True)
    optimizer = optim.Adam(model.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)

    #                                                         eta_min=args.min_learning_rate)
    for epoch in range(args.total_epochs*6//10, args.total_epochs):
        epoch = int(epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break #跳出迭代，结束训练
        for index, batch in enumerate(train_dataloader):
            if (index+1)%10==0:
                tmp_time = time.time()
                model.eval()
                X, y = next(iter(test_dataloader))
                y_true = y.to(device)
                y_pred = model(X.to(device),epoch)
                loss = F.cross_entropy(input=y_pred, target=y_true)
                y_pred = y_pred.argmax(dim=1)
                AC = accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
                PR = precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
                RC = recall_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
                F1 = f1_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
                cur_time = time.time()
                print("Val epoch: {} | Batch:{} | Time:{:.4f} | Loss: {:.4f} | AC: {:.4f} | PR: {:.4f} | RC: {:.4f} | F1: {:.4f}".format(epoch+1,(index+1)//10, cur_time-tmp_time,loss.item(),AC,PR,RC,F1))
                early_stopping(loss, accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()), model)
                tmp_time = cur_time
                if early_stopping.early_stop:
                    print("Early stopping")
                    break #跳出迭代，结束训练
            tmp_time = time.time()
            model.train()
            X, y = batch[0], batch[1]
            y_true = y.to(device)
            y_pred = model(X.to(device),epoch)
            loss = F.cross_entropy(input=y_pred, target=y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            y_pred = y_pred.argmax(dim=1)
            AC = accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            PR = precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
            RC = recall_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
            F1 = f1_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
            cur_time = time.time()
            print("Train epoch: {} | Batch:{} | Time:{:.4f} | Loss: {:.4f} | AC: {:.4f} | PR: {:.4f} | RC: {:.4f} | F1: {:.4f}".format(epoch+1,(index+1), cur_time-tmp_time,loss.item(),AC,PR,RC,F1))
            tmp_time = cur_time
    print('Total Time: {}'.format(tmp_time-start_time))
    print('Extraction Feature spend {}s'.format(model.extraction_time))
    print('Execution spend {}s'.format(model.execution_time))
    

