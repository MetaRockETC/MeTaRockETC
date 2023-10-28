import pandas as pd
import os 
import numpy as np
from sktime.datasets import write_ndarray_to_tsfile
import tldextract as te
from tqdm import tqdm
import random
import time
from sklearn.preprocessing import StandardScaler

ISCXVPN = {
    'Chat': ['chat', 'aim', 'icq'],
    'Email': ['imaps', 'pop3s', 'smpts', 'email'],
    'File transfer': ['ftps','sftp', 'file'],
    'Streaming': ['vimeo', 'youtube'],
    'Torrent': ['bittorrent', 'utorrent', 'torrent'],
    'VoIP': ['audio', 'video']
}
ISCXTor = {
    'Chat': ['chat', 'aim', 'icq'],
    'Email': ['imaps', 'pop3s', 'smpts', 'email'],
    'File transfer': ['p2p', 'ftp', 'sftp'],
    'VoIP': ['audio', 'video'],
    'Streaming': ['vimeo', 'youtube', 'spotify'],
    'Browsing': ['ssl', 'gate', 'tor']
}
def iscxlabel(x):
    file_name = x.loc[x.index.str.contains('file')].values[0]
    if 'vpn' in file_name:
        dict = ISCXVPN
        if 'NonVPN' in file_name or 'nonvpn' in file_name:
            tag = ''
        else:
            tag = 'vpn'
    else:
        dict = ISCXTor
        if 'NonTor' in file_name or 'nontor' in file_name:
            tag = ''
        else:
            tag = 'tor'
    for key, value in dict.items():
        for cl in value:
            if cl in file_name:
                return tag+key
    return ''

def tldlabel(x):
    if x.loc[x.index.str.contains('SNI')].notnull().values[0]:
        return te.extract(x.loc[x.index.str.contains('SNI')].values[0]).domain
    else:
        return ''

def parse_csv2tsv(path_list, num_packets, num_instance, out_dir):
    start_time = time.time()
    dataset = pd.DataFrame()
    scaler = StandardScaler()
    # if False:
    for index, path in enumerate(path_list):
        if not os.path.isdir(path):
            temp = pd.read_csv(path, low_memory=False)
            if 'Bottlenet-HTTPS2016' in path or 'CTU-HTTPS2017' in path:
                problem_name = 'Bottlenet-HTTPS2016' if 'Bottlenet-HTTPS2016' in path else 'CTU-HTTPS2017'
                
                temp = temp[(temp["numOfPackets"]>=num_packets) & 
                (temp["protocol"]==1) & (temp["label"]!='') & (temp["label"].notnull()) & 
                (temp['label'].isin((temp["label"].value_counts()<num_instance)[(temp["label"].value_counts()<num_instance).values==True].index.tolist()))]
            elif 'ISCX' in path:
                problem_name = 'VPN'
                temp['label'] = temp.apply(iscxlabel, axis=1)
                if 'tor' in path or 'Tor' in path:
                    problem_name = 'Tor'
                    temp = temp[(temp["numOfPackets"]>=num_packets) & (temp["protocol"]==1) & (temp["label"]!='') & (temp["label"].notnull())]
                    packet_seq = temp.loc[:,temp.columns.str.contains('packet_')].reset_index(drop=True)
                    time_seq = temp.loc[:,temp.columns.str.contains('time_')].reset_index(drop=True)
                    rest = temp.loc[:, (~temp.columns.str.contains('packet_') & ~temp.columns.str.contains('time_'))].reset_index(drop=True)
                    new_temp = pd.DataFrame()
                    for i in range(temp.shape[0]):
                        for j in range(packet_seq.shape[1]//num_packets):
                            if j*num_packets>rest.loc[i]["numOfPackets"]:
                                break
                            item = pd.concat([rest.iloc[i],packet_seq.iloc[i,j*num_packets:(j+1)*num_packets], time_seq.iloc[i,j*num_packets:(j+1)*num_packets]], axis=0, ignore_index=True)
                            item.index = rest.columns.to_list()+packet_seq.iloc[:,:num_packets].columns.to_list()+time_seq.iloc[:,:num_packets].columns.to_list()
                            new_temp = new_temp.append(item, ignore_index=True)
                    temp = new_temp
                else:
                    temp = temp[(temp["numOfPackets"]>=num_packets) & (temp["protocol"]==1) & (temp["label"]!='') & (temp["label"].notnull()) & (temp['label'].isin((temp["label"].value_counts()>=num_instance)[(temp["label"].value_counts()>=num_instance).values==True].index.tolist()))]
            elif '5GAD' in path:
                problem_name = '5GAD'
                temp['label'] = 1 if 'Attacks' in path else 0
                temp = temp[(temp["protocol"]==1) & (temp["label"]!='') & (temp["label"].notnull())].reset_index(drop=True)
                packet_seq = temp.loc[:,temp.columns.str.contains('packet_')].iloc[:,:num_packets]
                time_seq = temp.loc[:,temp.columns.str.contains('time_')].iloc[:,:num_packets]
                rest = temp.loc[:, (~temp.columns.str.contains('packet_') & ~temp.columns.str.contains('time_'))]
                temp = pd.concat([rest,packet_seq, time_seq], axis=1, ignore_index=True)
                temp.columns = rest.columns.tolist()+packet_seq.columns.tolist()+time_seq.columns.tolist()
                
            dataset = pd.concat([dataset, temp], axis=0)
        else:
            columns = ['label']
            columns.extend(['packet_{}'.format(i) for i in range(num_packets)])
            columns.extend(['time_{}'.format(i) for i in range(num_packets)])
            if 'GoogleHome' in path:
                problem_name = 'GoogleHome'
                for root, _, files in os.walk(path):
                    for file in tqdm(files):
                        sample_path = os.path.join(root, file)
                        label = file[:file.find('_??')-1]
                        temp = pd.read_csv(sample_path, low_memory=False)
                        new_temp = [label]
                        packet_seq = (temp['size'].values*temp['direction'].values).tolist()
                        time_seq = temp['time'].values.tolist()
                        new_temp.extend(packet_seq[:num_packets] if num_packets<=len(packet_seq) else packet_seq+[0.0]*(num_packets-len(packet_seq)))
                        new_temp.extend(time_seq[:num_packets] if num_packets<=len(time_seq) else time_seq+[0.0]*(num_packets-len(time_seq)))
                        tmp_dict = dict()
                        for k, v in zip(columns, new_temp):
                            tmp_dict[k] = v
                        dataset = dataset.append(tmp_dict, ignore_index=True)
                        
                dataset.iloc[:, 1:num_packets+1]=scaler.fit_transform(dataset.iloc[:, 1:num_packets+1].values.tolist())
                dataset.iloc[:, num_packets+1:]=scaler.fit_transform(dataset.iloc[:, num_packets+1:].values.tolist())

    
    # dataset.loc[:, dataset.columns.str.contains("packet_|time_|label")].to_csv(os.path.join(out_dir,problem_name+'.csv'), index=False)
       
    
    class_counter = dataset['label'].value_counts()

    # dataset_all
    class_value_list = dataset.loc[:, 'label'].values.reshape(-1)
    class_label = class_counter.index.tolist()
    dataset_all = dataset.loc[:,dataset.columns.str.contains("packet_|time_")].values
    series_length = dataset_all.shape[1]//2
    dataset_all = dataset_all.reshape(dataset_all.shape[0], -1, series_length)
    write_ndarray_to_tsfile(dataset_all, out_dir, problem_name=problem_name, class_label=class_label, missing_values='0', equal_length=True, series_length=series_length, class_value_list=class_value_list)
    print('Spend {}s in Extracting Feature'.format(time.time()-start_time))
    
path_list = ['/data1/5GAD/Attacks.csv', '/data1/5GAD/Normal.csv']
parse_csv2tsv(path_list=path_list, num_packets=40, num_instance=45, out_dir='/home/hyp/code/MeTAL/datasets')

