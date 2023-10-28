import concurrent.futures
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.parser_utils import get_args



class FewShotLearningDatasetParallel(Dataset):
    def __init__(self, args):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.data_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.data_loaded_in_memory = False
        self.args = args
        self.closed_world = args.closed_world
        self.train_val_test_split = args.train_val_test_split
        self.current_set_name = "train"
        self.num_target_samples = args.num_target_samples
        self.reset_stored_filepaths = args.reset_stored_filepaths
        train_rng = np.random.RandomState(seed=args.train_seed)
        val_rng = np.random.RandomState(seed=args.val_seed)
        test_rng = np.random.RandomState(seed=args.val_seed)
        train_seed = train_rng.randint(1, 999999)
        val_seed = val_rng.randint(1, 999999)
        test_seed = test_rng.randint(1, 999999)
        args.val_seed = val_seed
        args.train_seed = train_seed
        args.test_seed = test_seed
        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0
        self.num_samples_per_class = args.num_samples_per_class
        self.num_classes_per_set = args.num_classes_per_set
        self.rng = np.random.RandomState(seed=self.seed['train'])

        self.datasets, self.labels = self.load_dataset()
        self.label_set, self.dataset_size_dict = self.get_label_set()
        self.data_length = {name: self.datasets[name].shape[0] for name in self.datasets.keys()}

        print("data", self.data_length)
        self.observed_seed_set = None

    def load_dataset(self):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)
        """
        data_paths = self.load_datapaths()
        labels = ''
        dataset = pd.DataFrame()
        for data_path in data_paths:
            x_tmp, y_tmp = load_from_tsfile_to_dataframe(data_path)
            dataset = pd.concat([dataset, x_tmp], axis=0)
            if labels=='':
                labels=y_tmp
            else:
                labels = np.concatenate((labels, y_tmp), axis=0)
        dataset = dataset.reset_index(drop=True)
        encoder = LabelEncoder()
        # total_data_num = dataset.shape[0]
        labels_unique = np.unique(labels).tolist()
        num_dataset = dataset.shape[0]
        num_labels = len(labels_unique)
        per_type_class_data_index = {k: np.where(labels==k)[0] for k in labels_unique}
        train_rng = np.random.RandomState(self.seed["train"])
        val_rng = np.random.RandomState(self.seed["val"])
        test_rng = np.random.RandomState(self.seed["test"]*2)
        x_train = pd.DataFrame()
        x_val = pd.DataFrame()
        x_test = pd.DataFrame()
        if self.closed_world:
            train_val_test_split = []
            for p in self.train_val_test_split:
                # v = self.args.num_sample_per_class
                # if p*num_dataset//num_labels > v:
                v = int(p*num_dataset//num_labels)
                train_val_test_split.append(v)
            for label in labels_unique:
                x_train = pd.concat([x_train, dataset.iloc[train_rng.choice(per_type_class_data_index[label], size=train_val_test_split[0], replace=True)]], axis=0) 
                x_val = pd.concat([x_val, dataset.iloc[val_rng.choice(per_type_class_data_index[label], size=train_val_test_split[1], replace=True)]], axis=0) 
                x_test = pd.concat([x_test, dataset.iloc[test_rng.choice(per_type_class_data_index[label], size=train_val_test_split[2], replace=True)]], axis=0) 
        else:
            selected_classes = train_rng.choice(labels_unique,
                                      size=int(self.train_val_test_split[0]*num_labels), replace=False)
            train_rng.shuffle(selected_classes)
            for label in selected_classes:
                x_train = pd.concat([x_train, dataset.iloc[train_rng.choice(per_type_class_data_index[label], size=num_dataset//num_labels, replace=True)]], axis=0) 
            # x_train = dataset.sample(frac=self.train_val_test_split[0], random_state=self.seed['train'])
            x_val = dataset[~dataset.index.isin(x_train.index)].sample(frac=(self.train_val_test_split[1])/sum(self.train_val_test_split[1:]), random_state=self.seed['val'])
            x_test = dataset[~dataset.index.isin(x_train.index) & ~dataset.index.isin(x_val.index)]
            # labels = encoder.fit_transform(labels)
        y_train = labels[x_train.index]
        y_train = encoder.fit_transform(y_train)
        y_val = labels[x_val.index]
        y_val = encoder.transform(y_val)
        y_test = labels[x_test.index]
        y_test = encoder.transform(y_test)

        x_train = self.process_ts_data(x_train.reset_index(drop=True), normalise=False)
        x_val = self.process_ts_data(x_val.reset_index(drop=True), normalise=False)
        x_test = self.process_ts_data(x_test.reset_index(drop=True), normalise=False)
        dataset_splits = {"train": x_train, "val":x_val , "test": x_test}
        label_splits = {"train": y_train, "val": y_val, "test":y_test}

        return dataset_splits, label_splits
    def load_datapaths(self):
        data_path = []
        for dataset in self.dataset_name:
            tmp = os.path.join(self.data_path, dataset, '{}.ts'.format(dataset))
            if not os.path.exists(tmp):
                self.reset_stored_filepaths = True
            else:
                data_path.append(tmp)
        return data_path

    def process_ts_data(self, X,
                    vary_len: str = "suffix-noise",
                    normalise: bool = False):
        """
        This is a function to process the data, i.e. convert dataframe to numpy array
        :param X:
        :param normalise:
        :return:
        """
        num_instances, num_dim = X.shape
        columns = X.columns
        max_len = np.max([len(X[columns[0]][i]) for i in range(num_instances)])
        output = np.zeros((num_instances, num_dim, max_len), dtype=np.float64)

        for i in range(num_dim):
            for j in range(num_instances):
                output[j, i, :] = X[columns[i]][j].values
            output[:, i, :] = self.fill_missing(
                output[:, i, :],
                max_len,
                vary_len,
                normalise
            )

        return output

    def fill_missing(self, x: np.array,
                 max_len: int,
                 vary_len: str = "suffix-noise",
                 normalise: bool = True):
        if vary_len == "zero":
            if normalise:
                x = StandardScaler().fit_transform(x)
            x = np.nan_to_num(x)
        elif vary_len == 'prefix-suffix-noise':
            for i in range(len(x)):
                series = list()
                for a in x[i, :]:
                    if np.isnan(a):
                        break
                    series.append(a)
                series = np.array(series)
                seq_len = len(series)
                diff_len = int(0.5 * (max_len - seq_len))

                for j in range(diff_len):
                    x[i, j] = random.random() / 1000

                for j in range(diff_len, seq_len):
                    x[i, j] = series[j - seq_len]

                for j in range(seq_len, max_len):
                    x[i, j] = random.random() / 1000

                if normalise:
                    tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                    x[i] = tmp[:, 0]
        elif vary_len == 'uniform-scaling':
            for i in range(len(x)):
                series = list()
                for a in x[i, :]:
                    if np.isnan(a):
                        break
                    series.append(a)
                series = np.array(series)
                seq_len = len(series)

                for j in range(max_len):
                    scaling_factor = int(j * seq_len / max_len)
                    x[i, j] = series[scaling_factor]
                if normalise:
                    tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                    x[i] = tmp[:, 0]
        else:
            for i in range(len(x)):
                for j in range(len(x[i])):
                    if np.isnan(x[i, j]):
                        x[i, j] = random.random() / 1000

                if normalise:
                    tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                    x[i] = tmp[:, 0]

        return x

    def get_label_set(self):
        labels = ""
        per_type_class_data_index = {}
        for key, value in self.labels.items():
            if labels=="":
                labels = value
            else:
                labels = np.concatenate((labels, value), axis=0)
            per_type_class_data_index[key] = {k: np.where(value==k)[0] for k in np.unique(value).tolist()}
        return set(labels.tolist()), per_type_class_data_index

    def load_batch(self, dataset_types, batch_index):
        return self.datasets[dataset_types][batch_index]
        
    def shuffle(self, x, rng):
        indices = np.arange(len(x))
        rng.shuffle(indices)
        x = x[indices]
        return x

    def __len__(self):
        return self.data_length[self.current_set_name]

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name
        if set_name == "train":
            self.update_seed(dataset_name=set_name, seed=self.init_seed[set_name] + current_iter)

    def update_seed(self, dataset_name, seed=100):
        self.seed[dataset_name] = seed

    def __getitem__(self, idx, ):
        dataset_type = self.current_set_name
        seed = self.seed[dataset_type]+idx
        rng = np.random.RandomState(seed)
        if self.args.few_shot:
            selected_classes = rng.choice(list(self.dataset_size_dict[dataset_type].keys()),
                                        size=self.num_classes_per_set, replace=False)
            rng.shuffle(selected_classes)
            episode_labels = [i for i in range(self.num_classes_per_set)]
            class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                    zip(selected_classes, episode_labels)}

            x_data = []
            y_data = []
            for class_entry in selected_classes:
                choose_samples_list = rng.choice(self.dataset_size_dict[dataset_type][class_entry],
                                                size=self.num_samples_per_class + self.num_target_samples, replace=True)
                class_samples = []
                class_labels = []
                for sample in choose_samples_list:
                    x_class_data = self.load_batch(dataset_type, sample)
                    # k = k_dict[class_entry]
                    class_samples.append(x_class_data)
                    class_labels.append(int(class_to_episode_label[class_entry]))
                x_data.append(class_samples)
                y_data.append(class_labels)
        else:
            x_data = []
            y_data = []
            choose_samples_list = rng.choice(self.datasets[dataset_type],
                                                size=self.data_length[dataset_type], replace=True)

        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        support_set = x_data[:, :self.num_samples_per_class]
        support_set_label = y_data[:,:self.num_samples_per_class]
        target_set = x_data[:, self.num_samples_per_class:]
        target_set_label = y_data[:,self.num_samples_per_class:]
        return support_set, target_set, support_set_label, target_set_label, idx
    def reset_seed(self):
        self.seed = self.init_seed

class MetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.samples_per_iter = args.samples_per_iter
        self.num_workers = args.num_dataprovider_workers
        self.total_train_iters_produced = 0
        self.dataset = FewShotLearningDatasetParallel(args=args)
        self.batches_per_iter = args.samples_per_iter
        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)
        self.args = args

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter),
                          shuffle=True, num_workers=self.num_workers, drop_last=True)

    def continue_from_iter(self, current_iter):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_iter:
        """
        self.total_train_iters_produced += (current_iter * (self.num_of_gpus * self.batch_size * self.samples_per_iter))

    def get_train_batches(self, total_batches=-1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
        # self.dataset.set_augmentation(augment_images=augment_images)
        self.total_train_iters_produced += (self.num_of_gpus * self.batch_size * self.samples_per_iter)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched


    def get_val_batches(self, total_batches=-1):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1 or self.args.few_shot==False:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="val")
        # self.dataset.set_augmentation(augment_images=augment_images)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched


    def get_test_batches(self, total_batches=-1):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1 or self.args.few_shot==False:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name='test')
        # self.dataset.set_augmentation(augment_images=augment_images)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

