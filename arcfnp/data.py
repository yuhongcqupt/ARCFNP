import math
import os
import numpy as np
import torch
import torch.utils.data as data
import scipy.io as scio
from sklearn.model_selection import KFold

class DatasetLoader(data.Dataset):

    def __init__(self, X, y, data_inds=None, batch_size=128, shuffle=False):
        super(DatasetLoader, self).__init__()
        self.X = X
        self.y = y
        self.data_inds = data_inds
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.index = torch.randperm(self._len())
        else:
            self.index = torch.arange(0, self._len(), dtype=torch.int)
        self.start = 0
        return self

    def __next__(self):
        if self.start < self._len():
            self.end = min(self.start + self.batch_size, self._len())
            X = self.X[self.data_inds[self.index[self.start:self.end]]]
            y = self.y[self.data_inds[self.index[self.start:self.end]]]
            self.start = self.end
            if X.dim() == 1:
                X = X.unsqueeze(0)
                y = y.unsqueeze(0)
            return (X, y)
        else:
            raise StopIteration

    def _len(self):
        return len(self.data_inds)

    def __len__(self):
        return math.ceil(self._len() / self.batch_size)

class Dataset:

    def __init__(self, datadir='./Datasets', configs={}, nfold=10):
        self.datadir = datadir
        self.datafile = os.path.join(datadir, self.name(), self.name() + '.mat')
        self.dtype = configs['dtype']
        self.data_standardizing = configs['data_standardizing']
        self.eps = configs['eps']
        self._data(nfold, configs['shuffle'], configs['rand_seed'])
        self.feat_dim = self.X.size(1)
        self.num_class = self.y.size(1)
        self.train_dataloader = DatasetLoader(self.X, self.y_partial, batch_size=configs['train_batch_size'], shuffle=configs['shuffle'])
        self.test_dataloader = DatasetLoader(self.X, self.y, batch_size=configs['test_batch_size'], shuffle=False)
        self.val_dataloader = DatasetLoader(self.X_val, self.y_val, data_inds=np.arange(self.X_val.size(0)), batch_size=configs['test_batch_size'], shuffle=False)

    def name(self):
        return self.__class__.__name__

    def size(self):
        return self.train_dataloader._len()

    def data_cv_splitter(self, fold, nfold=10, shuffle=False, random_state=None):
        train_inds_file = self._get_path(fold, 'train', nfold, shuffle, random_state)
        test_inds_file = self._get_path(fold, 'test', nfold, shuffle, random_state)
        if not os.path.exists(train_inds_file):
            self.dataset_split(nfold, shuffle, random_state)
        data = scio.loadmat(train_inds_file)
        self.train_dataloader.data_inds = data['index'][0]
        data = scio.loadmat(test_inds_file)
        self.test_dataloader.data_inds = data['index'][0]

    def dataset_split(self, nfold=10, shuffle=False, random_state=None):
        spliter = KFold(n_splits=nfold, shuffle=shuffle, random_state=random_state)
        count = 1
        for train_inds, test_inds in spliter.split(self.X):
            train_inds_file = self._get_path(count, 'train', nfold, shuffle, random_state)
            test_inds_file = self._get_path(count, 'test', nfold, shuffle, random_state)
            scio.savemat(train_inds_file, {'index': train_inds})
            scio.savemat(test_inds_file, {'index': test_inds})
            count += 1

    def _data(self, nfold=10, shuffle=False, random_state=None):
        self._data_loading()
        self._data_preprocess()
        self._val_data_split(nfold, shuffle, random_state)

    def _data_loading(self):
        data = scio.loadmat(self.datafile)
        self.X = torch.from_numpy(data['X'].astype(float)).type(self.dtype)
        self.y = torch.from_numpy(data['Y']).type(self.dtype)
        self.y_partial = torch.from_numpy(data['Y_partial']).type(self.dtype)
        self.alpha = self._noise_rate_est(self.y, self.y_partial)

    def _noise_rate_est(self, y, y_partial):
        y_neg_inds = y == 0
        noisy_pos_inds = y_partial == 1
        num1 = torch.sum(y_neg_inds & noisy_pos_inds, dim=0, keepdim=True)
        num2 = torch.sum(y_neg_inds, dim=0, keepdim=True)
        return num1 / (num2 + self.eps)

    def _data_preprocess(self):
        if self.data_standardizing:
            max_X = torch.max(self.X, dim=0, keepdim=True)[0]
            min_X = torch.min(self.X, dim=0, keepdim=True)[0]
            self.X = (self.X - min_X) / (max_X - min_X + self.eps)

    def _val_data_split(self, nfold=10, shuffle=False, random_state=None):
        split_config = '_{:d}_{:d}_{:d}_'.format(shuffle, random_state, nfold)
        traintest_inds_file = os.path.join(self.datadir, self.name(), self.name() + split_config + '_traintest.mat')
        val_inds_file = os.path.join(self.datadir, self.name(), self.name() + split_config + '_val.mat')
        if not os.path.exists(traintest_inds_file):
            spliter = KFold(n_splits=nfold, shuffle=shuffle, random_state=random_state)
            for traintest_inds, val_inds in spliter.split(self.X):
                scio.savemat(traintest_inds_file, {'index': traintest_inds})
                scio.savemat(val_inds_file, {'index': val_inds})
                break
        data = scio.loadmat(val_inds_file)
        self.X_val = self.X[data['index'][0]]
        self.y_val = self.y[data['index'][0]]
        self.y_partial_val = self.y_partial[data['index'][0]]
        data = scio.loadmat(traintest_inds_file)
        self.X = self.X[data['index'][0]]
        self.y = self.y[data['index'][0]]
        self.y_partial = self.y_partial[data['index'][0]]

    def _get_path(self, fold, data_type='train', nfold=10, shuffle=False, random_state=None):
        split_config = '_{:d}_{:d}_{:d}_'.format(shuffle, random_state, nfold)
        fileName = os.path.join(self.datadir, self.name(), self.name() + split_config + '_{:s}_cv{:d}.mat'.format(data_type, fold))
        return fileName

class SyntheticDataset(Dataset):

    def __init__(self, datadir='./Datasets', configs={}, nfold=10):
        self.datadir = datadir
        self.datafile = os.path.join(datadir, self.name(), self.name() + '.mat')
        self.dtype = configs['dtype']
        self.data_standardizing = configs['data_standardizing']
        self.eps = configs['eps']
        self.noise_level = configs['noise_level']
        self.label_filter = configs['label_filter']
        self._data(nfold, configs['shuffle'], configs['rand_seed'])
        self.feat_dim = self.X.size(1)
        self.num_class = self.y.size(1)
        self.train_dataloader = DatasetLoader(self.X, self.y_partial, batch_size=configs['train_batch_size'], shuffle=configs['shuffle'])
        self.test_dataloader = DatasetLoader(self.X, self.y, batch_size=configs['test_batch_size'], shuffle=False)
        self.val_dataloader = DatasetLoader(self.X_val, self.y_val, data_inds=np.arange(self.X_val.size(0)), batch_size=configs['test_batch_size'], shuffle=False)

    def _data(self, nfold=10, shuffle=False, random_state=None):
        self._data_loading(random_state)
        self._data_preprocess()
        self._val_data_split(nfold, shuffle, random_state)

    def _data_loading(self, random_state=None):
        if self.label_filter:
            syn_datafile = os.path.join(self.datadir, self.name(), self.name() + '_' + self.noise_level + '_{:d}_filtered.mat'.format(random_state))
        else:
            syn_datafile = os.path.join(self.datadir, self.name(), self.name() + '_' + self.noise_level + '_{:d}.mat'.format(random_state))
        if not os.path.exists(syn_datafile):
            data = scio.loadmat(self.datafile)
            X = torch.from_numpy(data['X'].astype(float)).type(self.dtype)
            y = torch.from_numpy(data['Y']).type(self.dtype)
            y_partial, alpha = self._generate_noisy_labels(y)
            scio.savemat(syn_datafile, {'X': X.numpy().astype(np.float64), 'Y': y.numpy().astype(np.float64), 'Y_partial': y_partial.numpy().astype(np.float64), 'alpha': alpha.numpy().astype(np.float64)})
        data = scio.loadmat(syn_datafile)
        self.X = torch.from_numpy(data['X'].astype(float)).type(self.dtype)
        self.y = torch.from_numpy(data['Y']).type(self.dtype)
        self.y_partial = torch.from_numpy(data['Y_partial']).type(self.dtype)
        self.alpha = torch.from_numpy(data['alpha']).type(self.dtype)

    def _filter_rare_labels(self, X, y, keep_num=15):
        if y.size(1) > keep_num:
            y_sum = torch.sum(y, dim=0)
            _, keep_inds = torch.topk(y_sum, keep_num)
            y = torch.index_select(y, 1, keep_inds)
            y_sum = torch.sum(y, dim=1)
            keep_inds = y_sum > 0
            X = X[keep_inds]
            y = y[keep_inds]
        return (X, y)

    def _generate_noisy_labels(self, y):
        noisy_labels = y.clone()
        if self.noise_level == 'high_level':
            alpha = 0.1 * torch.randint(5, 9, (1, y.size(1)), dtype=y.dtype)
        else:
            alpha = 0.1 * torch.randint(2, 6, (1, y.size(1)), dtype=y.dtype)
        rand_mat = torch.rand(y.size())
        mask = rand_mat < alpha
        noisy_labels[mask & (y == 0)] = 1
        return (noisy_labels, alpha)

    def _val_data_split(self, nfold=10, shuffle=False, random_state=None):
        if self.label_filter:
            split_config = '_filtered_{:d}_{:d}_{:d}_'.format(shuffle, random_state, nfold)
        else:
            split_config = '_{:d}_{:d}_{:d}_'.format(shuffle, random_state, nfold)
        traintest_inds_file = os.path.join(self.datadir, self.name(), self.name() + split_config + '_traintest.mat')
        val_inds_file = os.path.join(self.datadir, self.name(), self.name() + split_config + '_val.mat')
        if not os.path.exists(traintest_inds_file):
            spliter = KFold(n_splits=nfold, shuffle=shuffle, random_state=random_state)
            for traintest_inds, val_inds in spliter.split(self.X):
                scio.savemat(traintest_inds_file, {'index': traintest_inds})
                scio.savemat(val_inds_file, {'index': val_inds})
                break
        data = scio.loadmat(val_inds_file)
        self.X_val = self.X[data['index'][0]]
        self.y_val = self.y[data['index'][0]]
        self.y_partial_val = self.y_partial[data['index'][0]]
        data = scio.loadmat(traintest_inds_file)
        self.X = self.X[data['index'][0]]
        self.y = self.y[data['index'][0]]
        self.y_partial = self.y_partial[data['index'][0]]

    def _get_path(self, fold, data_type='train', nfold=5, shuffle=False, random_state=None):
        if self.label_filter:
            split_config = '_filtered_{:d}_{:d}_{:d}_'.format(shuffle, random_state, nfold)
        else:
            split_config = '_{:d}_{:d}_{:d}_'.format(shuffle, random_state, nfold)
        fileName = os.path.join(self.datadir, self.name(), self.name() + split_config + '_{:s}_cv{:d}.mat'.format(data_type, fold))
        return fileName

class YeastMF(Dataset):
    pass

class music_emotion(Dataset):
    pass

class mirflickr(Dataset):
    pass

class Music_style(Dataset):
    pass
