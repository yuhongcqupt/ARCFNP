import copy
import json
import torch
import scipy.io as scio
DEFAULT_CONFIG = {'dataset_name': 'YeastMF', 'noise_level': 'low' + '_level', 'n_rules': 4, 'max_epoch': 200, 'in_features': None, 'num_classes': None, 'train_batch_size': 64, 'shuffle': True, 'data_standardizing': True, 'lr': 0.0003, 'weight_decay': 0.0001, 'start_epoch': 0, 'drop_ratio': 0.0, 'lr_meta': 0.001, 'display_freq': 10, 'save_checkpoint_path': 'PASEModel/cv_exp/yeast/checkpoint', 'rand_seed': 2023, 'test_batch_size': 64, 'label_metrics': ['HammingLoss'], 'score_metrics': ['OneError', 'Coverage', 'RankingLoss', 'AveragePrecision'], 'number_metrics': 5, 'threshold': 0.5, 'dtype': torch.float32, 'eps': 1e-05, 'use_gpu': True, 'device': torch.device(type='cuda'), 'label_filter': True}

def merge_config(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge_config(dst[key], value)
        else:
            dst[key] = value
    return dst

def load_config(path_or_json=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path_or_json:
        text = path_or_json.strip()
        if text.startswith('{'):
            override = json.loads(text)
        else:
            with open(path_or_json, 'r', encoding='utf-8') as f:
                override = json.load(f)
        merge_config(cfg, override)
    if isinstance(cfg.get('device'), str):
        cfg['device'] = torch.device(cfg['device'])
    elif cfg.get('use_gpu') and torch.cuda.is_available():
        cfg['device'] = torch.device('cuda')
    else:
        cfg['device'] = torch.device('cpu')
    data = scio.loadmat('Datasets/' + cfg['dataset_name'] + '/' + cfg['dataset_name'] + '.mat')
    cfg['in_features'] = data['X'].shape[1]
    cfg['num_classes'] = data['Y'].shape[1]
    return cfg
