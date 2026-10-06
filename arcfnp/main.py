import argparse
from .config import load_config
from .data import YeastMF, music_emotion, mirflickr, Music_style
from .experiment import PASEModel, cross_validation, _fmt
DATASETS = {'YeastMF': YeastMF, 'music_emotion': music_emotion, 'mirflickr': mirflickr, 'Music_style': Music_style}

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--nfold', type=int, default=10)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    return parser

def main():
    args = build_arg_parser().parse_args()
    configs = load_config(args.cfg)
    dataset_cls = DATASETS[configs['dataset_name']]
    model = PASEModel(configs)
    dataset = dataset_cls(configs=configs, nfold=args.nfold)
    val_metrics, _ = cross_validation(model, dataset, nfold=args.nfold, random_state=configs['rand_seed'], quiet_mode=args.quiet, save_model=args.save_model)
    vals_last = {}
    for key in val_metrics:
        mean_v = float(val_metrics[key].value()[0])
        std_v = float(val_metrics[key].value()[1])
        vals_last[key] = mean_v
    ap = vals_last.get('AveragePrecision')
    ham = vals_last.get('HammingLoss')
    one = vals_last.get('OneError')
    cov = vals_last.get('Coverage')
    rank = vals_last.get('RankingLoss')
if __name__ == '__main__':
    main()
