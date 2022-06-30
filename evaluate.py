"""BYOL for Audio: Linear evaluation using MLP classifier.

This program performs downstream task evaluation by following steps.

1. First, all the data audio files in the downstream task dataset are converted to the representation embeddings.
2. With the audio embeddings and corresponding labels, the linear layer is trained by using an MLP classifier,
   which is basically compatible with sklearn implementation, and then test accuracy is calculated.
   For the leave-one-out CV task, this step repeats for all folds and averages the accuracy.
3. Repeat the previous step, and average the accuracy.

Notes:
- TorchMLPClassifier is used instead of sklearn's MLPClassifier for faster evaluation.

"""

from tqdm import tqdm
from collections import defaultdict
import logging
import re
from sklearn.preprocessing import StandardScaler
try:
    from utils.torch_mlp_clf import TorchMLPClassifier
except:
    raise Exception('Please follow Getting Started on the README.md to download and patch external modules.')


from byol_a.augmentations import PrecomputedNorm
from byol_a.dataset import WaveInLMSOutDataset
from byol_a.dataset import DataInLMSOutDataset
from byol_a.models import AudioNTT2020
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from byol_a.common import (os, sys, np, Path, random, torch, nn, DataLoader,
     get_logger, load_yaml_config, seed_everything, get_timestamp)


logging.basicConfig(level=logging.DEBUG)
device = torch.device('cpu')




def calc_norm_stats(cfg, data_src, n_stats=10000):
    """Calculates statistics of log-mel spectrogram features in a data source for normalization.

    Args:
        cfg: Configuration settings.
        data_src: Data source class object.
        n_stats: Maximum number of files to calculate statistics.
    """

    def data_for_stats(data_src):
        # use all files for LOO-CV (Leave One Out CV)
        if data_src.loocv:
            return data_src
        # use training samples only for non-LOOCV (train/eval/test) split.
        return data_src.subset([0])

    stats_data = data_src
    n_stats = min(n_stats, len(stats_data))
    logging.info(f'Calculating mean/std using random {n_stats} samples from training population {len(stats_data)} samples...')
    sample_idxes = np.random.choice(range(len(stats_data)), size=n_stats, replace=False)
    ds = WaveInLMSOutDataset(cfg, stats_data, labels=None, tfms=None)
    X = [ds[i] for i in tqdm(sample_idxes)]
    X = np.hstack(X)
    norm_stats = np.array([X.mean(), X.std()])
    logging.info(f'  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}')
    return norm_stats

def calc_norm_datastats(cfg, data_src, n_stats=10000):
    """Calculates statistics of log-mel spectrogram features in a data source for normalization.

    Args:
        cfg: Configuration settings.
        data_src: Data source class object.
        n_stats: Maximum number of files to calculate statistics.
    """

    stats_data = data_src
    n_stats = min(n_stats, len(stats_data))
    logging.info(f'Calculating mean/std using random {n_stats} samples from training population {len(stats_data)} samples...')
    sample_idxes = np.random.choice(range(len(stats_data)), size=n_stats, replace=False)
    ds = DataInLMSOutDataset(cfg, stats_data, labels=None, tfms=None)
    X = [ds[i] for i in tqdm(sample_idxes)]
    X = np.hstack(X)
    norm_stats = np.array([X.mean(), X.std()])
    logging.info(f'  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}')
    return norm_stats

def get_model_feature_d(model_filename):
    """Read number of fature_d in the filename."""

    r = re.search('d\d+', Path(model_filename).stem)
    if r is None:
        print(f'WARNING: feature dimension not found, falling back to 512-d: {model_filename}')
        d = 512
    else:
        d = int(r.group(0)[1:])
    return d


def get_embeddings(cfg, files, model, norm_stats):
    """Get representation embeddings of audio files, converted by the model.

    Args:
        cfg: Configuration settings.
        files: Audio files (.wav) to convert.
        model: Trained model that converts audio to embeddings.
        norm_stats: Mean & standard deviation calcurlated by calc_norm_stats().
    """

    ds = WaveInLMSOutDataset(cfg, files, labels=None, tfms=PrecomputedNorm(norm_stats))
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.bs, num_workers=cfg.num_workers,
                                     pin_memory=False, shuffle=False, drop_last=False)
    embs = []
    with torch.no_grad():
        for X in tqdm(dl):
            Y = model(X.to(device)).cpu().detach()
            embs.extend(Y.numpy())#extend list
    '''for X in tqdm(dl):
        Y = model(X.to(device)).cpu().detach()
        embs.extend(Y.numpy())'''
    return np.array(embs)

def get_pre(cfg, files, model, model1, norm_stats):
    """Get representation embeddings of audio files, converted by the model.

    Args:
        cfg: Configuration settings.
        files: Audio files (.wav) to convert.
        model: Trained model that converts audio to embeddings.
        norm_stats: Mean & standard deviation calcurlated by calc_norm_stats().
    """

    ds = DataInLMSOutDataset(cfg, files, labels=None, tfms=PrecomputedNorm(norm_stats))
    bs = len(files)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=cfg.num_workers,
                                     pin_memory=False, shuffle=False, drop_last=False)
    embs = []
    with torch.no_grad():
        for X in tqdm(dl):
            Y = model(X.to(device)).cpu().detach()
            scaler = StandardScaler()
            scaler.fit(Y)
            Y1 = scaler.transform(Y)
            Y1.dtype = np.float
            Y1_tensor = torch.from_numpy(Y1)
            Y_pre = model1(Y1_tensor.float()).cpu().detach()
            embs.extend(Y_pre.numpy())#extend list    ''
    return np.array(embs)


def _one_linear_eval(X, y, X_val, y_val, X_test, y_test, hidden_sizes, epochs, early_stopping, debug):
    """Perform a single run of linear evaluation."""

    if len(X_test.shape) > 2:
        X = X.mean(axis=1)
        X_test = X_test.mean(axis=1)
    scaler = StandardScaler()#define normalize
    scaler.fit(X)#cal mean and sq
    X = scaler.transform(X)#do normalize
    if X_val is not None:
        X_val = scaler.transform(X_val)

    clf_cls = TorchMLPClassifier
    clf = clf_cls(hidden_layer_sizes=hidden_sizes, max_iter=epochs,
                  early_stopping=early_stopping, batch_size= 32, debug=debug)
    modelsaved,_,_ = clf.fit(X, y, X_val=X_val, y_val=y_val)

    to_file = 'D:/byol/byol-a-master/checkpoints/mlp.pth'
    torch.save(modelsaved.state_dict(), to_file)

    X_test = scaler.transform(X_test)
    score = clf.score(X_test, y_test, device=torch.device('cpu'))
    return score

def linear_eval_single(folds, hidden_sizes=(), epochs=200, early_stopping=True, debug=False):
    """Evaluate a single train/test split with MLPClassifier.

    Args:
        folds: Holds dataset X, y as follows:
            0 = training set
            1 = validation set
            2 = test set
        hidden_sizes: MLP's hidden layer sizes
        epochs: Training epochs.
        early_stopping: Enables early stopping or not.
    """

    X, test_X, y, test_y = train_test_split(folds['X'], folds['y'], test_size=0.4, random_state=50)
    len_samples = int(len(test_X)/2)
    X_val, y_val = test_X[:len_samples-1], test_y[:len_samples-1]
    X_test, y_test = test_X[len_samples:], test_y[len_samples:]

    print(f'Training:{len(X)}, validation:{len(X_val)}, test:{len(X_test)} samples.')

    score = _one_linear_eval(X, y, X_val, y_val, X_test, y_test, hidden_sizes, epochs, early_stopping, debug)
    print('score:',f' {score:.6f}')

    return score


def prepare_linear_evaluation(weight_file, audio_path, unit_sec, n_stats=10000):
    """Prepare for linear evaluation.
    - Loads configuration settings, model, and downstream task data source.
    - Converts audio to representation embeddings.
    - Build folds for MLP classification.

    Returns:
        cfg: Configuration settings
        folds: Folds that hold X, y for all folds.
        loocv: True if the task is 10-folds LOO-CV, or False if it is a single fold (train/valid/test).
    """

    cfg = load_yaml_config('config.yaml')
    cfg.unit_sec = unit_sec
    #cfg.feature_d = get_model_feature_d(weight_file)
    print(cfg)

    model = AudioNTT2020(n_mels=cfg.n_mels, d=cfg.feature_d)
    model.load_weight(weight_file, device)

    seed_everything(42)
    labels = []
    #files = sorted(Path(audio_path).glob('*/*.wav'))
    files = sorted(Path(audio_path).glob('*/new/*.wav'))

    random.shuffle(files)

    for x in files:
        x = str(x)
        # print('file:',x)
        #label = x.split("\\")[-2]
        label = x.split("\\")[-3]

        label=torch.tensor(int(label)-1, dtype=torch.long)#label start from 0
        #label = torch.tensor(int(label) , dtype=torch.long)
        #label = list(str(label))
        #print(label)

        labels.append(label)


    # norm_stats
    norm_stats = calc_norm_stats(cfg, files, n_stats=n_stats)

    # embeddings
    model = model.to(device)
    model.eval()
    folds = defaultdict(list)

    folds['X'] = get_embeddings(cfg, files, model, norm_stats)
    folds['y'] = labels

    return cfg, folds


def do_eval(weight, audio_path, unit_sec=1.0, epochs=200, early_stopping=True, seed=42):
    """Main program of linear evaluation."""

    # run deterministically
    seed_everything(seed)

    cfg, folds = prepare_linear_evaluation(weight, audio_path, unit_sec)

    # run evaluation cycle

    score = linear_eval_single(folds, hidden_sizes=((128,)), epochs=epochs,
                early_stopping=early_stopping, debug=True)

    print('score:',score)


if __name__ == '__main__':

    audio_path = 'your path of ICML2013'


    weightpath = 'your path of byol-A weights'
    do_eval(weightpath,audio_path,0.95)
