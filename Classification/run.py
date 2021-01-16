# coding: UTF-8
import time, os
import torch
import numpy as np
from tqdm import tqdm
from train_eval import train_PTM
from utils import build_iterator_PTM, get_time_dif
from importlib import import_module
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Choose Classification Model and Task')
parser.add_argument('--model', type=str, required=True, help='choose a model: BERT, RoBERTa, XLNet, ELECTRA, ALBERT.')
parser.add_argument('--dataset', type=str, required=True, help='choose a dataset.')
parser.add_argument('--max_vocab_size', type=int, required=True, default=10000, help='the number of max vocab size.')
parser.add_argument('--pad_size', type=int, required=True, default=512, help='the number of padding size.')
args = parser.parse_args()

MAX_VOCAB_SIZE = args.max_vocab_size
pad_size = args.pad_size
UNK, PAD, CLS, SEP = '<UNK>', '<PAD>', '[CLS]', '[SEP]'  


def build_dataset_PTM(config):
    
    def load_dataset(path, pad_siz):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


if __name__ == '__main__':
    
    # Configuration
    dataset = 'datasets'  
    model_name = args.model 
   
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    
    # Seed for getting same result
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True 
    
    start_time = time.time()
    print("Loading data...")
    
    # Loading dataset
    train_data, dev_data, test_data = build_dataset_PTM(config)
    train_iter = build_iterator_PTM(train_data, config)
    dev_iter = build_iterator_PTM(dev_data, config)
    test_iter = build_iterator_PTM(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # Train
    model = x.Model(config).to(config.device)
    train_PTM(config, model, train_iter, dev_iter, test_iter)