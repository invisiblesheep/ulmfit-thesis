#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

from fastai import *
from fastai.text import * 
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split


parser = ArgumentParser()
parser.add_argument("--folder", help="source folder of text data")
parser.add_argument("--vocab_size", help="vocab size", default=30000)
parser.add_argument("--encoder", help="path to encoder")

ARGS = parser.parse_args()

FOLDER = ARGS.folder
VOCAB_SIZE = ARGS.vocab_size
ENCODER = ARGS.encoder

SAVE_PATH = FOLDER

data_processed = False
if os.path.isfile(Path(FOLDER) / 'data_clas.pkl'):
    data_processed = True
else: 
    if not os.path.exists(Path(FOLDER) / 'ulmfit_finetuned'):
        os.mkdir(Path(FOLDER) / 'ulmfit_finetuned')
    
    vocab_save_path = Path(FOLDER) / f'ulmfit_finetuned'/ f'vocab_{VOCAB_SIZE}'
    SAVE_PATH = vocab_save_path
    if not os.path.exists(vocab_save_path):
        os.mkdir(vocab_save_path)
    
if not data_processed:
    df_clas = pd.read_csv(Path(FOLDER) / 'clas_data.csv')
    spprocessor = SPProcessor(
            sp_model= f'../vocabs/wsl.{VOCAB_SIZE}.model',
            sp_vocab= f'../vocabs/ulmfit.{VOCAB_SIZE}.vocab'
        )
    data_train, data_valid = train_test_split(df_clas, test_size=0.2, random_state=42, stratify=df_clas['label'])
    data_train['is_valid'] = False
    data_valid['is_valid'] = True
    data_clas_combined = pd.concat([data_train, data_valid])
    data = (TextList.from_df(data_clas_combined, processor=spprocessor)
                 .split_from_df(col='is_valid')
                 .label_from_df(cols='label')
                 .databunch(bs=200))
    
    data.save(f'{SAVE_PATH}/data_clas.pkl')
    print(f'Data saved to {SAVE_PATH}/data_clas.pkl')
data = load_data(SAVE_PATH, 'data_clas.pkl')
learn = text_classifier_learner(data, AWD_LSTM)
learn.load_encoder(ENCODER)
learn.freeze_to(-1)
learn.fit_one_cycle(1,1e-2)
learn.save(Path(SAVE_PATH) / 'clas_0')
#learn.load('clas_0')
# unfreeze relu layer and fit another epoch
learn.freeze_to(-2)
learn.fit_one_cycle(1,1e-2)
learn.save(Path(SAVE_PATH) / 'clas_1')
# unfreeze all layers and execute final fit
#learn.load('clas_1')
learn.unfreeze()
learn.fit_one_cycle(3, slice(3e-04,3e-03))
learn.fit_one_cycle(8, slice(2e-05,2e-04))
learn.save(Path(SAVE_PATH) / 'clas_2')
learn.export(Path(SAVE_PATH) / 'ulmfit_classifier')
