# -*- coding: utf8 -*-
#

import pathlib

DATA_PATH = pathlib.Path('.').parent.joinpath('data')

TRAIN_PATH = DATA_PATH.joinpath('full.train.conllu')
DEV_PATH = DATA_PATH.joinpath('full.valid.conllu')
TEST_PATH = DATA_PATH.joinpath('full.test.conllu')

MODEL_PATH = DATA_PATH.joinpath('savepoint')
if not MODEL_PATH.exists():
    MODEL_PATH.mkdir(exist_ok=True)
