# -*- coding: utf8 -*-
#
from src.config import TRAIN_PATH, DEV_PATH
from src.semantic_dependency_parser import SemanticDependencyParser

SemanticDependencyParser().fit(
    train_path=TRAIN_PATH,
    dev_path=DEV_PATH,
    pretrained_model_name='hfl/chinese-electra-180g-small-discriminator',
    lr=1e-4,
    batch_size=32,
    epoch=300
)
