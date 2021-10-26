# -*- coding: utf8 -*-
#
from src.config import MODEL_PATH, DEV_PATH
from src.semantic_dependency_parser import SemanticDependencyParser

m = SemanticDependencyParser()
m.load(
    pretrained_model_name='hfl/chinese-electra-180g-small-discriminator',
    model_path=str(MODEL_PATH.joinpath('dev_metric_8.1216e-02.pt')),
    device='cuda'
)


dev = m.build_dataloader(
    DEV_PATH,
    transformer=m.tokenizer,
    batch_size=2,
    shuffle=False
)

m.evaluate_dataloader(dev)