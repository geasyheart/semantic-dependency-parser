# -*- coding: utf8 -*-
#
import math
from typing import Optional, Union

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, set_seed

from src.config import MODEL_PATH
from src.metric import ChartMetric
from src.model import SemanticDependencyModel
from src.transform import get_labels, SDPTransform
from src.utils import logger


class SemanticDependencyParser(object):
    def __init__(self):
        self.model: Optional[SemanticDependencyModel, None] = None
        self.tokenizer = None
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.labels = get_labels()

    def build_model(self, transformer):
        self.model = SemanticDependencyModel(
            transformer=transformer,
            n_labels=len(self.labels),
        )
        self.model.to(self.device)
        logger.info(self.model)
        return self.model

    def build_tokenizer(self, pretrained_model_name: str):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def build_optimizer(
            self,
            warmup_steps: Union[float, int],
            num_training_steps: int,
            lr=1e-5, weight_decay=0.01,
    ):
        """
        https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/trainer.py#L232
        :param warmup_steps:
        :param num_training_steps:
        :param lr:
        :param weight_decay:
        :return:
        """
        if warmup_steps <= 1:
            warmup_steps = int(num_training_steps * warmup_steps)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def build_dataloader(self, path, transformer, batch_size, shuffle):
        return SDPTransform(
            path=path,
            transformer=transformer,
            device=self.device
        ).to_dataloader(batch_size=batch_size, shuffle=shuffle)

    def fit(self, train_path, dev_path, epoch=100, lr=1e-3, pretrained_model_name=None, batch_size=32,
            warmup_steps=0.1):
        set_seed(seed=123231)

        self.build_tokenizer(pretrained_model_name=pretrained_model_name)

        train_dataloader = self.build_dataloader(
            path=train_path,
            transformer=self.tokenizer,
            batch_size=batch_size,
            shuffle=True
        )
        dev_dataloader = self.build_dataloader(
            path=dev_path,
            transformer=self.tokenizer,
            batch_size=batch_size,
            shuffle=False
        )

        self.build_model(transformer=pretrained_model_name)

        optimizer, scheduler = self.build_optimizer(
            warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * epoch,
            lr=lr
        )
        return self.fit_loop(train_dataloader, dev_dataloader, epoch=epoch, optimizer=optimizer,
                             scheduler=scheduler)

    def fit_loop(self, train, dev, epoch, optimizer, scheduler):
        # loss
        min_train_loss, min_dev_loss = math.inf, math.inf
        # metric
        max_dev_metric = 0

        for _epoch in range(1, epoch + 1):
            train_loss = self.fit_dataloader(
                train=train,
                optimizer=optimizer,
                scheduler=scheduler
            )
            if train_loss < min_train_loss:
                logger.info(f'Epoch:{_epoch} save min train loss:{train_loss} model')
                min_train_loss = train_loss
                self.save_weights(
                    save_path=str(MODEL_PATH.joinpath(f'train_loss_{train_loss:.4e}.pt'))
                )

            dev_loss, dev_metric = self.evaluate_dataloader(dev)

            if dev_loss < min_dev_loss:
                logger.info(f'Epoch:{_epoch} save min dev loss:{dev_loss} model')
                min_dev_loss = dev_loss
                self.save_weights(
                    save_path=str(MODEL_PATH.joinpath(f'dev_loss_{dev_loss:.4e}.pt'))
                )

            if dev_metric > max_dev_metric:
                logger.info(f'Epoch:{_epoch} save max dev metric:{dev_metric.score} model')
                max_dev_metric = dev_metric
                self.save_weights(
                    save_path=str(MODEL_PATH.joinpath(f'dev_metric_{dev_metric.score:.4e}.pt'))
                )

            logger.info(
                f'Epoch:{_epoch} lr: {scheduler.get_last_lr()[0]:.4e} train loss: {train_loss} ' + \
                f'dev loss: {dev_loss} ' + \
                f'dev metric: {dev_metric}'
            )

    def fit_dataloader(self, train, optimizer, scheduler):
        self.model.train()
        total_loss = 0.

        for data in tqdm(train, desc='fit_dataloader'):
            subwords, tags, labels = data
            word_mask = subwords.ne(self.tokenizer.pad_token_id)
            mask = word_mask if len(subwords.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(subwords)
            loss = self.model.loss(s_edge, s_label, labels, mask)
            total_loss += loss.item()
            loss.backward()

            self._step(optimizer=optimizer, scheduler=scheduler)
        total_loss /= len(train)
        return total_loss

    @torch.no_grad()
    def evaluate_dataloader(self, dev):
        self.model.eval()

        total_loss, metric = 0, ChartMetric()

        for data in tqdm(dev, desc='evaluate_dataloader'):
            subwords, tags, labels = data
            word_mask = subwords.ne(self.tokenizer.pad_token_id)
            mask = word_mask if len(subwords.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(subwords)
            loss = self.model.loss(s_edge, s_label, labels, mask)
            total_loss += loss.item()

            label_preds = self.model.decode(s_edge, s_label)
            # if not label_preds.eq(-1).all():
            #     print('debug')
            metric(label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
        total_loss /= len(dev)

        return total_loss, metric

    loaded = False

    def load(self, pretrained_model_name, model_path: str, device='cpu'):
        self.device = torch.device(device)
        if not self.loaded:
            self.build_tokenizer(pretrained_model_name=pretrained_model_name)
            self.build_model(transformer=pretrained_model_name)
            self.load_weights(save_path=model_path)
            self.loaded = True

    def _step(self, optimizer, scheduler):
        #
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    def save_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model.module.state_dict(), save_path)

    def load_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            self.model.load_state_dict(torch.load(save_path))
        else:
            self.model.module.load_state_dict(torch.load(save_path))
