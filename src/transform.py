# -*- coding: utf8 -*-
#
import json
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import dataset, dataloader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import TRAIN_PATH, DATA_PATH, DEV_PATH


class CoNLLSentence(object):
    def __init__(self, lines: List[str]):
        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i - 1] = line
            else:
                self.annotations[len(self.values)] = line
                self.values.append(value)
        self.values = list(zip(*self.values))

    def __repr__(self):
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values))}}
        return '\n'.join(merged.values()) + '\n'

    @property
    def words(self):
        return self.values[1]

    @property
    def tags(self):
        return self.values[3]

    def get_labels(self):
        sequence = self.values[8]
        labels = [[None] * (len(sequence) + 1) for _ in range(len(sequence) + 1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edge, label = pair.split(':')
                    labels[i][int(edge)] = label
        return labels

    def __len__(self):
        return len(self.words)


def get_labels():
    if DATA_PATH.joinpath('label_map.json').exists():
        with open(DATA_PATH.joinpath('label_map.json'), 'r') as f:
            return json.loads(f.read())
    label_map = {'[PAD]': 0}

    def _i(path):
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
        i, start, sentences = 0, 0, []
        for line in tqdm(lines, desc='get labels'):
            if not line:
                sentences.append(CoNLLSentence(lines[start:i]))
                start = i + 1
            i += 1

        for s in sentences:
            for line in s.get_labels():
                for label in line:
                    if label is not None:
                        label_map.setdefault(label, len(label_map))

    _i(TRAIN_PATH)
    _i(DEV_PATH)
    with open(DATA_PATH.joinpath('label_map.json'), 'w') as f:
        f.write(json.dumps(label_map, ensure_ascii=False, indent=2))
    return label_map


def get_tags():
    if DATA_PATH.joinpath('tag_map.json').exists():
        with open(DATA_PATH.joinpath('tag_map.json'), 'r') as f:
            return json.loads(f.read())

    tags_map = {'[PAD]': 0}

    def _i(path):
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
        i, start, sentences = 0, 0, []
        for line in tqdm(lines, desc='get tags'):
            if not line:
                sentences.append(CoNLLSentence(lines[start:i]))
                start = i + 1
            i += 1

        for s in sentences:
            for tag in s.tags:
                tags_map.setdefault(tag, len(tags_map))
    _i(TRAIN_PATH)
    _i(DEV_PATH)
    with open(DATA_PATH.joinpath('tag_map.json'), 'w') as f:
        f.write(json.dumps(tags_map, ensure_ascii=False, indent=2))
    return tags_map


def encoder_texts(texts: List[List[str]], tokenizer):
    # 统计句子中最大的词长度
    fix_len = max([max([len(word) for word in text]) for text in texts])

    matrix = []
    for text in texts:
        vector = []

        # text = [tokenizer.cls_token, *text, tokenizer.sep_token]
        text = [tokenizer.cls_token, *text]
        input_ids = tokenizer.batch_encode_plus(
            text,
            add_special_tokens=False,
        )['input_ids']

        for _input_ids in input_ids:
            # 修复例如: texts = [['\ue5f1\ue5f1\ue5f1\ue5f1']] 这种情况
            _input_ids = _input_ids or [tokenizer.unk_token_id]
            vector.append(_input_ids + (fix_len - len(_input_ids)) * [tokenizer.pad_token_id])
        matrix.append(torch.tensor(vector, dtype=torch.long))
    return pad_sequence(matrix, batch_first=True)


class SDPTransform(dataset.Dataset):
    def __init__(self, path: str, transformer: str, device: torch.device = 'cpu'):
        super(SDPTransform, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(transformer) if isinstance(transformer, str) else transformer
        self.labels = get_labels()
        self.tags = get_tags()
        self.sentences = []

        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
        i, start = 0, 0
        for line in tqdm(lines, desc='transform'):
            if not line:
                self.sentences.append(CoNLLSentence(lines[start:i]))
                start = i + 1
            i += 1

        # 统计下
        # l = {}
        # for sentence in self.sentences:
        #     ll = len(sentence) // 10
        #     l.setdefault(ll, 0)
        #     l[ll] += 1
        # print(l)
        self.sentences = sorted([i for i in self.sentences if len(i) < 100], key=lambda x: len(x))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]

    def collate_fn(self, batch: List[CoNLLSentence]):
        subwords = encoder_texts(texts=[i.words for i in batch], tokenizer=self.tokenizer)
        tags = []
        labels = []
        for _batch in batch:
            tag = [0, ]
            for _tag in _batch.tags:
                tag.append(self.tags[_tag])
            tags.append(tag)

            label = []
            for line in _batch.get_labels():
                label.append([])
                for _label in line:
                    label[-1].append(self.labels[_label] if _label is not None else -1)
            labels.append(label)

        tags_max_len = max([len(i) for i in tags])
        labels_max_len = max([len(i) for i in labels])
        tags_matrix = torch.zeros(len(batch), tags_max_len, dtype=torch.long)
        for index, tag in enumerate(tags):
            tags_matrix[index, :len(tag)] = torch.tensor(tag)
        labels_matrix = torch.zeros(len(batch), labels_max_len, labels_max_len, dtype=torch.long)
        for index, label in enumerate(labels):
            label_len = len(label)
            labels_matrix[index, :label_len, :label_len] = torch.tensor(label)
        return subwords.to(self.device), tags_matrix.to(self.device), labels_matrix.to(self.device)

    def to_dataloader(self, batch_size, shuffle):
        return dataloader.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)


if __name__ == '__main__':
    for subwords, tags, labels in SDPTransform(
            path=TRAIN_PATH,
            transformer='hfl/chinese-electra-180g-small-discriminator'
    ).to_dataloader(batch_size=32, shuffle=False):
        assert (subwords.size(1) == tags.size(1) == labels.size(1) == labels.size(2))
