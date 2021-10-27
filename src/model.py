# -*- coding: utf8 -*-
#
import torch
from torch import nn

from src.layers.affine import Biaffine
from src.layers.mlp import MLP
from src.layers.transformer import TransformerEmbedding


class SemanticDependencyModel(nn.Module):
    def __init__(self, transformer: str, n_labels: int, n_tags: int, n_edge_mlp=600, n_label_mlp=600):
        super(SemanticDependencyModel, self).__init__()

        self.encoder = TransformerEmbedding(model=transformer,
                                            n_layers=4,
                                            pooling='mean',
                                            pad_index=0,
                                            dropout=0.33,
                                            requires_grad=True)
        self.tag_embedding = nn.Embedding(num_embeddings=n_tags, embedding_dim=64)
        self.edge_mlp_d = MLP(n_in=self.encoder.n_out + self.tag_embedding.embedding_dim, n_out=n_edge_mlp, dropout=0.33)
        self.edge_mlp_h = MLP(n_in=self.encoder.n_out + self.tag_embedding.embedding_dim, n_out=n_edge_mlp, dropout=0.33)

        self.label_mlp_d = MLP(n_in=self.encoder.n_out + self.tag_embedding.embedding_dim, n_out=n_label_mlp, dropout=0.33)
        self.label_mlp_h = MLP(n_in=self.encoder.n_out + self.tag_embedding.embedding_dim, n_out=n_label_mlp, dropout=0.33)

        self.edge_attn = Biaffine(n_in=n_edge_mlp, n_out=2, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, subwords, tags):
        bert_out = self.encoder(subwords)
        tag_out = self.tag_embedding(tags)

        bert_out = torch.cat([bert_out, tag_out], dim=-1)
        edge_mlp_d = self.edge_mlp_d(bert_out)
        edge_mlp_h = self.edge_mlp_h(bert_out)

        label_mlp_d = self.label_mlp_d(bert_out)
        label_mlp_h = self.label_mlp_h(bert_out)

        s_edge = self.edge_attn(edge_mlp_d, edge_mlp_h).permute(0, 2, 3, 1)
        s_label = self.label_attn(label_mlp_d, label_mlp_h).permute(0, 2, 3, 1)
        return s_edge, s_label

    # @torch.no_grad()
    # def _loss2(self, s_edge, s_label, labels, mask, interpolation=0.1):
    #     criterion2 = nn.CrossEntropyLoss(reduction='none')
    #
    #     edge_mask = labels.ge(0) & mask
    #     edge_loss = criterion2(s_edge.reshape(-1, s_edge.size(-1)), edge_mask.reshape(-1).long())
    #     edge_loss = edge_loss[mask.reshape(-1)]
    #     edge_loss = edge_loss.sum() / edge_mask.size(0)
    #
    #     label_loss = criterion2(s_label.reshape(-1, s_label.size(-1)), labels.reshape(-1))
    #     label_loss = label_loss[edge_mask.reshape(-1)]
    #     label_loss = label_loss.sum() / label_loss.size(0)
    #
    #     print('loss2: ', interpolation * label_loss + (1 - interpolation) * edge_loss)

    def loss(self, s_edge, s_label, labels, mask, interpolation=0.1):
        # self._loss2(s_edge, s_label, labels, mask, interpolation)

        edge_mask = labels.ge(0) & mask
        edge_loss = self.criterion(s_edge[mask], edge_mask[mask].long())
        label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
        return interpolation * label_loss + (1 - interpolation) * edge_loss

    def decode(self, s_edge, s_label):
        return s_label.argmax(-1).masked_fill_(s_edge.argmax(-1).lt(1), -1)
