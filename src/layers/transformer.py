import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

from src.layers.scale_mix import ScalarMix


class TransformerEmbedding(nn.Module):
    r"""
    A module that directly utilizes the pretrained models in `transformers`_ to produce BERT representations.
    While mainly tailored to provide input preparation and post-processing for the BERT model,
    it is also compatible with other pretrained language models like XLNet, RoBERTa and ELECTRA, etc.
    Args:
        model (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of BERT layers to use. If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        stride (int):
            A sequence longer than max length will be splitted into several small pieces
            with a window size of ``stride``. Default: 10.
        pooling (str):
            Pooling way to get from token piece embeddings to token embedding.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        pad_index (int):
            The index of the padding token in BERT vocabulary. Default: 0.
        dropout (float):
            The dropout ratio of BERT layers. Default: 0. This value will be passed into the :class:`ScalarMix` layer.
        requires_grad (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, model: str, n_layers: int, n_out=0, stride=256, pooling='mean', pad_index=0, dropout=0,
                 requires_grad=True):
        super().__init__()

        from transformers import AutoConfig
        self.transformer = AutoModel.from_pretrained(
            model,
            config=AutoConfig.from_pretrained(
                model,
                output_hidden_states=True
            )
        )

        self.transformer = self.transformer.requires_grad_(requires_grad)

        self.model = model
        self.n_layers = n_layers or self.transformer.config.num_hidden_layers
        self.hidden_size = self.transformer.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.stride = stride
        self.pooling = pooling
        self.pad_index = pad_index
        self.dropout = dropout
        self.requires_grad = requires_grad
        self.max_len = int(max(0, self.transformer.config.max_position_embeddings) or 1e12) - 2

        self.scalar_mix = ScalarMix(self.n_layers, dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"stride={self.stride}, pooling={self.pooling}, pad_index={self.pad_index}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, subwords):
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_subwords]
        subwords = pad_sequence(subwords[mask].split(lens.tolist()), batch_first=True, padding_value=self.pad_index)
        bert_mask = pad_sequence(mask[mask].split(lens.tolist()), batch_first=True, padding_value=self.pad_index)

        # return the hidden states of all layers
        bert = self.transformer(subwords[:, :self.max_len], attention_mask=bert_mask[:, :self.max_len].float())[-1]
        # [n_layers, batch_size, max_len, hidden_size]
        bert = bert[-self.n_layers:]
        # [batch_size, max_len, hidden_size]
        bert = self.scalar_mix(bert)
        # [batch_size, n_subwords, hidden_size]
        for i in range(self.stride,
                       (subwords.shape[1] - self.max_len + self.stride - 1) // self.stride * self.stride + 1,
                       self.stride):
            part = self.transformer(
                subwords[:, i:i + self.max_len],
                attention_mask=bert_mask[:, i:i + self.max_len].float()
            )[-1]
            bert = torch.cat((bert, self.scalar_mix(part[-self.n_layers:])[:, self.max_len - self.stride:]), 1)

        # [batch_size, seq_len]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size).masked_scatter_(mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            embed = embed[:, :, 0]
        elif self.pooling == 'last':
            embed = embed.gather(2, (bert_lens - 1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(
                2)
        else:
            embed = embed.sum(2) / bert_lens.unsqueeze(-1)
        embed = self.projection(embed)

        return embed

# TODO: 改成这个试试
class TransformerWordEmbedding(torch.nn.Module):
    def __init__(self, transformer: str, ):
        super(TransformerWordEmbedding, self).__init__()
        self.encoder = AutoModel.from_pretrained(transformer)
        self.dropout = nn.Dropout(p=self.encoder.config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, attention_mask, word_index, word_attention_mask):
        hidden_states = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output = hidden_states[-1]
        sequence_output = self.dropout(sequence_output)

        bert_output = sequence_output[:, 1:-1, :]
        # bert_output = torch.cat([bert_output[:, :1, :], torch.gather(
        #     bert_output[:, 1:, :], dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, bert_output.size(-1))
        # )], dim=1)
        # 这个方式是采用首字作为整个词的embedding
        bert_output = torch.gather(bert_output, dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, bert_output.size(-1)))
        return bert_output, word_attention_mask


