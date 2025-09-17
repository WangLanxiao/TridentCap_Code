import numpy as np
import torch
import torch.nn as nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.utils.initialization import trunc_normal_
from ..layers.create_act import get_act_layer
from .build import S_E_REGISTRY
from ..layers.multihead_attention import MultiHeadAttention
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
import collections

__all__ = ["BaseSEMerge"]

@S_E_REGISTRY.register()



class BaseSEMerge(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            vocab_size: int,
            word_embedding_size: int,
            word_vec_size: int,
            hidden_size: int,
            bidirectional=False,
            input_dropout_p=0,
            dropout_p=0,
            n_layers=1,
            rnn_type='lstm',
            style: list,
            **kwargs
        ):
        super(BaseSEMerge, self).__init__()
        self.style = style
        self.style_embedding = nn.Embedding(len(self.style), word_embedding_size)
        self.style_mlp = nn.Sequential(
                                 nn.Linear(word_embedding_size, word_embedding_size),
                                 nn.ReLU())
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU())
        # self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
        #                                          batch_first=True,
        #                                          bidirectional=bidirectional,
        #                                          dropout=dropout_p)
        self.att_lang = MultiHeadAttention(
                                      d_model=word_embedding_size , \
                                      d_k=word_embedding_size , \
                                      d_v=word_embedding_size , \
                                      num_head=8, \
                                      dropout=dropout_p
                                      )

        self.att_vis = MultiHeadAttention(
            d_model=word_embedding_size, \
            d_k=word_embedding_size, \
            d_v=word_embedding_size, \
            num_head=8, \
            dropout=dropout_p
        )

        self.input_dropout1 = nn.Dropout(input_dropout_p)
        # self.mlp1 = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
        #                          nn.ReLU())

        # self.rnn1 = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
        #                                          batch_first=True,
        #                                          bidirectional=bidirectional,
        #                                          dropout=dropout_p)

        self.num_dirs = 2 if bidirectional else 1

        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)

        # Text feature encoder (BERT)
        # self.vocab = load_vocab('/data1/wlx/project/202303c2p_TransferLearning/dataprocessing/bert_vocb')
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_proj = nn.Linear(768, word_embedding_size)
        # self.bert_output_layers = 4
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        #
        # for _name, _weight in self.bert.named_parameters():
        #     _weight.requires_grad = False
        # a=1
        # for v in self.bert.pooler.parameters():
        #     v.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "word_embedding_size": cfg.MODEL.S_E.WORD_EMBEDDING_SIZE,
            "word_vec_size": cfg.MODEL.S_E.WORD_VEC_SIZE,
            "hidden_size": cfg.MODEL.S_E.HIDDEN_SIZE,
            "bidirectional": cfg.MODEL.S_E.BIDIRECTIONAL,
            "input_dropout_p": cfg.MODEL.S_E.WORD_DROP_OUT,
            "dropout_p": cfg.MODEL.S_E.RNN_DROP_OUT,
            "n_layers": cfg.MODEL.S_E.RNN_NUM_LAYERS,
            "rnn_type": cfg.MODEL.S_E.RNN_TYPE,
            "style": cfg.DATALOADER.TYPE,
        }
        if cfg.MODEL.S_E.TOKEN_DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.S_E.TOKEN_DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        activation_name = (cfg.MODEL.S_E.TOKEN_ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None
            act_kwargs = {}
            if activation_name in {"elu", "celu"}:
                act_kwargs["alpha"] = cfg.MODEL.TOKEN_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act
        return kwargs

    def forward(self, batched_inputs, mode='token'):

        # input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # input_mask = [1] * len(input_ids)
        #
        # # Text features
        # word_feat, _ = self.bert(word_id, token_type_ids=None, attention_mask=word_mask)
        # word_feat = torch.stack(word_feat[-self.bert_output_layers:], 1).mean(1)
        # word_feat = self.bert_proj(word_feat)
        # word_feat = word_feat.permute(1, 0, 2)  # NxLxC -> LxNxC

        if mode == 'token':
            token_embedded = self.embedding(batched_inputs)  # (n, seq_len, word_embedding_size)
            token_embedded = self.embeddings_act(token_embedded)
            token_embedded = self.embeddings_dropout(token_embedded)
            return token_embedded
        else:
            ret={}
            if 'style' in mode:
                input_style = batched_inputs['STYLE_TOKEN']
                style_embedded=self.style_mlp(self.input_dropout(self.style_embedding(input_style)))
                input_sents = batched_inputs['G_TOKENS_IDS']  # b * 20
                img_feats = batched_inputs['ATT_FEATS']
                coco_embedded = self.embedding(input_sents)  # (n, seq_len, word_embedding_size)
                coco_embedded_dp = self.input_dropout(coco_embedded)  # (n, seq_len, word_embedding_size)
                coco_embedded_dp = self.mlp(coco_embedded_dp)  # (n, seq_len, word_vec_size)
                # coco_output, coco_hidden = self.rnn(coco_embedded_dp)

                # coco_hidden = self.att_lang(coco_embedded_dp.mean(-2,keepdims=True),  coco_embedded_dp,  coco_embedded_dp)

                img_update = self.att_vis(img_feats, img_feats,  img_feats)+img_feats
                coco_hidden = self.att_lang(img_feats, coco_embedded_dp, coco_embedded_dp)#+coco_embedded_dp.mean(-2,keepdims=True)

                ret.update({'SENT_FEATS': coco_hidden, 'STYLE_FEATS': style_embedded, 'ATT_FEATS': img_update})
            else:
                input_sents = batched_inputs['MID_G_TOKENS_IDS']  # b * 20
                coco_embedded = self.embedding(input_sents)  # (n, seq_len, word_embedding_size)
                coco_embedded_dp = self.input_dropout(coco_embedded)  # (n, seq_len, word_embedding_size)
                coco_embedded_dp = self.mlp(coco_embedded_dp)  # (n, seq_len, word_vec_size)
                coco_output, coco_hidden = self.rnn(coco_embedded_dp)

                input_style = batched_inputs['STYLE_TOKEN']-batched_inputs['STYLE_TOKEN']
                style_embedded = self.style_embedding(input_style)

                ret.update({'SENT_FEATS': coco_hidden[0], 'STYLE_FEATS': style_embedded})
                # ret.update({'SENT_FEATS': coco_hidden[0]})
            return ret

