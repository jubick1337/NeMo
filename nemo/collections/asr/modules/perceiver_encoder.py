# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
from collections import OrderedDict

import torch

from torch import nn

from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, RelPositionalEncoding
from nemo.core import NeuralModule, Exportable
from nemo.core.classes.common import typecheck

from nemo.collections.nlp.modules.common.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.nlp.modules.common.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.nlp.modules.common.transformer.transformer_modules import AttentionBridge

__all__ = ["PerceiverEncoder"]

from nemo.core.neural_types import NeuralType, SpectrogramType, LengthsType, AcousticEncodedRepresentation


class PerceiverEncoder(NeuralModule, Exportable):
    def __init__(
            self,
            num_layers: int,
            hidden_size: int,
            inner_size: int,
            mask_future: bool = False,
            num_attention_heads: int = 1,
            attn_score_dropout: float = 0.0,
            attn_layer_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            hidden_act: str = "relu",
            pre_ln: bool = False,
            pre_ln_final_layer_norm: bool = True,
            hidden_steps: int = 32,
            hidden_init_method: str = "default",
            hidden_blocks: int = 2,
            proj_size: int = 256
    ):
        super().__init__()

        self._hidden_steps = hidden_steps
        self._hidden_init_method = hidden_init_method
        self._hidden_blocks = hidden_blocks

        if self._hidden_init_method == "default":
            self._hidden_init_method = "params"

        if self.hidden_init_method not in self.supported_init_methods:
            raise ValueError(
                "Unknown hidden_init_method = {hidden_init_method}, supported methods are {supported_init_methods}".format(
                    hidden_init_method=self.hidden_init_method, supported_init_methods=self.supported_init_methods,
                )
            )

        if self.hidden_init_method == "params":
            # learnable initial hidden values
            self.init_hidden = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(hidden_steps, proj_size)))
            self.init_cross_att = TransformerDecoder(
                num_layers=1,
                hidden_size=proj_size,
                inner_size=inner_size,
                num_attention_heads=num_attention_heads,
                attn_score_dropout=attn_score_dropout,
                attn_layer_dropout=attn_layer_dropout,
                ffn_dropout=ffn_dropout,
                hidden_act=hidden_act,
                pre_ln=pre_ln,
                pre_ln_final_layer_norm=pre_ln_final_layer_norm,
            )
            ### bug??
            self.init_cross_att.diagonal = None
        elif self.hidden_init_method == "bridge":
            # initialize latent with attention bridge
            self.att_bridge = AttentionBridge(hidden_size=hidden_size, k=hidden_steps, bridge_size=inner_size, )

        # cross-attention encoder
        layer = TransformerDecoder(
            num_layers=1,
            hidden_size=proj_size,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )
        layer.diagonal = None
        self.cross_att_layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(hidden_blocks)])

        # self-attention encoder
        layer = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=proj_size,
            inner_size=inner_size,
            mask_future=mask_future,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )
        self.self_att_layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(hidden_blocks)])

        self.pos_enc = RelPositionalEncoding(
            d_model=proj_size, dropout_rate=0.1, max_len=5000, xscale=True
        )

        self.feat_proj = nn.Linear(hidden_size, proj_size)

    @property
    def supported_init_methods(self):
        return ["params", "bridge"]

    @property
    def hidden_steps(self):
        return self._hidden_steps

    @property
    def hidden_blocks(self):
        return self._hidden_blocks

    @property
    def hidden_init_method(self):
        return self._hidden_init_method

    @staticmethod
    def make_pad_mask(seq_lens, max_time, device=None):
        """Make masking for padding."""
        bs = seq_lens.size(0)
        seq_range = torch.arange(0, max_time, dtype=torch.int32)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_time)
        seq_lens = seq_lens.type(seq_range_expand.dtype).to(seq_range_expand.device)
        seq_length_expand = seq_lens.unsqueeze(-1)
        mask = seq_range_expand < seq_length_expand

        if device:
            mask = mask.to(device)
        return mask

    @typecheck()
    def forward(self, audio_signal, length):
        audio_mask = self.make_pad_mask(length, length.max(), length.device)
        audio_signal = audio_signal.transpose(-1, -2)
        audio_signal = self.feat_proj(audio_signal)
        audio_signal, _ = self.pos_enc(audio_signal)
        # all hidden values are active
        hidden_mask = torch.ones(
            audio_signal.shape[0], self._hidden_steps, dtype=length.dtype, device=length.device
        )

        audio_mask = audio_mask.to(length.dtype)

        # initialize hidden state
        if self._hidden_init_method == "params":
            # initialize latent with learned parameters
            hidden_states = self.init_hidden.unsqueeze(0).expand(audio_signal.shape[0], -1, -1)
            hidden_states = self.init_cross_att(
                decoder_states=hidden_states,
                decoder_mask=hidden_mask,
                encoder_states=audio_signal,
                encoder_mask=audio_mask,
            )

        elif self._hidden_init_method == "bridge":
            # initialize latent with attention bridge
            hidden_states = self.att_bridge(hidden=audio_signal, hidden_mask=audio_mask, )

        # apply block (cross-attention, self-attention) multiple times
        # for block in range(self._hidden_blocks):
        for self_att, cross_att in zip(self.self_att_layers, self.cross_att_layers):
            residual = hidden_states

            # cross attention of hidden over encoder states
            hidden_states = cross_att(
                decoder_states=hidden_states,
                decoder_mask=hidden_mask,
                encoder_states=audio_signal,
                encoder_mask=audio_mask,
            )

            # self-attention over hidden
            hidden_states = self_att(encoder_states=hidden_states, encoder_mask=hidden_mask, )

            # residual connection
            hidden_states += residual

        hidden_mask = hidden_mask.sum(-1)
        hidden_states = hidden_states.transpose(-1, -2)
        return hidden_states, hidden_mask

    @property
    def input_types(self):
        return OrderedDict({
            'audio_signal': NeuralType(('B', 'D', 'T'), SpectrogramType()),
            'length': NeuralType(('B',), LengthsType())
        })

    @property
    def output_types(self):
        return OrderedDict({
            'outputs': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'encoded_lengths': NeuralType(('B',), LengthsType())
        })
