# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    SamePad,
)
from torch.nn.modules import padding

from .tdnn import FTDNN, TDNN, sinusoidal_embedding


@dataclass
class FlowMatchConfig(FairseqDataclass):
    channels: int = 1024
    sigma_min: float = 0.01
    time_embedding_dim: int = 128


@register_model("tdnn_flowmatch_text", dataclass=FlowMatchConfig)
class TDNNFlowMatchModel(BaseFairseqModel):
    def __init__(self, cfg: FlowMatchConfig, target_dict):
        super().__init__()
        self.cfg = cfg
        self.zero_index = target_dict.index("<SIL>") if "<SIL>" in target_dict else 0

        output_size = len(target_dict)
        self.pad = target_dict.pad()
        self.eos = target_dict.eos()
        self.blank_index = 0
        assert self.blank_index != target_dict.unk()

        # Using FTDNN with time embeddings for flow matching
        self.model = FTDNN(
            in_dim=cfg.channels,
            use_time_embedding=True,
            time_embedding_dim=cfg.time_embedding_dim,
        )


        self.vocab_embed = torch.nn.Embedding(output_size, cfg.channels)

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task.target_dictionary)

    def forward(self, labels):
        with torch.no_grad():
            features = self.vocab_embed(labels)

        padding_mask = labels == self.pad

        x_1 = features
        x_1[padding_mask] = 0

        x_0 = torch.randn_like(x_1)
        x_0[padding_mask] = 0
        t = torch.rand((x_0.shape[0]), device=x_0.device)

        # Generate sinusoidal time embeddings
        t_emb = sinusoidal_embedding(t, self.cfg.time_embedding_dim)

        t = t[:, None, None]
        x_t = (1 - (1 - self.cfg.sigma_min) * t) * x_0 + t * x_1

        u_t = x_1 - (1 - self.cfg.sigma_min) * x_0

        # Pass through TDNN layers with time embeddings
        x = x_t

        model_output = self.model(x, t_emb)

        model_output[padding_mask] = 0

        loss = torch.pow(model_output - u_t, 2).sum() / (torch.sum(~padding_mask) * model_output.shape[-1])

        return {
            "losses": {"cfm":loss},
        }

    @torch.no_grad
    def generate(self, labels, n_timesteps=100):
        self.model = self.model.eval()

        padding_mask = labels == self.pad
        features = self.vocab_embed(labels)

        x_t = torch.randn_like(features)
        x_t[padding_mask] = 0

        h = 1 / n_timesteps
        t = 0

        for i in range(n_timesteps):
            t = torch.ones(x_t.shape[0], device=x_t.device)
            t_emb = sinusoidal_embedding(t, self.cfg.time_embedding_dim)
            v_t = self.model(x_t, t_emb) 

            x_t = x_t + h * v_t
            x_t[padding_mask] = 0

        self.model = self.model.train()

        return x_t

    @torch.no_grad
    def vec_to_tok(self, vec):
        dist = torch.cdist(vec, self.vocab_embed.weight)
        frame_labels = dist.argmin(dim=-1)
        hyp_labels = frame_labels.unique_consecutive()
        hyp_labels = hyp_labels[hyp_labels != 0]
        return hyp_labels
