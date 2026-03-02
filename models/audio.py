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

from .tdnn import FTDNN, TDNN, sinusoidal_embedding


@dataclass
class FlowMatchConfig(FairseqDataclass):
    channels: int = 1024
    sigma_min: float = 0.01
    time_embedding_dim: int = 128


@register_model("tdnn_flowmatch", dataclass=FlowMatchConfig)
class TDNNFlowMatchModel(BaseFairseqModel):
    def __init__(self, cfg: FlowMatchConfig):
        super().__init__()
        self.cfg = cfg
        # Using FTDNN with time embeddings for flow matching
        self.model = FTDNN(
            in_dim=cfg.channels,
            use_time_embedding=True,
            time_embedding_dim=cfg.time_embedding_dim,
        )

    @classmethod
    def build_model(cls, args, task):
        return cls(args)

    def forward(self, features, padding_mask):
        x_1 = features

        x_0 = torch.randn_like(x_1)
        t = torch.rand((x_0.shape[0]))

        # Generate sinusoidal time embeddings
        t_emb = sinusoidal_embedding(t, self.cfg.time_embedding_dim)

        t = t[:, None, None]
        x_t = (1 - (1 - self.cfg.sigma_min) * t) * x_0 + t * x_1

        u_t = x_1 - (1 - self.cfg.sigma_min) * x_0

        # Pass through TDNN layers with time embeddings
        x = x_t

        model_output = self.model(x, t_emb)

        loss = torch.pow(model_output - u_t, 2).mean()

        return {
            "losses": {"cfm":loss},
        }


@register_model("tdnn_simple_flowmatch", dataclass=FlowMatchConfig)
class TDNNSimpleFlowMatchModel(BaseFairseqModel):
    def __init__(self, cfg: FlowMatchConfig):
        super().__init__()
        self.cfg = cfg
        # Using a stack of TDNN layers with time embeddings for flow matching
        self.layers = nn.ModuleList(
            [
                TDNN(
                    input_dim=cfg.channels,
                    output_dim=512,
                    context_size=5,
                    padding=2,
                    use_time_embedding=True,
                    time_embedding_dim=cfg.time_embedding_dim,
                ),
                TDNN(
                    input_dim=512,
                    output_dim=512,
                    context_size=3,
                    padding=1,
                    use_time_embedding=True,
                    time_embedding_dim=cfg.time_embedding_dim,
                ),
                TDNN(
                    input_dim=512,
                    output_dim=cfg.channels,
                    context_size=1,
                    use_time_embedding=True,
                    time_embedding_dim=cfg.time_embedding_dim,
                ),
            ]
        )

    @classmethod
    def build_model(cls, args, task):
        return cls(args)

    def forward(self, features, padding_mask):
        x_1 = features

        x_0 = torch.randn_like(x_1)
        t = torch.rand((x_0.shape[0]))

        # Generate sinusoidal time embeddings
        t_emb = sinusoidal_embedding(t, self.cfg.time_embedding_dim)

        t = t[:, None, None]
        x_t = (1 - (1 - self.cfg.sigma_min) * t) * x_0 + t * x_1

        u_t = x_1 - (1 - self.cfg.sigma_min) * x_0

        # Pass through TDNN layers with time embeddings
        x = x_t
        for layer in self.layers:
            x = layer(x, t_emb)
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        model_output = x

        loss = torch.pow(model_output - u_t, 2).mean()

        return {
            "loss": loss,
        }
