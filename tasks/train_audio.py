# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from dataclasses import dataclass, field
import logging
import math
import os
from typing import Optional
import torch

from fairseq.logging import metrics
from fairseq.tasks import FairseqTask, register_task
from examples.wav2vec.unsupervised.data import (
    ExtractedFeaturesDataset,
    RandomInputDataset,
)

from fairseq.data import (
    Dictionary,
    data_utils,
    StripTokenDataset,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.distributed.utils import get_data_parallel_world_size
from omegaconf import MISSING

from examples.speech_recognition.kaldi.kaldi_decoder import (
    KaldiDecoder,
    KaldiDecoderConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class AudioConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory containing audio"}
    )
    max_length: Optional[int] = None
    sort_by_length: bool = field(
        default=True, metadata={"help": "sort examples by length of audio timesteps"}
    )
    shuffle: bool = field(default=True, metadata={"help": "shuffle examples"})


@register_task("train_audio", dataclass=AudioConfig)
class TrainAudio(FairseqTask):
    """ """

    cfg: AudioConfig

    def __init__(
        self,
        cfg: AudioConfig,
    ):
        super().__init__(cfg)

    @classmethod
    def setup_task(cls, cfg: AudioConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def optimizer_step(self, optimizer, model, update_num):
        if hasattr(model, "get_groups_for_update"):
            groups = model.get_groups_for_update(update_num)
            optimizer.step(groups={groups})
        else:
            optimizer.step()

    def valid_step(self, sample, model, criterion):
        res = model(
            **sample["net_input"],
        )

        sample_shape = sample["net_input"]["features"].shape
        nsentences = sample_size[0]
        ntokens = nsentences * sample_shape[1]
        sample_size = ntokens * sample_shape[2]

        loss = res["losses"]["cfm"]

        logging_output = {
            "loss": loss.item(),
            "sample_size": sample_size,
            "ntokens": ntokens,
            "nsentences": nsentences,
        }

        return 0, 1, logging_output

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        self.datasets[split] = ExtractedFeaturesDataset(
            path=data_path,
            split=split,
            min_length=3,
            max_length=task_cfg.max_length,
            shuffle=getattr(task_cfg, "shuffle", True),
            sort_by_length=task_cfg.sort_by_length,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg)

        return model
