#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import ast
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, auto
import hydra
from hydra.core.config_store import ConfigStore
import logging
import math
import os
from omegaconf import OmegaConf
from typing import Optional
import sys

import editdistance
import torch
from npy_append_array import NpyAppendArray

from hydra.core.hydra_config import HydraConfig

from fairseq import checkpoint_utils, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.configs import FairseqDataclass, FairseqConfig
from fairseq.logging.meters import StopwatchMeter
from omegaconf import open_dict


logging.root.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO add num sample steps and other options

@dataclass
class UnsupGenerateConfig(FairseqDataclass):
    fairseq: FairseqConfig = FairseqConfig()
    results_path: Optional[str] = field(
        default=None,
        metadata={"help": "where to store results"},
    )


def get_dataset_itr(cfg, task):
    return task.get_batch_iterator(
        dataset=task.dataset(cfg.fairseq.dataset.gen_subset),
        max_tokens=cfg.fairseq.dataset.max_tokens,
        max_sentences=cfg.fairseq.dataset.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=cfg.fairseq.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.fairseq.dataset.required_batch_size_multiple,
        num_shards=cfg.fairseq.dataset.num_shards,
        shard_id=cfg.fairseq.dataset.shard_id,
        num_workers=cfg.fairseq.dataset.num_workers,
        data_buffer_size=cfg.fairseq.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)




def prepare_result_files(cfg: UnsupGenerateConfig):

    l_f = open(os.join(cfg.results_path, f"{cfg.fairseq.dataset.gen_subset}.lengths"))
    npaa = NpyAppendArray(os.join(cfg.results_path, f"{cfg.fairseq.dataset.gen_subset}.npy"))

    return {
        "lengths":l_f,
        "npaa":npaa,
    }


def optimize_models(cfg: UnsupGenerateConfig, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.eval()
        if cfg.fairseq.common.fp16:
            model.half()
        if use_cuda:
            model.cuda()


GenResult = namedtuple(
    "GenResult",
    [
        "count",
        "gen_timer",
        "num_feats",
        "num_sentences",
    ],
)


def generate(cfg: UnsupGenerateConfig, models, saved_cfg, use_cuda):
    task = tasks.setup_task(cfg.fairseq.task)
    task.load_dataset(cfg.fairseq.dataset.gen_subset, task_cfg=saved_cfg.task)
    # Set dictionary
    tgt_dict = task.target_dictionary
    logger.info(
        "| {} {} {} examples".format(
            cfg.fairseq.task.data,
            cfg.fairseq.dataset.gen_subset,
            len(task.dataset(cfg.fairseq.dataset.gen_subset)),
        )
    )
    # Load dataset (possibly sharded)
    itr = get_dataset_itr(cfg, task)
    # Initialize generator
    gen_timer = StopwatchMeter()

    num_sentences = 0
    if cfg.results_path is not None and not os.path.exists(cfg.results_path):
        os.makedirs(cfg.results_path)

    res_files = prepare_result_files(cfg)
    count = 0
    num_feats = 0

    gen_timer.start()

    start = 0
    end = len(itr)

    with progress_bar.build_progress_bar(cfg.fairseq.common, itr) as t:
        for i, sample in enumerate(t):
            if i < start or i >= end:
                continue

            if "net_input" not in sample:
                continue

            gen_vec, num_feats = gen(
                models, num_feats, sample, task, use_cuda
            )

            for i, padding_mask in enumerate(sample["padding_mask"]):
                length = (~padding_mask).sum()
                print(length, res_files["lengths"])
                res_files["npaa"].append(gen_vec[i, length].numpy())


            num_sentences += (
                sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
            )

    if res_files is not None:
        res_files["lengths"].close()

    gen_timer.stop(num_sentences)

    return GenResult(
        count,
        gen_timer,
        num_feats,
        num_sentences,
    )


def gen(models, num_feats, sample, task, use_cuda):
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    if "features" in sample["net_input"]:
        num_feats += (
            sample["net_input"]["features"].shape[0]
            * sample["net_input"]["features"].shape[1]
        )

    gen_vec = models.generate(**["net_input"])
    return gen_vec, num_feats


def main(cfg: UnsupGenerateConfig, model=None):
    if (
        cfg.fairseq.dataset.max_tokens is None
        and cfg.fairseq.dataset.batch_size is None
    ):
        cfg.fairseq.dataset.max_tokens = 1024000

    use_cuda = torch.cuda.is_available() and not cfg.fairseq.common.cpu

    task = tasks.setup_task(cfg.fairseq.task)

    overrides = ast.literal_eval(cfg.fairseq.common_eval.model_overrides)

    if model is None:
        # Load ensemble
        logger.info("| loading model(s) from {}".format(cfg.fairseq.common_eval.path))
        # models, saved_cfg = checkpoint_utils.load_model_ensemble(
        #     cfg.fairseq.common_eval.path.split("\\"),
        #     arg_overrides=overrides,
        #     task=task,
        #     suffix=cfg.fairseq.checkpoint.checkpoint_suffix,
        #     strict=(cfg.fairseq.checkpoint.checkpoint_shard_count == 1),
        #     num_shards=cfg.fairseq.checkpoint.checkpoint_shard_count,
        # )
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [cfg.fairseq.common_eval.path],
            arg_overrides={}
        )
        optimize_models(cfg, use_cuda, models)
    else:
        models = [model]
        saved_cfg = cfg.fairseq

    with open_dict(saved_cfg.task):
        saved_cfg.task.shuffle = False
        saved_cfg.task.sort_by_length = False

    gen_result = generate(cfg, models, saved_cfg, use_cuda)

    logger.info(
        "| Processed {} sentences ({} tokens) in {:.1f}s ({:.2f}"
        " sentences/s, {:.2f} tokens/s)".format(
            gen_result.num_sentences,
            gen_result.gen_timer.n,
            gen_result.gen_timer.sum,
            gen_result.num_sentences / gen_result.gen_timer.sum,
            1.0 / gen_result.gen_timer.avg,
        )
    )


@hydra.main(
    config_path=os.path.join("./", "config"), config_name="generate"
)
def hydra_main(cfg):
    with open_dict(cfg):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        cfg.job_logging_cfg = OmegaConf.to_container(
            HydraConfig.get().job_logging, resolve=True
        )

    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=False, enum_to_str=False)
    )
    OmegaConf.set_struct(cfg, True)
    logger.info(cfg)

    utils.import_user_module(cfg.fairseq.common)

    _, score = main(cfg)

    if cfg.is_ax:
        return score, None
    return score


def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=UnsupGenerateConfig)
    hydra_main()


if __name__ == "__main__":
    cli_main()

