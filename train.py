#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The BigCode & HuggingFace Inc. teams. All rights reserved.
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
"""
Script to instruction fine-tune causal language models on a Hub dataset

Adapted from huggingface/transformers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""

import logging
import math
import os
import random
import sys
from itertools import chain

import datasets
import torch
import transformers
from config import DataArguments, ModelArguments, TrainingArguments
from datasets import load_dataset
from dialogues import get_dialogue_template, mask_user_labels, prepare_dialogue
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          default_data_collator, set_seed)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from utils import StarChatArgumentParser, hf_login

logger = logging.getLogger(__name__)


def main():
    parser = StarChatArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a YAML file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    # parse command line args and yaml file
    elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
    # parse command line args only
    else:
        model_args, data_args, training_args = parser.parse_args()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu},"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load dataset
    data = load_dataset(data_args.dataset_name, data_args.dataset_config_name)

    # Preprocess dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=data_args.max_seq_length, truncation=True)

    tokenized_data = data.map(preprocess_function, batched=True)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"] if training_args.do_train else None,
        eval_dataset=tokenized_data["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(tokenized_data["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(tokenized_data["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("***** Running evaluation *****")
        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(tokenized_data["validation"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(tokenized_data["validation"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
