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
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ##########################################################################
    # Setup logging
    ##########################################################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    ##########################################################################
    # Detecting last checkpoint.
    ##########################################################################
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ##########################################################################
    # Load tokenizer
    ##########################################################################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ##########################################################################
    # Load model
    ##########################################################################
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32),
    )

    ##########################################################################
    # Load dataset
    ##########################################################################
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
    )

    ##########################################################################
    # Preprocess dataset
    ##########################################################################
    # This is a simplified version - actual implementation would include
    # more preprocessing steps for instruction fine-tuning
    
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples["text"])
        return result

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    ##########################################################################
    # Initialize Trainer
    ##########################################################################
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    ##########################################################################
    # Training
    ##########################################################################
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_datasets["train"])

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    ##########################################################################
    # Evaluation
    ##########################################################################
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(tokenized_datasets["validation"])
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##########################################################################
    # Push to hub
    ##########################################################################
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()