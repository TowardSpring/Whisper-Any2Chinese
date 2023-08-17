# secondary train whisper with new data

from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict, Dataset
from pprint import pprint
import os
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from peft import prepare_model_for_int8_training,LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model,PeftModel, PeftConfig
from transformers_model import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl,WhisperForConditionalGeneration, Seq2SeqTrainer
from transformers_model.trainer_utils import PREFIX_CHECKPOINT_DIR
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers_model.models.whisper.english_normalizer import BasicTextNormalizer
from tokenizer import Tokenizer
import argparse
import yaml
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union
from __init__ import load_model
from audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)

from decoding import DecodingOptions, DecodingResult
from timing import add_word_timestamps
from tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from utils import (
    exact_div,
    format_timestamp,
    get_writer,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)


def run():
    # run whisper model
    from __init__ import available_models
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="./config/en_to_chinese.yaml", help="the path to the task yaml configuration file")
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # parse configs
    model_name_or_path = configs["model"]
    model_dir = configs["model_dir"]
    use_peft = configs["use_peft"]
    input_peft_dir = configs["input_peft_dir"]
    output_peft_dir = configs["output_peft_dir"]

    dataset_dir = configs["dataset_dir"]
    task = configs["task"]
    language = configs["language"]
    language_abbr = configs["language_abbr"]

    train_args = configs["train_args"]
    per_device_train_batch_size = train_args["per_device_train_batch_size"]
    gradient_accumulation_steps = train_args["gradient_accumulation_steps"]
    learning_rate = train_args["learning_rate"]
    warmup_steps = train_args["warmup_steps"]
    num_train_epochs = train_args["num_train_epochs"]
    evaluation_strategy = train_args["evaluation_strategy"]
    save_steps = train_args["save_steps"]
    save_total_limit = train_args["save_total_limit"]
    fp16 = train_args["fp16"]
    per_device_eval_batch_size = train_args["per_device_eval_batch_size"]
    generation_max_length = train_args["generation_max_length"]
    logging_steps = train_args["logging_steps"]
    max_steps = train_args["max_steps"]
    remove_unused_columns = train_args["remove_unused_columns"]
    label_names = train_args["label_names"]

    # load dataset
    dataset = load_dataset("audiofolder", data_dir="/home/towardspring/hdd2/dataset/asr/whisper_en",streaming=True)
    train_data = dataset["train"]
    test_data = dataset["test"]

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if use_peft and input_peft_dir is None:
        # 训练新的peft模型
        model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")

        # model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
        model = prepare_model_for_int8_training(model)

    elif use_peft and input_peft_dir:
        # 加载上一次训练获得的peft模型，输出peft模型
        peft_config = PeftConfig.from_pretrained(input_peft_dir)
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, input_peft_dir)
        model.config.use_cache = True

    else:
        # 加载原始模型，训练原始模型
        model = load_model(model_name_or_path, device=device, download_root=model_dir)

    # process dataset
    train_data = train_data.map()


    


    # train


    
    # eval


    # save model
    # trainer.save_model("model/whisper-large-v2-uygur-peft")



if __name__ == "__main__":
    run()









#nohup python /home/towardspring/projects/asr/faster_whisper_fineturning/whsiper_train.py > ./log/whisper_train_v2.log 2>&1 &