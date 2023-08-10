# secondary train whisper with new data

from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict, Dataset
from pprint import pprint
from transformers import WhisperFeatureExtractor
import os
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

from datasets import Audio
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from transformers import WhisperForConditionalGeneration,Seq2SeqTrainingArguments
from peft import prepare_model_for_int8_training,LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model,PeftModel, PeftConfig
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl,WhisperForConditionalGeneration, Seq2SeqTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from whisper_model.tokenizer import Tokenizer

# 参数配置
model_name_or_path = "openai/whisper-large-v2"
task = "transcribe"   # transcribe or translate or translate-chinese

dataset_name = "mozilla-foundation/common_voice_11_0"
language = "uyghur"
language_abbr = "ug" # Short hand code for the language we want to fine-tune

# 加载数据
common_voice = DatasetDict()
common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation")
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test")
pprint(common_voice)

item = common_voice['train'][0]
pprint(item)

features = common_voice.get("train").features
pprint(features)

common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
)
print(common_voice)


# 加载模型
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# 数据处理
pprint(common_voice["train"][0])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

pprint(common_voice["train"][0])

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=12)

common_voice["train"]


# 训练
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")


# model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
model = prepare_model_for_int8_training(model)

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()


training_args = Seq2SeqTrainingArguments(
    output_dir="reach-vb/test",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=100,
    evaluation_strategy="steps",
    save_steps = 1000,
    save_total_limit = 1,
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=100,
    max_steps=6000, # only for testing purposes, remove this from your final run :)
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)

# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()


trainer.save_model("model/whisper-large-v2-uygur-peft")




#nohup python /home/towardspring/projects/asr/faster_whisper_fineturning/whsiper_train.py > ./log/whisper_train_v2.log 2>&1 &