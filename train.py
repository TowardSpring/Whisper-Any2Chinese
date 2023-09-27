# The same code with any2chinese_train_*.ipynb

from huggingface_hub import notebook_login, login
from datasets import load_dataset, DatasetDict
from pprint import pprint
from transformers import WhisperFeatureExtractor
import os
from tokenization_whisper import WhisperTokenizer
from transformers import WhisperProcessor

from datasets import Audio
import torch
from config.whisper_config import MODEL_CONFIG_FILE_ROOT_PATH, DATASET_NAME_OR_PATH, LANGUAGE, LANGUAGE_ABBR, MODEL_NAME_OR_PATH, TASK, NEW_TOKENS, OUTPUT_DIR
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from transformers import WhisperForConditionalGeneration,Seq2SeqTrainingArguments
from peft import prepare_model_for_int8_training,LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model,PeftModel, PeftConfig
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl,WhisperForConditionalGeneration, Seq2SeqTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR



#设定CUDA_LAUNCH_BLOCKING=1
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# load dataset
common_voice = DatasetDict()

common_voice["train"] = load_dataset(DATASET_NAME_OR_PATH, split="train")
common_voice["test"] = load_dataset(DATASET_NAME_OR_PATH, split="test")
common_voice["validation"] = load_dataset(DATASET_NAME_OR_PATH, split="validation")

print(common_voice)

# load model
print(f'MOEDL_NAME_OR_PATH:{MODEL_NAME_OR_PATH}')
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME_OR_PATH)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME_OR_PATH, language=LANGUAGE_ABBR, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME_OR_PATH, language=LANGUAGE_ABBR, task=TASK)
print('Model has been loaded.')

import json

tokenizer.add_tokens(NEW_TOKENS)
tokenizer.save_pretrained(MODEL_CONFIG_FILE_ROOT_PATH)   # Note: you need to change the path (config.whisper_config.py) to your own path

special_tokens_file = os.path.join(MODEL_CONFIG_FILE_ROOT_PATH, 'special_tokens_map.json')
with open(special_tokens_file, 'r') as f:
    special_tokens_map = json.load(f)
print(special_tokens_map)
additional_special_tokens = special_tokens_map['additional_special_tokens']
# print(len(additional_special_tokens))
for new_item in NEW_TOKENS:
    if new_item not in additional_special_tokens:
        additional_special_tokens.append(new_item)

# save to special_tokens_map.json
special_tokens_map['additional_special_tokens'] = additional_special_tokens
json_str = json.dumps(special_tokens_map,ensure_ascii=False)

with open(special_tokens_file, 'w') as f:
    f.write(json_str)

input_str = common_voice["train"][0]["chinese"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)


print(f"Input: {input_str}")
print(f"Decoded with special: {decoded_with_special}")
print(f"Decoded without special: {decoded_str}")
print(f"Are equal: {input_str == decoded_str}")

# 数据处理
pprint(common_voice["train"][1])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

pprint(common_voice["train"][1])


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["chinese"]).input_ids   # Note: you need to change the language to your own language
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=21)

common_voice["train"]


# Configure training parameters
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # print(f'------> batch device = {batch.device}')
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

# metric = evaluate.load("wer")

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH, load_in_8bit=True, device_map="auto")


# model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
model = prepare_model_for_int8_training(model)

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)

model.print_trainable_parameters()
# 迁移model到CPU训练
# model.to('cpu')

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit = 6,
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=100,
    # max_steps=6000, # only for testing purposes, remove this from your final run :)s
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
    eval_dataset=common_voice["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!







trainer.train()

trainer.save_model("trained_model/whisper-large-v2-en2chinese")




# nohup python train.py > ./log/train_whisper_large_v2_en2chinese.log 2>&1 &