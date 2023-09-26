# test whsiper model

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
import pandas as pd
import evaluate
from transformers import WhisperForConditionalGeneration,Seq2SeqTrainingArguments
from peft import prepare_model_for_int8_training,LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model,PeftModel, PeftConfig
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl,WhisperForConditionalGeneration, Seq2SeqTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import librosa
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

class WHISPER_TOOL():
    def __init__(self):
        # 参数配置
        model_name_or_path = "openai/whisper-small"
        task = "translate-chinese"

        dataset_name = "mozilla-foundation/common_voice_11_0"
        language = "uyghur"
        language_abbr = "ug" # Short hand code for the language we want to fine-tune

        # 加载模型

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

        # 评估
        peft_model_id = "reach-vb/train_small/checkpoint-139800" # Use the same model ID as before.
        peft_config = PeftConfig.from_pretrained(peft_model_id)
        
        self.model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model.config.use_cache = True

    def prepare_dataset(self,batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

    def run_whisper(self, input_language,task,record):
        target_rate = 16000
        data_item = self.feature_extractor(record[0]["audio"], sampling_rate=target_rate).input_features[0]
        # data_label = self.tokenizer(record[0]["sentence"]).input_ids
        # record = record.map(self.prepare_dataset, num_proc=12)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=input_language, task=task)
        # normalizer = BasicTextNormalizer()

        self.model.eval()

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    self.model.generate(
                        input_features=torch.from_numpy(data_item).unsqueeze(0).to('cuda'),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                # labels = data_label.cpu().numpy()
                # labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
                decoded_preds = self.processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                # decoded_labels = self.processor.tokenizer.batch_decode(data_label, skip_special_tokens=True)

                # print(f'decoded_preds: {decoded_preds}, decoded_labels: {record[0]["sentence"]}')
                # 将预测结果保存到'/home/towardspring/projects/asr/faster_whisper_fineturning/data/pred.txt'文件中
                # with open('/home/towardspring/projects/asr/faster_whisper_fineturning/data/pred.txt', 'w') as f:
                #     f.write(decoded_preds)

                del generated_tokens
            gc.collect()

        return decoded_preds

if __name__ == "__main__":
    whisper_tool = WHISPER_TOOL()
    input_language = "uyghur"
    # target_sample_rate = 16000
    task = "translate-chinese"
    # 导入common_voice_ug_26049139.mp3
    
    
    file_path = "/home/towardspring/hdd2/dataset/asr/ug/data/train/common_voice_ug_32868881.mp3"  # 替换为你的MP3文件路径
    file_path = "/home/towardspring/hdd2/dataset/asr/ug/data/test/common_voice_ug_32925037.mp3"

    # 导入MP3文件并进行采样
    audio_data, sample_rate = librosa.load(file_path)


    data = {'audio':[audio_data],'sampling_rate':[sample_rate],'sentence':["ئۇ ئۆمۈر بويى ئۆز خىزمىتىگە سادىق بولۇپ، ئادىمىيلىك بۇرچىنى ئادا قىلىپ كەلگەنىدى."]}
    dataset = Dataset.from_dict(data)
    print(data)
    print("数据集大小:", len(dataset))
    # print("示例数据:", dataset)


    # print("shape of audio_data:",dataset.shape)
    # 从MP3文件中提取文本
    text = whisper_tool.run_whisper(input_language,task,dataset)
    print("text:",text)