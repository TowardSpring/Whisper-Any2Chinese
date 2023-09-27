# Whisper Config
# You should change the following config parameters according to your own environment
MODEL_CONFIG_FILE_ROOT_PATH = '/home/towardspring/.cache/huggingface/hub/models--openai--whisper-large-v2/snapshots/94ee83d319e97b04ac0d46235a8097e4865968b3'

DATASET_NAME_OR_PATH = "./data/en_mini"
LANGUAGE = "english"
LANGUAGE_ABBR = "ug" 

MODEL_NAME_OR_PATH = "/home/towardspring/.cache/huggingface/hub/models--openai--whisper-large-v2/snapshots/94ee83d319e97b04ac0d46235a8097e4865968b3"
TASK = "translate-chinese"
NEW_TOKENS = ['<|translate-chinese|>','<|en|>']

OUTPUT_DIR = './trained_model/whisper_large_v2'

TRANSLATED_CHINESE_TOKEN = "<|translate-chinese|>"
LANGUAGE_TOKEN = f'<|{LANGUAGE_ABBR}|>'