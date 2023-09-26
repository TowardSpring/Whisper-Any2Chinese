# Whisper Config
# You should change the following config parameters according to your own environment
MODEL_CONFIG_FILE_ROOT_PATH = '/home/towardspring/.cache/huggingface/hub/models--openai--whisper-small/snapshots/e34e8ae444c29815eca53e11383ea13b2e362eb0'

DATASET_NAME_OR_PATH = "./data/ug_mini"
LANGUAGE = "uyghur"
LANGUAGE_ABBR = "ug" 

MODEL_NAME_OR_PATH = "/home/towardspring/.cache/huggingface/hub/models--openai--whisper-small/snapshots/e34e8ae444c29815eca53e11383ea13b2e362eb0"
TASK = "translate-chinese"
NEW_TOKENS = ['<|translate-chinese|>','<|ug|>']

OUTPUT_DIR = './trained_model/whisper_small'

TRANSLATED_CHINESE_TOKEN = "<|translate-chinese|>"
LANGUAGE_TOKEN = f'<|{LANGUAGE_ABBR}|>'