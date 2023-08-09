import gradio as gr
from whisper_test import WHISPER_TOOL
import shutil

import librosa
from datasets import Dataset
import torch

def run_whisper(input_language,task,record):

    if input_language == 'Uyghur/维吾尔语':
        input_language = 'ug'
    elif input_language == 'English/英语':
        input_language = 'en'
    elif input_language == 'Chinese/中文':
        input_language = 'zh'

    
    if task == 'Transcribe/转录':
        task = 'transcribe'
    elif task == 'Any to English/转录+翻译成英语':
        task = 'translate'
    elif task == 'Any to Chinese/转录+翻译中文':
        task = 'translate-chinese'

    text  = '未能实现转录。'
    print(f'input_language:{input_language},task:{task},record:{record}')
    print(f'type of input_language:{type(input_language)},type of task:{type(task)},type of record:{type(record)}')

    whisper_tool = WHISPER_TOOL()
    # 检查record是否为文件路径
    if isinstance(record,str):
        # 保存文件到./data/online/temp_audio.mp3
        shutil.copy(record,'./data/online/temp_audio.mp3')
        record_file = './data/online/temp_audio.mp3'
        audio_data, sample_rate = librosa.load(record_file)


        data = {'audio':[audio_data],'sampling_rate':[sample_rate]}
        dataset = Dataset.from_dict(data)
        text = whisper_tool.run_whisper(input_language,task,dataset)
        print("text:",text)

    # 检查是否是numpy类型
    elif isinstance(record,tuple):
        sample_rate = record[0]
        data = {'audio':[torch.from_numpy(record[-1])],'sampling_rate':[sample_rate]}
        dataset = Dataset.from_dict(data)
        text = whisper_tool.run_whisper(input_language,task,dataset)
        print("text:",text)
    if type(text) == list:
        text = '\n'.join(text)
        return text
    
    else:
        return text

with gr.Blocks() as demo:    
    gr.Markdown("# <center>Whisper - Any to Chinese</center>")
    gr.Markdown("### Please select the upload method for the audio/请选择音频上传方式")

    with gr.Tab('上传音频文件'):
        with gr.Row(): #并行显示，可开多列
            with gr.Column(): # 并列显示，可开多行
                gr.Markdown("### Please select the input language/请选择输入语种")
                drop1 = gr.Dropdown(["Uyghur/维吾尔语", "English/英语", "Chinese/中文"], 
                label="Input Languages/输入语种", info="(Will add more languages later!/陆续添加更多语种！)") #单选

                gr.Markdown("### Please select the task type/请选择任务类型")
                task = gr.Dropdown(["Transcribe/转录","Any to English/转录+翻译成英语", "Any to Chinese/转录+翻译中文"], 
                label="Task Type/任务类型", info="(Will add more task types!/陆续添加更多任务类型！)") #单选

                gr.Markdown("### Plese click the button to start recording/请点击开始录音")
                record1=gr.Audio(type="filepath")
                bottom2 = gr.Button(label="demo2")

            outputs=gr.Textbox(label='ASR Output/语音自动识别输出')

            

        bottom2.click(run_whisper, inputs=[drop1, task, record1], outputs=outputs) # 触发
    
    with gr.Tab('麦克风录音'):
        with gr.Row(): #并行显示，可开多列
            with gr.Column(): # 并列显示，可开多行
                gr.Markdown("### Please select the input language/请选择输入语种")
                drop1 = gr.Dropdown(["Uyghur/维吾尔语", "English/英语", "Chinese/中文"], 
                label="Input Languages/输入语种", info="(Will add more languages later!/陆续添加更多语种！)") #单选

                gr.Markdown("### Please select the task type/请选择任务类型")
                task = gr.Dropdown(["Transcribe/转录","Any to English/转录+翻译成英语", "Any to Chinese/转录+翻译中文"], 
                label="Task Type/任务类型", info="(Will add more task types!/陆续添加更多任务类型！)") #单选

                gr.Markdown("### Plese click the button to start recording/请点击开始录音")
                record1 = gr.Microphone(source="microphone") #录音

                bottom1 = gr.Button(label="demo1")

            out1 = gr.Textbox(label='ASR Output/语音自动识别输出')
        bottom1.click(run_whisper, inputs=[drop1, task, record1], outputs=out1) # 触发

    







demo.launch(share=True,server_port=6672) # 启动





# PID: , nohup python3 app_run.py > ./log/app_run.log 2>&1 &