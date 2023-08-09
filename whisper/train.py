#对whisper模型进行二次训练，添加<|translate-chinese|>标记，实现Any-to-Chinese的翻译功能

# import os 
# import sys
# root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# root = os.path.join(root, 'whisper')
# sys.path.append(root)
from tokenizer import get_tokenizer

def test_tokenizer():
    gpt2_tokenizer = get_tokenizer(multilingual=False)
    multilingual_tokenizer = get_tokenizer(multilingual=True)

    text = "다람쥐 헌 쳇바퀴에 타고파"
    text = "这是一个测试。"
    text = 'دۇربۇن ئۇمۇ ئىككى گۇرۇپپا كۆپۈنگۈ لېنزىدىن تەشكىل تاپقان'
    gpt2_tokens = gpt2_tokenizer.encode(text)
    multilingual_tokens = multilingual_tokenizer.encode(text)

    assert gpt2_tokenizer.decode(gpt2_tokens) == text
    assert multilingual_tokenizer.decode(multilingual_tokens) == text
    assert len(gpt2_tokens) > len(multilingual_tokens)
    print(gpt2_tokens,text)





if __name__ == "__main__":
    test_tokenizer()