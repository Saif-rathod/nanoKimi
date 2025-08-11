# nanoKimi/tokenizer.py
import sentencepiece as spm
import os

class Tokenizer:
    def __init__(self, model_path: str = None, vocab_size=50257):
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
        else:
            self.sp = None
            self.vocab_size = vocab_size

    def train(self, input_file, model_prefix='spm', vocab_size=50257, model_type='bpe'):
        spm.SentencePieceTrainer.Train(
            f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type}'
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{model_prefix}.model')

    def encode(self, text: str):
        if self.sp:
            return self.sp.encode(text, out_type=int)
        else:
            return [ord(c) for c in text]

    def decode(self, ids):
        if self.sp:
            return self.sp.decode(ids)
        else:
            return ''.join(chr(i) for i in ids)
